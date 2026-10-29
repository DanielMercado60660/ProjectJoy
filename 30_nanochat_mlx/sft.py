"""
SFT on Grid Courier action traces (port of the shape of upstream scripts/chat_sft.py, with our
sandbox traces instead of SmolTalk/MMLU/GSM8K).

Each expert step becomes one single-turn conversation in upstream's chat format:
    <|bos|><|user_start|>{observation}<|user_end|><|assistant_start|>{action}<|assistant_end|>
Only the assistant tokens are supervised. Conversations are packed into rows of --seq-len.

    ../bin/gpu_run ../.venv/bin/python sft.py --base checkpoints/pilot/last --out checkpoints/sft
    ../bin/gpu_run ../.venv/bin/python sft.py --scratch --base checkpoints/pilot/last --out checkpoints/sft_scratch
"""
import argparse
import json
import math
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

import checkpoint as ckpt
from gpt import GPT, GPTConfig, loss_fn
from optim import build_optimizer, set_schedule
from sandbox import TRAIN_SEED0, expert_trace, make_env
from tokenizer import ChatTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))


def build_examples(tok, n_episodes, seed0):
    convs = []
    for s in range(seed0, seed0 + n_episodes):
        for obs, act in expert_trace(make_env(s, "train")):
            ids, mask = tok.render_conversation([{"role": "user", "content": obs}, {"role": "assistant", "content": act}])
            convs.append((ids, mask))
    return convs


def pack(convs, T, rng):
    """Greedy packing of shuffled conversations into rows of T+1 tokens. Returns x, y, m arrays."""
    order = rng.permutation(len(convs))
    rows, cur_i, cur_m = [], [], []
    for k in order:
        ids, mask = convs[k]
        if len(cur_i) + len(ids) > T + 1:
            rows.append((cur_i, cur_m))
            cur_i, cur_m = [], []
        cur_i += ids
        cur_m += mask
    if cur_i:
        rows.append((cur_i, cur_m))
    X = np.zeros((len(rows), T + 1), dtype=np.int32)
    M = np.zeros((len(rows), T + 1), dtype=np.float32)
    for r, (i, m) in enumerate(rows):
        X[r, :len(i)] = i
        M[r, :len(m)] = m
    # target at position t is token t+1; supervise it if token t+1 is an assistant token
    return X[:, :-1], X[:, 1:], M[:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/pilot/last")
    ap.add_argument("--scratch", action="store_true", help="same architecture, random init (no pretraining)")
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--episodes", type=int, default=8000)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--embedding-lr", type=float, default=0.3)
    ap.add_argument("--unembedding-lr", type=float, default=0.008)
    ap.add_argument("--init-lr-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=0, help="with --scratch: build config_for_depth(depth) instead of reading --base")
    a = ap.parse_args()
    mx.random.seed(a.seed)
    tok = ChatTokenizer.load()
    if a.scratch and a.depth:
        from gpt import config_for_depth
        cfg = config_for_depth(a.depth, tok.get_vocab_size())
    else:
        cfg = GPTConfig(**ckpt.load_meta(a.base)["config"])
    if a.scratch:
        model = GPT(cfg)
        model.init_weights(a.seed)
    else:
        model, _ = ckpt.load_model(a.base)
    t0 = time.time()
    train_convs = build_examples(tok, a.episodes, TRAIN_SEED0)
    val_convs = build_examples(tok, 300, TRAIN_SEED0 + 5_000_000)  # same distribution, unseen layouts
    n_tok = sum(len(c[0]) for c in train_convs)
    print(f"{len(train_convs)} step-examples, {n_tok/1e6:.2f}M tokens, built in {time.time()-t0:.1f}s", flush=True)
    rng = np.random.default_rng(a.seed)
    T, B = a.seq_len, a.batch
    rows_per_epoch = math.ceil(n_tok / T)
    n_steps = int(a.epochs * rows_per_epoch / B)
    # pretraining used lr_scale = sqrt(131072 / 2^19) = 0.5; SFT inherits it (upstream inherits lrs)
    opt, base_lrs = build_optimizer(model, a.matrix_lr, a.embedding_lr, a.unembedding_lr, weight_decay=0.0, lr_scale=0.5)
    lg = nn.value_and_grad(model, loss_fn)
    state = [model.state, opt.state]

    @lambda f: mx.compile(f, inputs=state, outputs=state)
    def step(x, y, m):
        loss, g = lg(model, x, y, m)
        opt.update(model, g)
        return loss

    vx, vy, vm = pack(val_convs, T, np.random.default_rng(123))

    def val_loss():
        tot, cnt = 0.0, 0.0
        for i in range(0, len(vx), B):
            ce = loss_fn(model, mx.array(vx[i:i + B]), mx.array(vy[i:i + B]), reduction="none")
            m = vm[i:i + B]
            tot += float((np.array(ce) * m).sum())
            cnt += float(m.sum())
        return tot / cnt

    out_dir = os.path.join(HERE, a.out)
    os.makedirs(out_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "log.jsonl"), "w")
    hist = [{"step": 0, "val_loss": val_loss()}]
    print("step 0 val (assistant-token) loss", hist[0]["val_loss"], flush=True)
    X = Y = M = None
    ptr = 0
    t0 = time.time()
    for it in range(n_steps):
        if X is None or ptr + B > len(X):
            X, Y, M = pack(train_convs, T, rng)
            ptr = 0
        frac = it / n_steps
        lr_mult = a.init_lr_frac * (1.0 if frac < 0.5 else (1 - frac) / 0.5)  # upstream: warmdown last 50% to 0
        set_schedule(opt, base_lrs, lr_mult, muon_momentum=0.95, muon_wd=0.0)
        loss = step(mx.array(X[ptr:ptr + B]), mx.array(Y[ptr:ptr + B]), mx.array(M[ptr:ptr + B]))
        mx.eval(loss, state)
        ptr += B
        rec = {"step": it + 1, "loss": loss.item(), "time": time.time() - t0}
        if (it + 1) % 100 == 0 or it + 1 == n_steps:
            rec["val_loss"] = val_loss()
            hist.append({"step": it + 1, "val_loss": rec["val_loss"]})
            print(f"step {it+1}/{n_steps} loss {rec['loss']:.4f} val {rec['val_loss']:.4f} {time.time()-t0:.0f}s", flush=True)
        log.write(json.dumps(rec) + "\n")
    ckpt.save(os.path.join(out_dir, "last"), model, None,
              {"step": n_steps, "history": hist, "args": vars(a), "base": a.base, "scratch": a.scratch,
               "train_examples": len(train_convs), "train_tokens": n_tok, "train_time_s": time.time() - t0})
    print("saved", out_dir, flush=True)


if __name__ == "__main__":
    main()
