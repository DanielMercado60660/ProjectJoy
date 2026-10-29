"""
Pretraining (port of upstream scripts/base_train.py) in MLX.

Example (pilot):
    ../bin/gpu_run ../.venv/bin/python train.py --depth 8 --total-batch 131072 --device-batch 32 \
        --num-iterations 3000 --time-budget-min 100 --out checkpoints/pilot
Resume: same command + --resume (picks up checkpoints/<out>/last).
"""
import argparse
import json
import math
import os
import signal
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_map

import checkpoint as ckpt
from data import TokenStream
from gpt import GPT, config_for_depth, loss_fn
from optim import build_optimizer, set_schedule

HERE = os.path.dirname(os.path.abspath(__file__))


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--aspect-ratio", type=int, default=64)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device-batch", type=int, default=32, help="rows per micro-batch")
    ap.add_argument("--total-batch", type=int, default=131072, help="tokens per optimizer step")
    ap.add_argument("--num-iterations", type=int, default=1000)
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop + save after this many minutes (0=off)")
    ap.add_argument("--matrix-lr", type=float, default=0.02)
    ap.add_argument("--embedding-lr", type=float, default=0.3)
    ap.add_argument("--unembedding-lr", type=float, default=0.008)
    ap.add_argument("--scalar-lr", type=float, default=0.5)
    ap.add_argument("--weight-decay", type=float, default=0.28)
    ap.add_argument("--warmup-steps", type=int, default=40)
    ap.add_argument("--warmdown-ratio", type=float, default=0.65)
    ap.add_argument("--final-lr-frac", type=float, default=0.05)
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    ap.add_argument("--no-compile", action="store_true")
    # efficiency flags (efficiency/REPORT.md). Only --ce-mode kernel is on by default (kept: parity
    # within bf16 reorder noise, +6% d6). --ce-mode ref reproduces the pilot's exact code path.
    ap.add_argument("--accum-mode", choices=["ref", "fast"], default="ref",
                    help="micro-batch accumulation loop (fast: on-device accumulation, one sync per step)")
    ap.add_argument("--fused-qkv", action="store_true", help="one GEMM for q/k/v (same parameters)")
    ap.add_argument("--ce-mode", choices=["ref", "kernel", "bf16cap"], default="kernel",
                    help="kernel (default since the efficiency pass): fused softcap+CE Metal kernel; "
                         "ref: the pilot's exact path; bf16cap: CHANGES TRAINING numerics")
    ap.add_argument("--muon-batched", action="store_true", help="stack equal-shape Muon matrices")
    ap.add_argument("--muon-addmm", action="store_true", help="Polar Express via mx.addmm")
    ap.add_argument("--ns-steps", type=int, default=5, help="Muon Polar Express iterations (<5 CHANGES TRAINING)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=16, help="val batches of device_batch x seq_len")
    ap.add_argument("--save-every-min", type=float, default=10)
    ap.add_argument("--sample-every", type=int, default=0, help="save a numbered checkpoint+samples every N steps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="checkpoints/pilot")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-at-step", type=int, default=-1, help="debug: exit (with save) at this step")
    ap.add_argument("--overfit-one-batch", action="store_true")
    return ap.parse_args(argv)


SAMPLE_PROMPTS = ["The capital of France is", "Once upon a time,", "The most important thing about water is",
                  "If yesterday was Friday, then tomorrow will be"]


def sample_text(model, tok, max_tokens=40):
    from engine import generate
    out = {}
    for p in SAMPLE_PROMPTS:
        ids = tok.encode(p, prepend_bos=True)
        gen = list(generate(model, ids, max_tokens=max_tokens, temperature=0.0))
        out[p] = tok.decode(gen)
    return out


def evaluate_bpb(model, stream, token_bytes, B, T, n_batches):
    """upstream loss_eval.evaluate_bpb: nats summed over tokens with >0 bytes / (ln2 * bytes)."""
    tot_nats, tot_bytes, tot_loss, n = 0.0, 0, 0.0, 0
    for x, y in stream.sequential_batches(B, T, n_batches):
        ce = loss_fn(model, mx.array(x), mx.array(y), reduction="none")
        nb = token_bytes[y]
        m = (nb > 0)
        ce_np = np.array(ce.astype(mx.float32))
        tot_nats += float((ce_np * m).sum())
        tot_bytes += int(nb.sum())
        tot_loss += float(ce_np.mean())
        n += 1
    return tot_loss / n, tot_nats / (math.log(2) * tot_bytes)


def make_update(model, opt, accum, mode="ref", compile_=True):
    """Build the per-optimizer-step function used by main() and efficiency/bench_step.py.

    update(get_batch) runs `accum` micro-batches (get_batch(k) -> numpy x, y), applies one
    MuonAdamW update and returns the mean loss as a Python float (one host sync per call for
    mode="fast"; per micro-batch for "ref", which is the original pilot loop).
      ref  : compiled micro (loss+grads) per micro-batch, eager tree_map accumulation,
             mx.eval + .item() after every micro-batch, compiled apply. (pilot behaviour)
      fast : compiled first/add micro-batch functions that accumulate on device, async_eval
             between micro-batches, compiled mean+apply, one sync per update.
    """
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    opt.init(model.trainable_parameters())  # state tree exists before compilation
    state = [model.state, opt.state]
    maybe = (lambda f, **kw: mx.compile(f, **kw)) if compile_ else (lambda f, **kw: f)

    if accum == 1:
        def full_step(x, y):
            loss, grads = loss_and_grad(model, x, y)
            opt.update(model, grads)
            return loss
        full_step = maybe(full_step, inputs=state, outputs=state)

        def update(get_batch):
            x, y = get_batch(0)
            loss = full_step(mx.array(x), mx.array(y))
            mx.eval(loss, state)
            return loss.item()
        return update

    if mode == "ref":
        def micro(x, y):
            return loss_and_grad(model, x, y)

        def apply(grads):
            opt.update(model, grads)
        micro = maybe(micro, inputs=state, outputs=state)
        apply = maybe(apply, inputs=state, outputs=state)

        def update(get_batch):
            gsum, lsum = None, 0.0
            for k in range(accum):
                x, y = get_batch(k)
                loss, grads = micro(mx.array(x), mx.array(y))
                gsum = grads if gsum is None else tree_map(lambda a, b: a + b, gsum, grads)
                mx.eval(loss, gsum)
                lsum += loss.item()
            gsum = tree_map(lambda g: g / accum, gsum)
            apply(gsum)
            mx.eval(state)
            return lsum / accum
        return update

    assert mode == "fast", mode
    mstate = [model.state]

    def first(x, y):
        return loss_and_grad(model, x, y)

    def add(x, y, lsum, gsum):
        loss, grads = loss_and_grad(model, x, y)
        return lsum + loss, tree_map(lambda a, b: a + b, gsum, grads)

    def apply_mean(lsum, gsum):
        opt.update(model, tree_map(lambda g: g / accum, gsum))
        return lsum / accum
    # outputs=mstate: nn.value_and_grad writes traced params into the model during the call
    first = maybe(first, inputs=mstate, outputs=mstate)
    add = maybe(add, inputs=mstate, outputs=mstate)
    apply_mean = maybe(apply_mean, inputs=state, outputs=state)

    def update(get_batch):
        lsum = gsum = None
        for k in range(accum):
            x, y = get_batch(k)
            if k == 0:
                lsum, gsum = first(mx.array(x), mx.array(y))
            else:
                lsum, gsum = add(mx.array(x), mx.array(y), lsum, gsum)
            mx.async_eval(lsum, gsum)
        loss = apply_mean(lsum, gsum)
        mx.eval(loss, state)
        return loss.item()
    return update


def main(args):
    mx.random.seed(args.seed)
    from tokenizer import ChatTokenizer, TOKENIZER_DIR
    tok = ChatTokenizer.load()
    token_bytes = np.array(json.load(open(os.path.join(TOKENIZER_DIR, "token_bytes.json"))), dtype=np.int64)
    cfg = config_for_depth(args.depth, tok.get_vocab_size(), args.seq_len, args.aspect_ratio, args.head_dim)
    model = GPT(cfg)
    model.compute_dtype = mx.bfloat16 if args.dtype == "bf16" else mx.float32
    model.set_fast_flags(fused_qkv=args.fused_qkv, ce_mode=args.ce_mode)
    out_dir = os.path.join(HERE, args.out)
    last_dir = os.path.join(out_dir, "last")
    resuming = args.resume and os.path.exists(os.path.join(last_dir, "meta.json"))

    B, T = args.device_batch, args.seq_len
    assert args.total_batch % (B * T) == 0
    accum = args.total_batch // (B * T)
    n_iter = args.num_iterations
    # upstream batch-size scaling of lrs and weight decay (reference: d12 @ 2^19 tokens)
    B_REF = 2 ** 19
    lr_scale = (args.total_batch / B_REF) ** 0.5
    d12 = GPT(config_for_depth(12, tok.get_vocab_size(), args.seq_len, args.aspect_ratio, args.head_dim))
    D_REF = 12 * d12.num_scaling_params()
    del d12
    target_tokens = args.total_batch * n_iter
    wd = args.weight_decay * math.sqrt(args.total_batch / B_REF) * (D_REF / target_tokens)

    if resuming:
        model.load_weights(os.path.join(last_dir, "model.safetensors"))
        meta = ckpt.load_meta(last_dir)
        assert meta["config"] == cfg.to_dict(), "config mismatch on resume"
        start_step = meta["step"]
        history = meta.get("history", [])
        elapsed_prev = meta.get("train_time_s", 0.0)
        print(f"resuming from step {start_step}")
    else:
        model.init_weights(args.seed)
        start_step, history, elapsed_prev = 0, [], 0.0
    opt, base_lrs = build_optimizer(model, args.matrix_lr, args.embedding_lr, args.unembedding_lr,
                                    args.scalar_lr, weight_decay=wd, lr_scale=lr_scale,
                                    muon_batched=args.muon_batched, muon_addmm=args.muon_addmm,
                                    ns_steps=args.ns_steps)
    if resuming:
        ckpt.load_optimizer(last_dir, opt, model)
    n_params = model.num_params()
    flops_per_token = model.estimate_flops()
    print(json.dumps({"config": cfg.to_dict(), "params": n_params, "scaling_params": model.num_scaling_params(),
                      "flops_per_token": flops_per_token, "accum": accum, "lr_scale": lr_scale, "wd": wd,
                      "target_tokens": target_tokens, "tokens_per_param": target_tokens / n_params}), flush=True)

    train = TokenStream("train", seed=args.seed)
    val = TokenStream("val")
    print(f"train tokens available: {train.total/1e6:.1f}M, planned: {target_tokens/1e6:.1f}M", flush=True)

    def get_lr_mult(it):
        wdn = round(args.warmdown_ratio * n_iter)
        if it < args.warmup_steps:
            return (it + 1) / args.warmup_steps
        if it <= n_iter - wdn:
            return 1.0
        p = (n_iter - it) / wdn
        return p + (1 - p) * args.final_lr_frac

    def get_muon_momentum(it):
        wdn = round(args.warmdown_ratio * n_iter)
        ws = n_iter - wdn
        if it < 400:
            f = it / 400
            return (1 - f) * 0.85 + f * 0.97
        if it >= ws:
            p = (it - ws) / max(wdn, 1)
            return 0.97 * (1 - p) + 0.90 * p
        return 0.97

    def get_wd(it):
        return wd * 0.5 * (1 + math.cos(math.pi * it / n_iter))

    update = make_update(model, opt, accum, mode=args.accum_mode, compile_=not args.no_compile)

    def save_last(step, train_time, tag="last"):
        d = os.path.join(out_dir, tag)
        ckpt.save(d, model, opt, {"step": step, "history": history, "train_time_s": train_time,
                                  "args": vars(args), "seed": args.seed, "base_lrs": base_lrs})
        return d

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(flag=True))
    log_path = os.path.join(out_dir, "log.jsonl")
    os.makedirs(out_dir, exist_ok=True)
    logf = open(log_path, "a")
    t_start = time.time()
    last_save = time.time()
    mx.reset_peak_memory()
    step = start_step
    ema = None
    xo, yo = train.batch(0, B, T)
    while step < n_iter:
        set_schedule(opt, base_lrs, get_lr_mult(step), get_muon_momentum(step), get_wd(step))
        t0 = time.time()
        lval = update(lambda k: (xo, yo) if args.overfit_one_batch else train.batch(step * accum + k, B, T))
        dt = time.time() - t0
        step += 1
        tps = args.total_batch / dt
        ema = lval if ema is None else 0.9 * ema + 0.1 * lval
        rec = {"step": step, "loss": lval, "dt": dt, "tok_per_s": tps,
               "lr_mult": get_lr_mult(step - 1), "peak_mem_gb": mx.get_peak_memory() / 1e9,
               "time": elapsed_prev + time.time() - t_start}
        if step % args.eval_every == 0 or step == n_iter or step == 1:
            vl, bpb = evaluate_bpb(model, val, token_bytes, B, T, args.eval_batches)
            rec.update(val_loss=vl, val_bpb=bpb)
            history.append({"step": step, "val_loss": vl, "val_bpb": bpb, "tokens": step * args.total_batch})
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if step % 10 == 0 or "val_loss" in rec:
            mfu_note = f"{flops_per_token * tps / 1e12:.2f} TFLOP/s"
            print(f"step {step}/{n_iter} loss {lval:.4f} (ema {ema:.4f}) dt {dt*1000:.0f}ms tok/s {tps:,.0f} {mfu_note} "
                  f"mem {rec['peak_mem_gb']:.1f}GB" + (f" | val_loss {rec['val_loss']:.4f} bpb {rec['val_bpb']:.4f}" if "val_loss" in rec else ""),
                  flush=True)
        train_time = elapsed_prev + time.time() - t_start
        if args.sample_every and step % args.sample_every == 0:
            d = save_last(step, train_time, tag=f"step_{step:05d}")
            with open(os.path.join(d, "samples.json"), "w") as f:
                json.dump(sample_text(model, tok), f, indent=1)
        timed_out = args.time_budget_min and (time.time() - t_start) / 60 > args.time_budget_min
        if time.time() - last_save > args.save_every_min * 60 or step == args.stop_at_step or timed_out or stop["flag"]:
            save_last(step, train_time)
            last_save = time.time()
            print(f"saved checkpoint at step {step}", flush=True)
        if step == args.stop_at_step or timed_out or stop["flag"]:
            print("stopping early:", "time budget" if timed_out else "requested", flush=True)
            break
    if step >= n_iter:
        save_last(step, elapsed_prev + time.time() - t_start)
        samples = sample_text(model, tok)
        with open(os.path.join(out_dir, "last", "samples.json"), "w") as f:
            json.dump(samples, f, indent=1)
        for k, v in samples.items():
            print(repr(k), "->", repr(v))
    print(f"done at step {step}; peak mem {mx.get_peak_memory()/1e9:.2f} GB", flush=True)
    return step, history


if __name__ == "__main__":
    main(get_args())
