"""MLX-only learner, to be run by Claude. Wall-clock training, resumable bounded saves.

--steps is an absolute optimizer-step ceiling for smoke/equal-update diagnostics only.
Production defaults have no epoch/step ceiling: sampling continues until the time guard.
"""
import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from gpt import GPT, GPTConfig, loss_fn
import r5_checkpoint as ckpt
from r5_data import digest, json_write, rate_plan, sample_indices
from r5_world import FACTORS
from r5_validation import validation_batches, validation_summary


class FactoredMLP(nn.Module):
    def __init__(self, din, hidden=256):
        super().__init__()
        self.l1, self.l2 = nn.Linear(din, hidden), nn.Linear(hidden, hidden)
        self.verb, self.item, self.arg = nn.Linear(hidden, 6), nn.Linear(hidden, 4), nn.Linear(hidden, 3)

    def __call__(self, x):
        h = nn.relu(self.l2(nn.relu(self.l1(x))))
        return self.verb(h), self.item(h), self.arg(h)


def mlp_loss(model, x, targets, unused=None):
    heads = model(x)
    labels = mx.array(FACTORS, dtype=mx.int32)[targets]
    losses = []
    for k, logits in enumerate(heads):
        y = labels[:, k]
        loss = nn.losses.cross_entropy(logits, mx.maximum(y, 0), reduction="none")
        losses.append(loss * (y >= 0))
    return (losses[0] + losses[1] + losses[2]).mean()


def model_from_meta(meta):
    if meta["kind"] == "lm":
        return GPT(GPTConfig(**meta["model_config"]))
    return FactoredMLP(**meta["model_config"])


def load_policy_model(path):
    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    model = model_from_meta(meta)
    model.load_weights(str(path / "model.safetensors"))
    mx.eval(model.parameters())
    return model, meta


def get_batch(data, kind, idx):
    if kind == "lm":
        tokens = data["tokens"][idx].astype(np.int32)
        return tokens[:, :-1], tokens[:, 1:], data["masks"][idx].astype(np.float32)
    return data["features"][idx], data["targets"][idx], np.zeros(len(idx), np.float32)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("lm", "mlp"), default="lm")
    ap.add_argument("--init", choices=("pretrained", "scratch"), default="pretrained")
    ap.add_argument("--base", default="checkpoints/pilot/last")
    ap.add_argument("--from-checkpoint", default="")
    ap.add_argument("--data", default="results/r5/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=7)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--replay", default="")
    ap.add_argument("--save-seconds", type=float, default=90)
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--pause-after-save-step", type=int, default=0, help="kill/resume smoke gate only")
    a = ap.parse_args()
    assert 0 < a.max_minutes <= 45
    mx.set_memory_limit(14 * 1024**3)
    mx.set_cache_limit(1024**3)
    started = time.monotonic()
    deadline = started + 60 * a.max_minutes
    # Reserve final validation + export time. An in-flight kernel cannot be preempted.
    reserve = min(8., a.max_minutes * 6)
    train_deadline = deadline - reserve
    out, data_dir = Path(a.out), Path(a.data)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "current.json").exists() and not a.resume:
        raise FileExistsError(f"preserving {out}; use --resume or a new --out")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    data, val = read_arrays(data_dir / "train.npz"), read_arrays(data_dir / "val.npz")
    for name in ("train", "val"):
        if digest(data_dir / f"{name}.npz") != manifest["counts"][name]["sha256"]:
            raise ValueError(f"{name} dataset digest mismatch")
    if digest("checkpoints/tokenizer/tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("tokenizer changed")
    replay = read_arrays(a.replay) if a.replay else None
    if replay is not None:
        assert replay["tokens"].shape[1] == data["tokens"].shape[1]
        assert len(replay["targets"]) > 0
    a.batch = a.batch or (16 if a.kind == "lm" else 256)
    a.lr = a.lr or (3e-4 if a.kind == "lm" else 1e-3)
    assert a.batch >= 2
    mx.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    resume_path = ckpt.current(out) if a.resume else None
    source = resume_path or (Path(a.from_checkpoint) if a.from_checkpoint else None)
    if source:
        model, old = load_policy_model(source)
        if old["kind"] != a.kind:
            raise ValueError("checkpoint kind mismatch")
        model_config = old["model_config"]
    elif a.kind == "mlp":
        model_config = dict(din=data["features"].shape[1], hidden=a.hidden)
        model = FactoredMLP(**model_config)
    else:
        if a.tiny:
            assert a.init == "scratch"
            cfg = GPTConfig(n_layer=1, n_head=1, n_kv_head=1, n_embd=64)
        else:
            cfg = GPTConfig(**json.loads((Path(a.base) / "meta.json").read_text())["config"])
        model_config = cfg.to_dict()
        model = GPT(cfg)
        if a.init == "pretrained":
            model.load_weights(str(Path(a.base) / "model.safetensors"))
        else:
            model.init_weights(a.seed)
    opt = optim.AdamW(learning_rate=a.lr, weight_decay=.01, bias_correction=True)
    opt.init(model.trainable_parameters())
    spec = dict(kind=a.kind, model_config=model_config, seed=a.seed, batch=a.batch, lr=a.lr,
                train_sha256=manifest["counts"]["train"]["sha256"],
                val_sha256=manifest["counts"]["val"]["sha256"],
                replay_sha256=digest(a.replay) if a.replay else None,
                tokenizer_sha256=manifest["tokenizer_sha256"],
                source_sha256={f: digest(f) for f in ("r5_world.py", "r5_data.py", "r5_train.py",
                                                       "r5_checkpoint.py", "r5_protocol.py", "r5_validation.py",
                                                       "gpt.py", "fused_ce.py")})
    step, elapsed_prior, initial_loss, last_loss = 0, 0., None, None
    if resume_path:
        old = ckpt.restore(resume_path, model, opt, rng)
        if old["spec"] != spec:
            raise ValueError("resume config/data changed; use original flags")
        step, elapsed_prior = old["step"], old["elapsed_total_s"]
        initial_loss, last_loss = old["initial_loss"], old["last_loss"]
    start_step = step
    lf = loss_fn if a.kind == "lm" else mlp_loss
    lg = nn.value_and_grad(model, lf)
    state = [model.state, opt.state]

    @lambda f: mx.compile(f, inputs=state, outputs=state)
    def train_step(x, y, m):
        loss, grads = lg(model, x, y, m)
        opt.update(model, grads)
        return loss

    mx.eval(model.parameters(), opt.state)
    stop = [False]
    def request_stop(signum, frame):
        stop[0] = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    log = open(out / "train.jsonl", "a", buffering=1)
    def emit(record):
        record = dict(step=step, session_start_step=start_step,
                      elapsed_s=time.monotonic() - started, **record)
        log.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def validate():
        # Turn 1 silently monitored just 32 rows, all from one-delivery episodes.
        # Cover the entire fixed set and aggregate by supervised units, not batch means.
        sums, counts, exact = [], [], []
        for idx in validation_batches(len(val["targets"]), a.batch):
            x, y, mask = map(mx.array, get_batch(val, a.kind, idx))
            if a.kind == "lm":
                logits = model(x)
                ce = nn.losses.cross_entropy(logits, y, reduction="none")
                row_sum = (ce * mask).sum(-1)
                row_count = mask.sum(-1)
                row_exact = ((mx.argmax(logits, axis=-1) == y) | (mask == 0)).all(-1)
            else:
                heads = model(x)
                lp = [h - mx.logsumexp(h, axis=-1, keepdims=True) for h in heads]
                scores = mx.stack([lp[0][:, v] + (lp[1][:, i] if i >= 0 else 0) +
                                   (lp[2][:, j] if j >= 0 else 0) for v, i, j in FACTORS], axis=-1)
                row_sum = -mx.take_along_axis(scores, y[:, None], axis=-1)[:, 0]
                row_count = mx.ones_like(row_sum)
                row_exact = mx.argmax(scores, axis=-1) == y
            sums.extend(np.array(row_sum).tolist())
            counts.extend(np.array(row_count).tolist())
            exact.extend(np.array(row_exact).tolist())
        return validation_summary(sums, counts, exact, val["deliveries"])

    val_initial = validate()
    emit(dict(event="start", val_loss=val_initial["nll"], val_metrics=val_initial, spec=spec,
              resume=bool(a.resume), allowance_s=a.max_minutes * 60,
              replay_examples=len(replay["targets"]) if replay else 0))
    val_last = val_initial
    last_save = last_log = last_val = rate_start = time.monotonic()
    rate_step = step
    warmed = False

    def save(final=False):
        meta = dict(version=1, kind=a.kind, model_config=model_config, spec=spec, args=vars(a),
                    step=step, optimizer_step=int(opt.step.item()), initial_loss=initial_loss,
                    last_loss=last_loss, val_loss=val_last["nll"], val_metrics=val_last,
                    elapsed_total_s=elapsed_prior + time.monotonic() - started,
                    session_elapsed_s=time.monotonic() - started,
                    session_steps=step - start_step, seed=a.seed,
                    initialization="continued" if a.from_checkpoint else a.init,
                    parent_checkpoint=a.from_checkpoint or (a.base if a.init == "pretrained" else None))
        ckpt.save(out, model, opt, meta, rng, final)

    while not stop[0] and time.monotonic() < train_deadline and (not a.steps or step < a.steps):
        nbase = a.batch if replay is None else a.batch - a.batch // 2
        idx = sample_indices(a.seed, step, len(data["targets"]), nbase)
        batch = get_batch(data, a.kind, idx)
        if replay is not None:
            ridx = sample_indices(a.seed + 1, step, len(replay["targets"]), a.batch // 2)
            rb = get_batch(replay, a.kind, ridx)
            batch = tuple(np.concatenate((x, y), axis=0) for x, y in zip(batch, rb))
        loss = train_step(*map(mx.array, batch))
        mx.eval(loss, state)
        last_loss = float(loss.item())
        if not np.isfinite(last_loss):
            raise FloatingPointError(f"nonfinite loss at step {step}")
        if initial_loss is None:
            initial_loss = last_loss
        step += 1
        now = time.monotonic()
        # Exclude graph compilation/first updates from the measured rate projection.
        if step - start_step == 5:
            rate_start, rate_step, warmed = now, step, True
        if now - last_val >= 60 and train_deadline - now > reserve:
            val_last = validate()
            last_val = time.monotonic()
        if now - last_log >= 10 or a.steps:
            units = a.batch * (data["tokens"].shape[1] - 1) if a.kind == "lm" else a.batch
            emit(dict(event="train", loss=last_loss, val_loss=val_last["nll"],
                      val_metrics=val_last, rate_warmed=warmed,
                      **rate_plan(step - rate_step, now - rate_start, train_deadline - now, units)))
            last_log = now
        if now - last_save >= a.save_seconds or (a.save_steps and step % a.save_steps == 0):
            save()
            emit(dict(event="checkpoint"))
            last_save = time.monotonic()
            if a.pause_after_save_step == step:
                emit(dict(event="pause_for_kill_test"))
                while not stop[0] and time.monotonic() < train_deadline:
                    time.sleep(.02)
    val_last = validate()
    save(final=True)
    reason = "signal" if stop[0] else "step_limit" if a.steps and step >= a.steps else "wall_guard"
    emit(dict(event="done", loss=last_loss, val_loss=val_last["nll"],
              val_metrics=val_last, stop_reason=reason,
              wall_utilization=(time.monotonic() - started) / (a.max_minutes * 60),
              peak_memory_gb=mx.get_peak_memory() / 1e9))
    log.close()
    if not a.steps and not stop[0] and time.monotonic() - started < a.max_minutes * 20:
        raise RuntimeError("underfilled allowance (<1/3); investigate before proceeding")


if __name__ == "__main__":
    main()
