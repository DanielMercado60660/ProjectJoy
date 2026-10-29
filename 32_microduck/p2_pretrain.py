"""Train Policy2's leaky image encoder on confirmed in-view teacher detections.

    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_pretrain.py runs/p2_pre \
        --index results/m5/bc_index.jsonl --img 64 --lamp-weight 3 --steps 4000
    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_pretrain.py runs/p2_pre --resume --steps 6000

Fit: train configs 0-199; held out from THIS pretraining: train configs 200-299, all tasks/k.
Frames and first-done/NaN cleaning come exclusively from p2_train.load_episode's existing cache.
Only c1..c4, img, and a disposable nine-label classifier train. A checkpoint contains:
  weights.safetensors + p2_config.json: full Policy2; GRU/body/instruction/action/aux heads stay fresh
  pretrain.safetensors, opt.safetensors, meta.json: classifier + optimizer + NumPy RNG for resume
To fine-tune, pass weights.safetensors to p2_train.py --init with --act leaky and matching --img,
--aux-weight (zero vs nonzero) and --done-from-aux. p2_train does not infer those flags from --init.
No dropout/MLX randomness follows initialization, so resume restores only the NumPy sampling RNG.
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from p2_labels import INVIEW_CLASSES, LABEL_VERSION, inview_labels
from p2_learner import Policy2, policy_config
from p2_train import CACHE_VERSION, load_episode

HERE = Path(__file__).resolve().parent
ENCODER_LAYERS = ("c1", "c2", "c3", "c4", "img")
LAMP_COLUMNS = [INVIEW_CLASSES.index("lamp_on"), INVIEW_CLASSES.index("lamp_off")]


def load_data(index_files, img=64, deadline=None):
    """Partition by the SOURCE header's config, before sampling any frames. Fail on bad metadata."""
    groups = {"fit": [], "heldout": []}
    unknown = {name: {} for name in groups}
    manifest, seen = [], set()
    for index in index_files:
        with open(HERE / index) as stream:
            for line in stream:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("wall guard reached while loading/cache-building; rerun to continue")
                if not line.strip():
                    continue
                row = json.loads(line)
                if "error" in row or row.get("split", "train") != "train":
                    continue
                path = (HERE / row["path"]).resolve()
                if path in seen:
                    continue
                seen.add(path)
                with np.load(path, allow_pickle=False) as z:
                    header = json.loads(str(z["header"]))
                    if header.get("split") != "train":
                        raise ValueError(f"index claims train, source does not: {path}")
                    config = header["config_id"]
                    match = re.fullmatch(r"train-(\d+)", config)
                    if match is None or ("config" in row and row["config"] != config):
                        raise ValueError(f"bad or contradictory config id: {path}")
                    cid = int(match[1])
                    if cid >= 300:
                        continue
                    memory = json.loads(str(z["memory"]))
                group = "fit" if cid < 200 else "heldout"
                ep = load_episode(str(path), img)          # the SAME resize + cleaned cache as policy training
                if len(memory) != int(ep["source_ticks"]):
                    raise ValueError(f"memory/cache length mismatch: {path}")
                labels = inview_labels(header, memory, unknown_counts=unknown[group])[ep["ticks"]]
                groups[group].append({"frames": ep["frames"], "labels": labels,
                                      "ticks": ep["ticks"], "config": cid, "path": str(path)})
                manifest.append({"path": str(path), "config": cid, "group": group,
                                 "size": int(ep["src_size"]), "mtime_ns": int(ep["src_mtime_ns"]),
                                 "ticks_sha256": hashlib.sha256(ep["ticks"].tobytes()).hexdigest()})
    if not groups["fit"] or not groups["heldout"]:
        raise ValueError("need both fit configs 0-199 and held-out configs 200-299")
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    report = {"fingerprint": fingerprint, "img": img, "label_version": LABEL_VERSION,
              "cache_version": CACHE_VERSION, "classes": list(INVIEW_CLASSES), "sources": manifest}
    for name, eps in groups.items():
        positives = sum((e["labels"].sum(axis=0, dtype=np.int64) for e in eps), np.zeros(len(INVIEW_CLASSES), np.int64))
        report[name] = {"episodes": len(eps), "configs": sorted({e["config"] for e in eps}),
                        "frames": sum(len(e["labels"]) for e in eps),
                        "positives": dict(zip(INVIEW_CLASSES, map(int, positives))),
                        "unknown_track_rows_in_sources": unknown[name]}
    return groups["fit"], groups["heldout"], report


class FrameSampler:
    """Uniform retained-frame sampling; open each selected episode map once per batch."""
    def __init__(self, eps, img):
        self.eps, self.img = eps, img
        self.ends = np.cumsum([len(e["labels"]) for e in eps], dtype=np.int64)

    def batch(self, rng, size):
        flat = rng.integers(0, self.ends[-1], size=size)
        which = np.searchsorted(self.ends, flat, side="right")
        starts = np.r_[0, self.ends[:-1]]
        local = flat - starts[which]
        frames = np.empty((size, self.img, self.img, 3), np.uint8)
        labels = np.empty((size, len(INVIEW_CLASSES)), np.float32)
        for j in np.unique(which):
            rows = np.flatnonzero(which == j)
            e = self.eps[int(j)]
            frames[rows] = e["frames"][local[rows]]
            labels[rows] = e["labels"][local[rows]]
        return frames, labels


class EncoderPretrainer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.policy = Policy2(**policy_config(cfg=cfg))
        if self.policy.act != "leaky":
            raise ValueError("encoder pretraining requires act='leaky'")
        self.policy.freeze()
        for name in ENCODER_LAYERS:
            getattr(self.policy, name).unfreeze()
        self.classifier = nn.Linear(192, len(INVIEW_CLASSES))

    def features(self, frames):
        """Same c1..c4/img operations as Policy2.encode, without body/instruction inputs."""
        x = frames
        for name in ENCODER_LAYERS[:-1]:
            x = nn.leaky_relu(getattr(self.policy, name)(x), 0.05)
        return nn.leaky_relu(self.policy.img(x.reshape(x.shape[0], -1)), 0.05)

    def __call__(self, frames):
        return self.classifier(self.features(frames))


def loss_fn(model, frames, labels, lamp_weight=1.0):
    logits = model(frames.astype(mx.float32) / 255.0)
    # Stable BCE on independent labels, including simultaneous on/off teacher detections.
    ce = (labels * nn.softplus(-logits) + (1 - labels) * nn.softplus(logits)).mean(-1)
    has_lamp = mx.maximum(labels[:, LAMP_COLUMNS[0]], labels[:, LAMP_COLUMNS[1]])
    weights = 1 + (lamp_weight - 1) * has_lamp       # whole-frame weighting, applied once
    return (ce * weights).sum() / weights.sum()


def binary_auc(scores, labels):
    """Rank AUC with average ranks for ties. None if either class is absent."""
    labels = np.asarray(labels, bool)
    positives, n = int(labels.sum()), len(labels)
    negatives = n - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="stable")
    scores = np.asarray(scores)[order]
    starts = np.flatnonzero(np.r_[True, scores[1:] != scores[:-1]])
    ends = np.r_[starts[1:], n]
    ranks = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return float((ranks[labels[order]].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def class_metrics(logits, labels):
    """Unweighted held-out metrics; raw accuracy alone rewards predicting every class absent."""
    out = {}
    for k, name in enumerate(INVIEW_CLASSES):
        y, pred = labels[:, k].astype(bool), logits[:, k] > 0
        n, pos = len(y), int(y.sum())
        neg = n - pos
        tp, tn = int((pred & y).sum()), int((~pred & ~y).sum())
        fp, fn = neg - tn, pos - tp
        out[name] = {"accuracy": (tp + tn) / n if n else None,
                     "balanced_accuracy": (tp / pos + tn / neg) / 2 if pos and neg else None,
                     "precision": tp / (tp + fp) if tp + fp else None,
                     "recall": tp / pos if pos else None,
                     "auc": binary_auc(logits[:, k], y),
                     "always_absent_accuracy": neg / n if n else None,
                     "majority_accuracy": max(pos, neg) / n if n else None,
                     "positive": pos, "negative": neg, "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return out


def evaluate(model, eps, batch_size=256, deadline=None):
    scores, targets = [], []
    total = sum(len(e["labels"]) for e in eps)
    for e in eps:
        for start in range(0, len(e["labels"]), batch_size):
            if deadline is not None and time.monotonic() >= deadline:
                break
            frames = e["frames"][start:start + batch_size]
            logits = model(mx.array(frames).astype(mx.float32) / 255.0)
            scores.append(np.array(logits))
            targets.append(e["labels"][start:start + batch_size])
        else:
            continue
        break
    logits = np.concatenate(scores) if scores else np.empty((0, len(INVIEW_CLASSES)), np.float32)
    labels = np.concatenate(targets) if targets else np.empty_like(logits)
    if not np.isfinite(logits).all():
        raise FloatingPointError("nonfinite held-out logits")
    ce = labels * np.logaddexp(0, -logits) + (1 - labels) * np.logaddexp(0, logits)
    return {"scope": "pretraining_only_train_configs_200_299", "complete": len(labels) == total,
            "frames_evaluated": len(labels), "total_frames": total, "threshold_logit": 0.0,
            "loss": float(ce.mean()) if len(labels) else None, "classes": class_metrics(logits, labels)}


def save(run, step, model, opt, rng, cfg, fingerprint):
    d = Path(run) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    opt.init(model.trainable_parameters())
    mx.eval(model.parameters(), opt.state)
    model.save_weights(str(d / "pretrain.safetensors"))
    model.policy.save_weights(str(d / "weights.safetensors"))  # original Policy2 names, no classifier
    mx.save_safetensors(str(d / "opt.safetensors"), dict(tree_flatten(opt.state)))
    (d / "p2_config.json").write_text(json.dumps(policy_config(cfg=cfg), indent=2))
    (d / "meta.json").write_text(json.dumps({"step": step, "cfg": cfg, "rng": rng.bit_generator.state,
        "fingerprint": fingerprint, "label_version": LABEL_VERSION, "cache_version": CACHE_VERSION,
        "classes": list(INVIEW_CLASSES), "mlx_rng": "initialization only; seed in cfg"}))
    (Path(run) / "latest.json").write_text(json.dumps({"step": step, "dir": d.name}))
    return d


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--index", action="append")
    ap.add_argument("--img", type=int, choices=(64, 96), default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lamp-weight", type=float, default=1.0, help="upweight BCE on frames with any lamp detection")
    ap.add_argument("--aux-weight", type=float, default=1.0, help="exported Policy2 architecture only: 0 removes aux heads")
    ap.add_argument("--done-from-aux", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    run, meta, checkpoint = HERE / a.run, None, None
    if a.resume:
        latest = json.loads((run / "latest.json").read_text())
        checkpoint = run / latest["dir"]
        meta = json.loads((checkpoint / "meta.json").read_text())
        if (meta["label_version"] != LABEL_VERSION or meta["cache_version"] != CACHE_VERSION
                or meta["classes"] != list(INVIEW_CLASSES)):
            ap.error("label/cache vocabulary changed; start a new run")
        explicit = {v.split("=", 1)[0] for v in argv if v.startswith("--")}
        for key in ("index", "img", "batch", "lr", "lamp_weight", "aux_weight", "done_from_aux", "seed"):
            old = meta["cfg"][key]
            if "--" + key.replace("_", "-") in explicit and getattr(a, key) != old:
                ap.error(f"--resume cannot change {key}")
            setattr(a, key, old)
        if a.steps < meta["step"]:
            ap.error("--steps is below the saved step")
    elif (run / "latest.json").exists():
        ap.error("run already has a checkpoint; use --resume or a new directory")
    if not a.index:
        ap.error("provide --index for a new run")
    if a.steps < 0 or min(a.batch, a.ckpt_every, a.eval_every, a.eval_batch) < 1:
        ap.error("steps must be >= 0; batch and checkpoint/evaluation intervals must be > 0")
    if not 0 < a.max_minutes <= 45 or not np.isfinite(a.lr) or a.lr <= 0:
        ap.error("need max-minutes in (0,45] and finite lr > 0")
    if not np.isfinite(a.lamp_weight) or a.lamp_weight < 1:
        ap.error("lamp-weight must be finite and >= 1")
    if not np.isfinite(a.aux_weight) or a.aux_weight < 0 or (a.done_from_aux and a.aux_weight == 0):
        ap.error("aux-weight must be finite and >= 0; done-from-aux needs auxiliary heads")
    cfg = dict(vars(a), act="leaky", aux=a.aux_weight > 0)
    t0 = time.monotonic()
    deadline = t0 + a.max_minutes * 60
    try:
        fit, heldout, data_report = load_data(a.index, a.img, deadline)
    except TimeoutError as e:
        print(str(e), flush=True)
        return
    fingerprint = data_report["fingerprint"]
    if meta is not None and meta["fingerprint"] != fingerprint:
        raise ValueError("source episodes/order changed since checkpoint; start a new run")
    run.mkdir(parents=True, exist_ok=True)
    (run / "dataset.json").write_text(json.dumps(data_report, indent=2))
    print(json.dumps({name: data_report[name] for name in ("fit", "heldout")}), flush=True)
    mx.random.seed(a.seed)
    model = EncoderPretrainer(cfg)
    opt = optim.Adam(learning_rate=a.lr)
    rng, step = np.random.default_rng(a.seed), 0
    if meta is not None:
        model.load_weights(str(checkpoint / "pretrain.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(checkpoint / "opt.safetensors")).items()))
        rng.bit_generator.state = meta["rng"]
        step = meta["step"]
        print(f"resumed at step {step}", flush=True)
    mx.eval(model.parameters(), opt.state)
    if meta is None:
        save(run, 0, model, opt, rng, cfg, fingerprint)     # retain the fresh non-encoder parameters
    sampler = FrameSampler(fit, a.img)
    vg = nn.value_and_grad(model, loss_fn)
    first_step, t_train, last_eval = step, time.monotonic(), None

    def report_eval():
        result = dict(evaluate(model, heldout, a.eval_batch, deadline), step=step)
        with open(run / "eval_log.jsonl", "a") as log:
            log.write(json.dumps(result) + "\n")
        (run / "heldout.json").write_text(json.dumps(result, indent=2))
        print(json.dumps({"heldout": result}), flush=True)
        return result

    with open(run / "train_log.jsonl", "a") as log:
        while step < a.steps and time.monotonic() < deadline:
            frames, labels = sampler.batch(rng, a.batch)
            loss, grads = vg(model, mx.array(frames), mx.array(labels), a.lamp_weight)
            grads, _ = optim.clip_grad_norm(grads, 1.0)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"nonfinite training loss at step {step + 1}")
            step += 1
            if step % 100 == 0 or step == a.steps:
                elapsed = time.monotonic() - t_train
                record = {"step": step, "loss": float(loss), "elapsed_s": elapsed,
                          "steps_per_s": (step - first_step) / max(elapsed, 1e-6)}
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(record, flush=True)
            if step % a.ckpt_every == 0:
                save(run, step, model, opt, rng, cfg, fingerprint)
            if step % a.eval_every == 0 or step == a.steps:
                last_eval = report_eval()
    if step < a.steps:
        print("wall guard reached; saving (held-out metrics may be incomplete)", flush=True)
    d = save(run, step, model, opt, rng, cfg, fingerprint)
    if last_eval is None or last_eval["step"] != step:
        last_eval = report_eval()
    (d / "heldout.json").write_text(json.dumps(last_eval, indent=2))
    (run / "summary.json").write_text(json.dumps({"start_step": first_step, "final_step": step,
        "wall_s": time.monotonic() - t0, "budget_minutes": a.max_minutes,
        "frames_seen_this_invocation": (step - first_step) * a.batch,
        "fit_frames": data_report["fit"]["frames"], "heldout_frames": data_report["heldout"]["frames"],
        "evaluation_complete": last_eval["complete"], "weights": str(d / "weights.safetensors")}))


if __name__ == "__main__":
    main()
