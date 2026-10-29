"""Phase 2 BC/DAgger: M5 windows/loss plus bot-belief lamp and target-match supervision.

    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_64 \
        --index results/m5/bc_index.jsonl --index episodes/dagger_index.jsonl \
        --index episodes/dagger2_index.jsonl --img 64 --aux-weight 1 --done-from-aux
    # Continue with saved data/model/optimizer/sampling settings:
    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_train.py runs/p2_64 --resume --steps 24000

Stored recordings are never rewritten. First-done truncation and finite-label filtering happen
before caching. Each version/img/source-keyed cache directory has frames.npy (mapped only while
sampling a window) and labels.npz (small arrays in RAM). 96px frames would exceed 20 GB if all loaded.
The loss and accuracy denominators exclude padding, burn-in and masked labels. Accuracy reports
describe the current training batch, with class counts; absent classes are null, not 100% correct.
T3: --vx-head binary replaces vx MSE with unit-weight walk BCE; decoding emits exactly 0 or 0.3.
"""
import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

from p2_learner import Policy2, body_vec, cmd_targets, instr_vec, policy_config, prep_frame

HERE = Path(__file__).resolve().parent
CACHE = HERE / "episodes" / "cache"
CACHE_VERSION = "p2_belief_v1"
REG_W = np.array([1, 1, 0.5, 0.5, 0.5, 0.5], np.float32)
# T3: compare cached normalized vx against the SAME float32 transform of 0.15; no cache rewrite.
WALK_REG_THRESHOLD = float(cmd_targets(np.array([0.15, 0, 0, 0, 0, 0, 0, 0], np.float32))[0][0])
# Amendment 6: turn-head labels on the bot's standing ticks (vx < 0.05): |wz| <= 0.1 none, > 0.1 left, < -0.1 right.
STAND_REG_THRESHOLD = float(cmd_targets(np.array([0.05, 0, 0, 0, 0, 0, 0, 0], np.float32))[0][0])
TURN_REG_THRESHOLD = float(cmd_targets(np.array([0, 0, 0.1, 0, 0, 0, 0, 0], np.float32))[0][1])
CLASSES = {"lamp": ("unknown", "on", "off"), "match": ("unknown", "matches", "mismatch")}


def label_rows(cmd_bot):
    """Keep through the first done (inclusive), then drop nonfinite command-label rows.

    Return ORIGINAL tick indices: removing a NaN row must not move press/sighting timestamps.
    cut_ticks counts the suffix removed by done; invalid_ticks counts invalid rows in the prefix.
    """
    cmd = np.asarray(cmd_bot)
    if cmd.ndim != 2 or cmd.shape[1] != 8:
        raise ValueError("cmd_bot must have shape (T, 8)")
    finished = np.flatnonzero(cmd[:, 7] > 0.5)
    end = int(finished[0]) + 1 if len(finished) else len(cmd)
    valid = np.isfinite(cmd[:end]).all(axis=1)
    return np.flatnonzero(valid), {"source_ticks": len(cmd), "cut_ticks": len(cmd) - end,
                                    "invalid_ticks": int((~valid).sum())}


def aux_labels(header, memory):
    """Teacher beliefs only: no truth, executed actions, or hindsight outcome labels.

    A bot press at tick p invalidates sightings with tick_seen <= p. A null lamp is unknown.
    Non-set_lamp match labels are -1 (masked); lamp labels remain useful on all tasks.
    """
    instruction = header["instruction"]
    is_lamp = instruction["task"] == "set_lamp"
    if is_lamp and instruction.get("target") not in ("on", "off"):
        raise ValueError("set_lamp needs an on/off target")
    presses = sorted(int(e["tick"]) for e in header.get("events", []) if e.get("event") == "press")
    lamp = np.zeros(len(memory), np.int32)
    match = np.full(len(memory), 0 if is_lamp else -1, np.int32)
    latest, j = -1, 0
    for t, mem in enumerate(memory):
        while j < len(presses) and presses[j] <= t:
            latest = presses[j]
            j += 1
        sighting = mem.get("lamp")
        if sighting is not None and sighting[1] > latest:
            state = sighting[0]
            if state not in ("on", "off"):
                raise ValueError(f"unexpected lamp belief {state!r} at tick {t}")
            lamp[t] = 1 if state == "on" else 2
            if is_lamp:
                match[t] = 1 if state == instruction["target"] else 2
    return lamp, match


class CachedFrames:
    """Hold a path, not an open map per episode (thousands of episodes can exhaust file handles)."""
    def __init__(self, path, ticks, img):
        self.path = Path(path)
        self.shape = (int(ticks), img, img, 3)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        frames = np.load(self.path, mmap_mode="r", allow_pickle=False)
        # Copy just this window; the mapping/file handle is released on return.
        return np.array(frames[:self.shape[0]][key], copy=True)   # a trimmed episode never shows its cut tick


def load_episode(rel, img=64):
    if img not in (64, 96):
        raise ValueError("img must be 64 or 96")
    p = (HERE / rel).resolve()
    stat = p.stat()
    key = hashlib.sha256(str(p).encode()).hexdigest()[:16]
    c = CACHE / f"{p.stem}_{key}_{img}_{CACHE_VERSION}"
    labels, frames_file = c / "labels.npz", c / "frames.npy"
    ep = None
    if labels.exists() and frames_file.exists():
        with np.load(labels, allow_pickle=False) as z:
            if (str(z["src"]) == str(p) and int(z["src_size"]) == stat.st_size
                    and int(z["src_mtime_ns"]) == stat.st_mtime_ns):
                ep = {k: z[k] for k in z.files}
    if ep is None:
        with np.load(p, allow_pickle=False) as z:
            header = json.loads(str(z["header"]))
            if header.get("split", "train") != "train":
                raise ValueError("only train episodes may enter the training cache")
            cmd = z["cmd_bot"]
            keep, stats = label_rows(cmd)
            if not len(keep):
                raise ValueError("no finite command labels before first done")
            memory = json.loads(str(z["memory"]))
            body = z["body"]
            frames = z["frames"]
            if not (len(memory) == len(body) == len(frames) == len(cmd)):
                raise ValueError("episode arrays have different tick counts")
            lamp, match = aux_labels(header, memory)
            reg, done = cmd_targets(cmd[keep])
            ep = {"body": np.stack([body_vec(b) for b in body[keep]]),
                  "instr": instr_vec(header["instruction"]), "reg": reg.astype(np.float32),
                  "done": done.astype(np.float32), "lamp_lbl": lamp[keep], "match_lbl": match[keep],
                  "ticks": keep, "src": np.array(str(p)), "src_size": np.array(stat.st_size),
                  "src_mtime_ns": np.array(stat.st_mtime_ns), **{k: np.array(v) for k, v in stats.items()}}
            c.mkdir(parents=True, exist_ok=True)
            # Publish complete files, labels last. Concurrent same-source cache builders must
            # never expose a partial .npy/.npz; an interrupted build can be retried safely.
            with tempfile.TemporaryDirectory(prefix=".build_", dir=c) as temp:
                work = Path(temp)
                np.save(work / "frames.npy", np.stack([prep_frame(frames[t], img) for t in keep]))
                np.savez(work / "labels.npz", **ep)
                (work / "frames.npy").replace(frames_file)
                (work / "labels.npz").replace(labels)
    ep["frames"] = CachedFrames(frames_file, len(ep["done"]), img)
    return ep


def load_all(index_files, img=64, deadline=None):
    eps, kept = [], []
    for f in index_files:
        counts = {"episodes": 0, "source_ticks": 0, "cut_ticks": 0, "invalid_ticks": 0, "kept_ticks": 0}
        with open(HERE / f) as stream:
            for line in stream:
                if deadline is not None and time.monotonic() >= deadline:
                    raise TimeoutError("wall guard reached while caching; rerun to finish the cache")
                if not line.strip():
                    continue
                row = json.loads(line)
                if "error" in row or row.get("split", "train") != "train":
                    continue
                try:
                    ep = load_episode(row["path"], img)
                except (KeyError, OSError, ValueError) as e:
                    print("skipping", row.get("path"), repr(e)[:120], flush=True)
                    continue
                if row.get("drop_done_tick") and len(ep["done"]) > 1 and ep["done"][-1] > 0.5:
                    # Amendment 3: a shadow-bot "done" judged unreliable is removed, never relabelled
                    # (relabelling would need simulator truth). The episode's other ticks stay.
                    ep = {k: (v[:-1] if k in ("body", "reg", "done", "lamp_lbl", "match_lbl", "ticks") else v)
                          for k, v in ep.items()}
                    ep["frames"] = CachedFrames(ep["frames"].path, len(ep["done"]), img)
                    counts["dropped_done"] = counts.get("dropped_done", 0) + 1
                eps.append(ep)
                kept.append(row)
                counts["episodes"] += 1
                for key in ("source_ticks", "cut_ticks", "invalid_ticks"):
                    counts[key] += int(ep[key])
                counts["kept_ticks"] += len(ep["done"])
        print(json.dumps({"index": str(f), "img": img, "cache_version": CACHE_VERSION, **counts}), flush=True)
    return kept, eps


def batch(eps, lens, rng, B, W=128, BURN=32, p_start=0.2, img=64, rec=None, rec_n=0, p_end=0.0):
    """rec (amendment 6): (episodes, focus positions) of the recovery group; the first rec_n windows come
    from it, starting uniformly in [focus, max(focus, n - W)], episode chosen in proportion to n - focus.
    With rec_n == 0 the random draws are exactly the original ones.
    p_end (amendment 7): share of main windows that end on the episode's last tick (its "done" tick); with
    p_end == 0 the random draws are exactly the original ones."""
    p = lens / lens.sum()
    F = np.zeros((B, W, img, img, 3), np.uint8)
    Bo = np.zeros((B, W, 34), np.float32)
    In = np.zeros((B, W, 9), np.float32)
    R = np.zeros((B, W, 6), np.float32)
    D, M = np.zeros((B, W), np.float32), np.zeros((B, W), np.float32)
    L, A = np.full((B, W), -1, np.int32), np.full((B, W), -1, np.int32)
    if rec_n:
        rec_eps, focus = rec
        rest = np.array([max(1, len(e["frames"]) - f) for e, f in zip(rec_eps, focus)], np.float64)
        rp = rest / rest.sum()
    for b in range(B):
        if b < rec_n:
            j = rng.choice(len(rec_eps), p=rp)
            e, f = rec_eps[j], int(focus[j])
            T = len(e["frames"])
            s = int(rng.integers(f, max(f, T - W) + 1))
        else:
            e = eps[rng.choice(len(eps), p=p)]
            T = len(e["frames"])
            u = rng.random()
            s = 0 if u < p_start else (max(0, T - W) if u < p_start + p_end else int(rng.integers(0, max(1, T - W + 1))))
        n = min(W, T - s)
        F[b, :n] = e["frames"][s:s + n]
        Bo[b, :n] = e["body"][s:s + n]
        In[b, :] = e["instr"]
        R[b, :n] = e["reg"][s:s + n]
        D[b, :n] = e["done"][s:s + n]
        L[b, :n] = e["lamp_lbl"][s:s + n]
        A[b, :n] = e["match_lbl"][s:s + n]
        M[b, (0 if s == 0 else BURN):n] = 1.0
    return F, Bo, In, R, D, M, L, A


class Streams:
    """Amendment 7 (history arm): a pool of streams, each playing one episode in order, W ticks at a time,
    carrying the GRU state from one chunk to the next, as at deployment. State resets to zero only when a
    stream starts a new episode (chosen uniformly, so every tick has equal weight, as with length-weighted
    windows). No burn-in: every real tick is in the loss mask. A chunk never crosses into the next episode;
    the short final chunk is padded and masked.
    Amendment 7a: with pool == B every stream advances every step, and on the real data consecutive batches
    are then so alike that training does not learn (H/Hr flat for 2,000 steps; i.i.d. chunks learn). With
    pool > B, each step draws B streams at random from the pool; a drawn stream's memory was computed under
    the weights of its previous draw (about pool / B steps earlier)."""
    def __init__(self, B, hidden, pool=None):
        P = B if pool is None else int(pool)
        self.B = B
        self.ep = np.full(P, -1, np.int64)
        self.pos = np.zeros(P, np.int64)
        self.h = np.zeros((P, hidden), np.float32)
        self.last = np.arange(B)

    def next(self, eps, rng, W, img=64):
        B = self.B
        self.last = np.arange(B) if len(self.ep) == B else np.sort(rng.choice(len(self.ep), B, replace=False))
        F = np.zeros((B, W, img, img, 3), np.uint8)
        Bo = np.zeros((B, W, 34), np.float32)
        In = np.zeros((B, W, 9), np.float32)
        R = np.zeros((B, W, 6), np.float32)
        D, M = np.zeros((B, W), np.float32), np.zeros((B, W), np.float32)
        L, A = np.full((B, W), -1, np.int32), np.full((B, W), -1, np.int32)
        for b, k in enumerate(self.last):
            if self.ep[k] < 0 or self.pos[k] >= len(eps[self.ep[k]]["frames"]):
                self.ep[k], self.pos[k] = int(rng.integers(len(eps))), 0
                self.h[k] = 0.0
            e, s = eps[self.ep[k]], int(self.pos[k])
            n = min(W, len(e["frames"]) - s)
            F[b, :n] = e["frames"][s:s + n]
            Bo[b, :n] = e["body"][s:s + n]
            In[b, :] = e["instr"]
            R[b, :n] = e["reg"][s:s + n]
            D[b, :n] = e["done"][s:s + n]
            L[b, :n] = e["lamp_lbl"][s:s + n]
            A[b, :n] = e["match_lbl"][s:s + n]
            M[b, :n] = 1.0
            self.pos[k] = s + W
        return (F, Bo, In, R, D, M, L, A), self.h[self.last].copy()

    def update(self, h_last):
        self.h[self.last] = h_last

    def state(self):
        return {"ep": self.ep, "pos": self.pos, "h": self.h}

    def load(self, z):
        self.ep, self.pos, self.h = np.array(z["ep"]), np.array(z["pos"]), np.array(z["h"], np.float32)


def masked_aux(logits, labels, M):
    mask = M * (labels >= 0)
    ce = nn.losses.cross_entropy(logits, mx.maximum(labels, 0), reduction="none")
    pred = mx.argmax(logits, axis=-1)
    counts = mx.stack([(mask * (labels == k)).sum() for k in range(3)])
    correct = mx.stack([(mask * (labels == k) * (pred == k)).sum() for k in range(3)])
    return (ce * mask).sum() / (mask.sum() + 1e-6), correct, counts


def loss_fn(model, F, Bo, In, R, D, M, L, A, aux_weight=1.0, h0=None):
    reg, dl, aux, h_last = model(F.astype(mx.float32) / 255.0, Bo, In, h0)
    if model.vx_head == "binary":                 # T3: vx is classified, all other regression weights unchanged
        mse = (((reg[..., 1:] - R[..., 1:]) ** 2) * mx.array(REG_W[1:])).sum(-1)
    else:
        mse = (((reg - R) ** 2) * mx.array(REG_W)).sum(-1)
    bce = 50.0 * D * nn.softplus(-dl) + (1 - D) * nn.softplus(dl)
    denom = M.sum() + 1e-6
    metrics = {"mse": (mse * M).sum() / denom, "bce": (bce * M).sum() / denom,
               "h_last": mx.stop_gradient(h_last)}   # amendment 7: carried to the next chunk in stream mode
    total = metrics["mse"] + metrics["bce"]
    metrics["walk_bce"] = mx.array(0.0)
    if model.vx_head == "binary":
        walk_y = (R[..., 0] > WALK_REG_THRESHOLD).astype(mx.float32)  # bot vx > 0.15
        walk = walk_y * nn.softplus(-aux["walk"]) + (1 - walk_y) * nn.softplus(aux["walk"])
        metrics["walk_bce"] = (walk * M).sum() / denom
        total = total + metrics["walk_bce"]        # weight 1, NOT aux_weight; same padding/burn mask
    metrics["turn_ce"] = mx.array(0.0)
    if getattr(model, "turn_head", "regress") == "discrete":   # amendment 6: unit weight, standing ticks only
        wz = R[..., 1]
        lbl = mx.where(wz > TURN_REG_THRESHOLD, 1, mx.where(wz < -TURN_REG_THRESHOLD, 2, 0))
        tm = M * (R[..., 0] < STAND_REG_THRESHOLD)
        ce = nn.losses.cross_entropy(aux["turn"], lbl, reduction="none")
        metrics["turn_ce"] = (ce * tm).sum() / (tm.sum() + 1e-6)
        total = total + metrics["turn_ce"]
    for name, labels in (("lamp", L), ("match", A)):
        if name in aux:
            value, correct, counts = masked_aux(aux[name], labels, M)
            total = total + aux_weight * value
        else:
            value, correct, counts = mx.array(0.0), mx.zeros(3), mx.zeros(3)
        metrics[name + "_loss"] = value
        metrics[name + "_correct"] = correct
        metrics[name + "_count"] = counts
    return total, metrics


def save(run, step, model, opt, rng, cfg, streams=None):
    cfg = dict(cfg, vx_head=model.vx_head, turn_head=model.turn_head)   # T3 / amendment 6: the actual heads
    d = Path(run) / f"ckpt_{step:06d}"
    d.mkdir(parents=True, exist_ok=True)
    opt.init(model.trainable_parameters())                 # also permits a wall-guard save at step 0
    random_keys = list(mx.random.state)
    mx.eval(model.parameters(), opt.state, random_keys)
    model.save_weights(str(d / "weights.safetensors"))
    mx.save_safetensors(str(d / "opt.safetensors"), dict(tree_flatten(opt.state)))
    if streams is not None:                                # amendment 7: where each row is, and its memory
        np.savez(d / "streams.npz", **streams.state())
    (d / "p2_config.json").write_text(json.dumps(policy_config(cfg=cfg), indent=2))
    meta = {"step": step, "rng": rng.bit_generator.state,
            "mlx_rng": [np.array(key).tolist() for key in random_keys], "cfg": cfg,
            "cache_version": CACHE_VERSION}
    (d / "meta.json").write_text(json.dumps(meta))
    (Path(run) / "latest.json").write_text(json.dumps({"step": step, "dir": d.name}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--index", action="append")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--init", default=None, help="strict, same-architecture weights (M5: 64px/no-aux only)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--burn", type=int, default=32)
    ap.add_argument("--p-start", type=float, default=0.2)
    ap.add_argument("--img", type=int, choices=(64, 96), default=64)
    ap.add_argument("--aux-weight", type=float, default=1.0)
    ap.add_argument("--done-from-aux", action="store_true")
    ap.add_argument("--act", choices=("relu", "leaky"), default="relu", help="image-encoder activation")
    ap.add_argument("--vx-head", choices=("regress", "binary"), default="regress")  # T3
    ap.add_argument("--turn-head", choices=("regress", "discrete"), default="regress")  # amendment 6
    ap.add_argument("--recovery-index", action="append", help="amendment 6: recovery group (rows need takeover_tick)")
    ap.add_argument("--recovery-frac", type=float, default=0.0, help="share of each batch from the recovery group")
    ap.add_argument("--recovery-lead", type=int, default=64, help="windows start no earlier than takeover_tick - lead")
    ap.add_argument("--p-end", type=float, default=0.0, help="amendment 7: share of windows ending on the last tick")
    ap.add_argument("--stream", action="store_true", help="amendment 7: whole episodes in order, memory carried")
    ap.add_argument("--stream-reset", action="store_true", help="amendment 7 control: --stream order, memory zeroed per chunk")
    ap.add_argument("--stream-pool", type=int, default=512, help="amendment 7a: streams in progress; --batch drawn per step")
    ap.add_argument("--stream-warmup", type=int, default=2000,
                    help="amendment 7a: with --stream, the first N steps use the window sampler (from scratch, stream "
                         "chunks alone never start learning)")
    a = ap.parse_args()
    run = HERE / a.run
    meta, d = None, None
    if a.resume:
        if a.init:
            ap.error("--init and --resume are mutually exclusive")
        lt = json.loads((run / "latest.json").read_text())
        d = run / lt["dir"]
        meta = json.loads((d / "meta.json").read_text())
        if meta["cache_version"] != CACHE_VERSION:
            ap.error("cache/label version changed; start a new run")
        # Omitted options inherit the saved run. Explicit changes to training semantics are errors.
        explicit = {s.split("=", 1)[0] for s in sys.argv[1:] if s.startswith("--")}
        legacy_defaults = {"act": "relu", "vx_head": "regress", "turn_head": "regress", "recovery_index": None,
                           "recovery_frac": 0.0, "recovery_lead": 64, "p_end": 0.0, "stream": False, "stream_reset": False, "stream_pool": 512, "stream_warmup": 2000}  # old runs keep their setup
        for key in ("index", "batch", "lr", "seed", "window", "burn", "p_start", "img", "aux_weight", "done_from_aux", "act", "vx_head",
                    "turn_head", "recovery_index", "recovery_frac", "recovery_lead", "p_end", "stream", "stream_reset", "stream_pool", "stream_warmup"):
            old = meta["cfg"].get(key, legacy_defaults[key]) if key in legacy_defaults else meta["cfg"][key]
            if "--" + key.replace("_", "-") in explicit and getattr(a, key) != old:
                ap.error(f"--resume cannot change {key}: saved {old!r}, requested {getattr(a, key)!r}")
            setattr(a, key, old)
        if a.steps < meta["step"]:
            ap.error("--steps is below the saved step")
        if "--ckpt-every" not in explicit:      # step 2 lesson: a resumed run kept the default 2,000
            a.ckpt_every = meta["cfg"].get("ckpt_every", a.ckpt_every)
    elif (run / "latest.json").exists():
        ap.error("run already has a checkpoint; use --resume or a new run directory")
    if not a.index:
        ap.error("at least one --index is required for a new run")
    if a.batch < 1 or a.steps < 0 or a.ckpt_every < 1 or a.window < 1 or not 0 <= a.burn < a.window:
        ap.error("need batch/ckpt-every/window > 0, steps >= 0, and 0 <= burn < window")
    if not 0 <= a.p_start <= 1 or not 0 < a.max_minutes <= 45 or not np.isfinite(a.lr) or a.lr <= 0:
        ap.error("need p-start in [0,1], max-minutes in (0,45], and finite lr > 0")
    if not np.isfinite(a.aux_weight) or a.aux_weight < 0 or (a.done_from_aux and a.aux_weight == 0):
        ap.error("aux-weight must be finite and >= 0; --done-from-aux requires aux-weight > 0")
    if not 0 <= a.recovery_frac < 1 or bool(a.recovery_index) != (a.recovery_frac > 0) or a.recovery_lead < 0:
        ap.error("--recovery-index and --recovery-frac in (0,1) go together; --recovery-lead >= 0")
    if not 0 <= a.p_end <= 1 - a.p_start or (a.stream and (a.p_end > 0 or a.recovery_index)) or (a.stream_reset and not a.stream) or a.stream_pool < a.batch or a.stream_warmup < 0:
        ap.error("need 0 <= p-end <= 1 - p-start; --stream takes neither --p-end nor --recovery-index; --stream-reset needs --stream; --stream-pool >= --batch")
    cfg = dict(vars(a), aux=a.aux_weight > 0)
    run.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    deadline = t0 + a.max_minutes * 60
    try:
        rows, eps = load_all(a.index, a.img, deadline=deadline)
    except TimeoutError as e:
        print(str(e), flush=True)
        return
    if not eps:
        raise ValueError("no usable training episodes")
    lens = np.array([len(e["frames"]) for e in eps], np.float64)
    rec_group, rec_n = None, 0     # not "rec": the progress log below reuses that name
    if a.recovery_index:
        try:
            rrows, reps = load_all(a.recovery_index, a.img, deadline=deadline)
        except TimeoutError as e:
            print(str(e), flush=True)
            return
        focus = [int(np.searchsorted(e["ticks"], max(0, int(r["takeover_tick"]) - a.recovery_lead))) for r, e in zip(rrows, reps)]
        focus = [min(f, len(e["frames"]) - 1) for f, e in zip(focus, reps)]
        rec_group, rec_n = (reps, focus), int(round(a.batch * a.recovery_frac))
        print(f"recovery group: {len(reps)} episodes, {sum(len(e['frames']) for e in reps)} ticks, {rec_n} of {a.batch} windows per batch", flush=True)
    load_s = time.monotonic() - t0
    print(f"loaded {len(rows)} episodes, {int(lens.sum())} ticks in {load_s:.0f} s", flush=True)
    mx.random.seed(a.seed)
    model = Policy2(**policy_config(cfg=cfg))
    mx.eval(model.parameters())
    opt = optim.Adam(learning_rate=a.lr)
    rng = np.random.default_rng(a.seed)
    streams = Streams(a.batch, model.gru.hidden_size, a.stream_pool) if a.stream else None
    step = 0
    if meta is not None:
        model.load_weights(str(d / "weights.safetensors"))
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(d / "opt.safetensors")).items()))
        rng.bit_generator.state = meta["rng"]
        # MLX's own RNG is only used for initialisation (no dropout), so it is not restored; this MLX's
        # _RandomState doesn't take item assignment. Resume stays bit-exact (tests/test_p2.py).
        step = meta["step"]
        if streams is not None:
            streams.load(np.load(d / "streams.npz"))
        print(f"resumed at step {step}", flush=True)
    elif a.init:
        model.load_weights(str(HERE / a.init))
    vg = nn.value_and_grad(model, loss_fn)
    start_step, seen, t_train = step, 0, time.monotonic()
    sup = {"loss_ticks": 0, "done_ticks": 0}       # amendment 7: supervision actually received (this invocation)
    with open(run / "train_log.jsonl", "a") as log:
        while step < a.steps:
            if time.monotonic() >= deadline:
                print("wall guard reached; saving and stopping", flush=True)
                break
            if streams is not None and step >= a.stream_warmup:
                arrays, h0 = streams.next(eps, rng, a.window, a.img)
                if a.stream_reset:
                    h0 = np.zeros_like(h0)
                (loss, metrics), g = vg(model, *[mx.array(x) for x in arrays], aux_weight=a.aux_weight, h0=mx.array(h0))
            else:
                arrays = batch(eps, lens, rng, a.batch, a.window, a.burn, a.p_start, a.img, rec_group, rec_n, a.p_end)
                (loss, metrics), g = vg(model, *[mx.array(x) for x in arrays], aux_weight=a.aux_weight)
            g, _ = optim.clip_grad_norm(g, 1.0)
            opt.update(model, g)
            mx.eval(model.parameters(), opt.state, loss, metrics)
            if streams is not None and step >= a.stream_warmup:
                streams.update(np.array(metrics["h_last"]))
            step += 1
            seen += a.batch * a.window
            sup["loss_ticks"] += int(arrays[5].sum())
            sup["done_ticks"] += int((arrays[4] * arrays[5]).sum())
            if step % 100 == 0:
                el = time.monotonic() - t_train
                rec = {"step": step, "loss": float(loss),
                       **{k: float(metrics[k]) for k in ("mse", "bce", "walk_bce", "turn_ce", "lamp_loss", "match_loss")},
                       "steps_per_s": round((step - start_step) / max(el, 1e-6), 2), "elapsed_s": round(el, 1), **sup}
                if step % 500 == 0:       # dead-encoder guard (P2-A's last conv went silent unnoticed)
                    rec["c4_alive"] = round(model.encoder_health(mx.array(arrays[0][:4, ::8].reshape(-1, *arrays[0].shape[2:])).astype(mx.float32) / 255.0), 3)
                if step % 1000 == 0:
                    rec["accuracy_scope"] = "current_training_batch_before_update"
                    for name, classes in CLASSES.items():
                        correct, counts = np.array(metrics[name + "_correct"]), np.array(metrics[name + "_count"])
                        rec[name + "_accuracy"] = {c: float(correct[k] / counts[k]) if counts[k] else None
                                                    for k, c in enumerate(classes)}
                        rec[name + "_count"] = {c: int(counts[k]) for k, c in enumerate(classes)}
                    print(rec, flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
            if step % a.ckpt_every == 0:
                save(run, step, model, opt, rng, cfg, streams)
    save(run, step, model, opt, rng, cfg, streams)
    wall = time.monotonic() - t0
    (run / "summary.json").write_text(json.dumps({"final_step": step, "start_step": start_step,
        "wall_s": round(wall, 1), "load_s": round(load_s, 1), "episodes": len(eps),
        "ticks": int(lens.sum()), "frames_seen": seen, **sup, "scope": "this_invocation",
        "budget_used": round(wall / 60 / a.max_minutes, 3)}))
    print("done", step, round(wall, 1), "s", flush=True)


if __name__ == "__main__":
    main()
