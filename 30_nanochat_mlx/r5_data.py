"""CPU preparation, pure helpers, and run manifests. Never imports MLX."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from r5_world import ACTION_ID, ACTIONS, Env, MODES, features, prompt, rollout, tune_planner
from r5_protocol import prompt_ids, action_ids

TRAIN_START, VAL_START, ROLLOUT_START = 1_000_000, 5_000_000, 6_000_000
TEST_START, SECONDARY_START = 81_241_000, 10_000_000


def json_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode_example(tok, o, action, seq_len):
    p = prompt_ids(tok, o)
    a = action_ids(tok, action)
    if len(p) + len(a) > seq_len + 1:
        raise ValueError(f"example needs {len(p)+len(a)} tokens; increase --seq-len, never truncate")
    ids = np.zeros(seq_len + 1, dtype=np.int32)
    mask = np.zeros(seq_len, dtype=np.float32)
    ids[:len(p) + len(a)] = p + a
    mask[len(p) - 1:len(p) + len(a) - 1] = 1
    assert mask.sum() == len(a)
    return ids, mask


def arrays_from_records(tok, records, seq_len):
    seqs, masks, fs, ys = [], [], [], []
    for r in records:
        ids, mask = encode_example(tok, r["observation"], r["action"], seq_len)
        seqs.append(ids)
        masks.append(mask)
        fs.append(features(r["observation"]))
        ys.append(ACTION_ID[r["action"]])
    if not records:
        raise ValueError("no training examples")
    return dict(tokens=np.array(seqs, dtype=np.uint16), masks=np.array(masks, dtype=np.uint8),
                features=np.array(fs, dtype=np.float32), targets=np.array(ys, dtype=np.int32),
                deliveries=np.array([len(r["observation"]["orders"]) for r in records], dtype=np.int32))


def sample_indices(seed, step, n, batch):
    # All sampling is a pure function of seed and optimizer step: resume has no hidden cursor.
    return np.random.default_rng(np.random.SeedSequence([seed, step, 734])).integers(n, size=batch)


def rate_plan(steps, seconds, remaining_seconds, units_per_step):
    rate = steps / max(seconds, 1e-9)
    return dict(measured_steps_per_s=rate, measured_units_per_s=rate * units_per_step,
                projected_remaining_steps=max(0, int(rate * remaining_seconds)))


def build_records(planner, start, n, deadline):
    records = []
    for seed in range(start, start + n):
        if time.monotonic() >= deadline:
            raise TimeoutError("data preparation exceeded --max-minutes; no partial dataset published")
        e = Env(seed, "train", MODES[seed % 4])
        # A small, reproducible amount of off-policy state coverage; shared by every learner.
        rng = np.random.default_rng(seed + 904)
        if rng.random() < .2:
            e.step(ACTIONS[int(rng.integers(len(ACTIONS)))])
        row = rollout(planner, e, trace=True, deadline=deadline)
        if row is None or not row["success"]:
            raise RuntimeError(f"teacher failed or timed out on seed {seed}")
        records.extend(row["trace"])
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/r5/data")
    ap.add_argument("--episodes", type=int, default=4096)
    ap.add_argument("--val-episodes", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=384)
    ap.add_argument("--max-minutes", type=float, default=2)
    a = ap.parse_args()
    assert 0 < a.episodes < 1_000_000 and 0 < a.val_episodes < 1_000_000
    started = time.monotonic()
    deadline = started + a.max_minutes * 60
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(f"use a new --out; preserving {out}")
    planner, tuning = tune_planner()
    from tokenizer import ChatTokenizer
    tok = ChatTokenizer.load()
    counts = {}
    for name, start, n in (("train", TRAIN_START, a.episodes), ("val", VAL_START, a.val_episodes)):
        records = build_records(planner, start, n, deadline)
        arr = arrays_from_records(tok, records, a.seq_len)
        np.savez(out / f"{name}.npz", **arr)
        counts[name] = dict(episodes=n, seed_start=start, examples=len(records),
                            action_counts=np.bincount(arr["targets"], minlength=len(ACTIONS)).tolist(),
                            sha256=digest(out / f"{name}.npz"))
        print(name, counts[name], flush=True)
    if time.monotonic() >= deadline:
        raise TimeoutError("preparation exceeded budget; manifest not published")
    manifest = dict(version=1, args=vars(a), counts=counts, planner=tuning,
                    tokenizer_sha256=digest("checkpoints/tokenizer/tokenizer.json"),
                    elapsed_s=time.monotonic() - started)
    json_write(out / "manifest.json", manifest)
    print("DATA READY", out, flush=True)


if __name__ == "__main__":
    main()
