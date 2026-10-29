"""Fresh solver-labelled boards and auditable exclusions, no MLX imports."""
import glob
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import c4
import solver
from r5_core import canonical_keys

SEEDS = dict(train=50241011, val=50241023, test=50241037)
RANGES = ((10, 15), (16, 23), (24, 32))


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def exclusions():
    files = sorted(glob.glob("checkpoints/**/replay.npz", recursive=True))
    if not files:
        raise RuntimeError("No retained replay found: cannot audit fresh-data exclusion")
    keys, manifest = set(), []
    for path in files:
        with np.load(path) as d:
            n = int(d["n"]) if "n" in d else len(d["pos"])
            keys.update(map(int, canonical_keys(d["pos"][:n], d["mask"][:n])))
        manifest.append(dict(path=path, sha256=fingerprint(path), rows=n))
    path = Path("results/solver/positions.json")
    if path.exists():
        for rows in json.loads(path.read_text())["sets"].values():
            pm = np.array([c4.from_moves(row) for row in rows], dtype=np.int64)
            keys.update(map(int, canonical_keys(pm[:, 0], pm[:, 1])))
    return keys, dict(replays=manifest, excluded_keys=len(keys),
                     historical_limit="Earlier overwritten replay buffers are unavailable; no claim of exclusion from them.")


def candidate(rng, group, deadline=float("inf")):
    """One board per fresh trajectory; half random, half safe-move rollout.

    Reject terminal trajectories, never replace a winning move with a quiet move.
    Target/group is redrawn only after a board has been obtained.
    """
    lo, hi = RANGES[group]
    target = int(rng.integers(lo, hi + 1))
    safe = bool(rng.integers(2))
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("Position generation deadline")
        p, m = np.int64(0), np.int64(0)
        moves = []
        for _ in range(target):
            legal = np.flatnonzero(c4.legal_moves(m))
            wins = [c for c in legal if c4.winning_move(p, m, c)]
            if safe and wins:
                break
            if safe:
                block = [c for c in legal if c4.winning_move(p ^ m, m, c)]
                if len(block) == 1:
                    legal = np.asarray(block)
                else:
                    ok = []
                    for c in legal:
                        p2, m2 = c4.play(p, m, c)
                        if not any(c4.winning_move(p2, m2, d) for d in range(7)
                                   if c4.can_play(m2, d)):
                            ok.append(c)
                    if ok:
                        legal = np.asarray(ok)
            c = int(rng.choice(legal))
            if c4.winning_move(p, m, c):
                break
            p, m = c4.play(p, m, c)
            moves.append(c)
        if len(moves) == target:
            return dict(pos=int(p), mask=int(m), group=group, safe=safe, moves=moves)


class Builder:
    def __init__(self, directory, seeds=None, extra_keys=(), extra_manifest=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seen, manifest = exclusions()
        self.seen.update(map(int, extra_keys))
        if extra_manifest is not None:
            manifest["extra_sources"] = extra_manifest
        self.manifest = manifest
        self.seeds = SEEDS if seeds is None else seeds
        assert set(self.seeds) == set(SEEDS)
        self.rows = {k: [] for k in self.seeds}
        self.rng = {k: np.random.default_rng(v) for k, v in self.seeds.items()}
        self.attempts = {k: 0 for k in SEEDS}
        self.rejects = {k: 0 for k in SEEDS}
        self.S = solver.Solver(22)  # about 38 MB, shared across boards
        saved = self.directory / "builder.json"
        if saved.exists():
            state = json.loads(saved.read_text())
            assert state["manifest"]["replays"] == manifest["replays"], "Replay changed during run"
            assert state["seeds"] == self.seeds, "Data seed changed during resume"
            assert state["manifest"].get("extra_sources") == manifest.get("extra_sources"), "Exclusions changed during resume"
            self.attempts, self.rejects = state["attempts"], state["rejects"]
            for k in SEEDS:
                self.rng[k].bit_generator.state = state["rng"][k]
                path = self.directory / (k + ".json")
                self.rows[k] = json.loads(path.read_text())
                self.seen.update(int(x["key"]) for x in self.rows[k])

    def add(self, split, n, deadline, nodes=3_000_000):
        start, before = time.monotonic(), len(self.rows[split])
        while len(self.rows[split]) < before + n and time.monotonic() < deadline:
            group = len(self.rows[split]) % 3
            try:
                row = candidate(self.rng[split], group, deadline)
            except TimeoutError:
                break
            key = int(canonical_keys(np.array([row["pos"]]), np.array([row["mask"]]))[0])
            if key in self.seen:
                self.rejects[split] += 1
                continue
            self.attempts[split] += 1
            self.S.reset()
            vals = self.S.move_values(row["pos"], row["mask"], node_limit=nodes)
            # Preserve every timeout, including hard groups. Headline population is
            # explicitly fully-solvable-under-node-cap, not every generated position.
            if np.any(vals == solver.UNSOLVED):
                with open(self.directory / "unsolved.jsonl", "a") as f:
                    f.write(json.dumps(dict(split=split, **row, values=vals.tolist(), nodes=nodes)) + "\n")
                continue
            row.update(key=key, values=vals.tolist())
            self.rows[split].append(row)
            self.seen.add(key)
        return dict(added=len(self.rows[split])-before, seconds=time.monotonic()-start)

    def save(self):
        for k in SEEDS:
            path = self.directory / (k + ".json")
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.rows[k])); tmp.replace(path)
        state = dict(manifest=self.manifest, attempts=self.attempts, rejects=self.rejects,
                     rng={k: r.bit_generator.state for k, r in self.rng.items()}, seeds=self.seeds)
        path = self.directory / "builder.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2)); tmp.replace(path)


def arrays(rows):
    return {k: np.asarray([r[k] for r in rows]) for k in ("pos", "mask", "group", "values", "key")}
