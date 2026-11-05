"""Round 8 Gate B metrics from the saved runs (definitions frozen in results/r8/stabcam/protocol.json).

    ../.venv/bin/python r8_stabcam_analyze.py      -> results/r8/stabcam/wobble.json + a printed table

Wobble: the stream's effective view orientation (yaw, pitch, roll), minus its centred 1 s moving mean, std (deg).
  Scripted segments: m3_gaze.detrended_std on the 50 Hz window 150..500 (M3.1's number). Bot episodes: the whole
  episode detrended, std over walking steps (executed vx > 0.15).
Frame change: mean |grey difference| between consecutive 10 Hz frames at 64 px (the 4A pipeline), walking only;
  eis streams over pixels valid in both frames.
Phase-locked: 50 Hz frames at each left touchdown (sensor-only stance switch); consecutive kept pairs (gap <= 1 s,
  walking throughout) vs (a) unlocked fixed clocks over the same walking stretches, random start phase, 8 draws:
  one at the kept frames' mean gap ("same rate") and one at 2 Hz (every 25 steps); (b) random-phase pairs with the
  same gap as each kept pair (phase-matched by construction when the gap is one stride: they measure the real motion
  per stride). The sampled sequences' own wobble: std of the detrended stream angle at the kept / clock frames.
  pl2 = both touchdowns. (Amendment 2026-11-05 15:47, before any measurement run: (a) added, (b) relabelled.)
"""
import json
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402
from m3_gaze import detrended_std  # noqa: E402

GATE = {"yaw": 2.0, "pitch": 1.0}
STREAMS_OFF = ("head", "trunk", "oracle", "eis", "eis_lag")
STREAMS_ON = ("reflex", "eis_reflex", "eis_reflex_lag")
SENSOR_ONLY = ("head", "reflex", "eis", "eis_lag", "eis_lagc", "eis_reflex", "eis_reflex_lag", "eis_reflex_lagc",
               "pl_head", "pl_eis", "pl2_head", "pl2_eis")
CLOCK_2HZ = 25                         # 50 Hz steps: a 2 Hz consumer's clock (every 5th 10 Hz tick)
N_CONTROL = 8
MAX_GAP = {"pl": 50, "pl2": 30}       # 50 Hz steps: a stride is ~27, half a stride ~13


def _load(path):
    from r8_stabcam_run import load_npz
    return load_npz(path)


def frame_changes(F, V=None):
    """Per consecutive pair: mean |diff| (over pixels valid in both when V is given)."""
    F = F.astype(np.float32)
    d = np.abs(np.diff(F, axis=0))
    if V is None:
        return d.mean((1, 2))
    both = V[1:] & V[:-1]
    n = both.sum((1, 2))
    return np.where(n > 0, (d * both).sum((1, 2)) / np.maximum(n, 1), np.nan)


def phase_locked(r, walk_steps, kind="pl", lo=0, hi=None):
    """Kept-pair vs random-phase same-gap control, for the 50 Hz head and eis frames, plus the angle residual at
    kept frames (the phase-locked stream's own wobble)."""
    hi = len(r["walk"]) if hi is None else hi
    s50 = r["s50"]
    if len(s50) == 0:
        return None
    pos = {int(s): i for i, s in enumerate(s50)}
    walk = np.zeros(len(r["walk"]), bool)
    walk[walk_steps] = True
    ev = r["events"]
    want = (0,) if kind == "pl" else (0, 1)
    kept = [int(s) for s, e in ev if e in want and lo <= s < hi and s in pos and walk[s]]
    # contiguous walking runs (with frames) for pair validity
    ok = np.zeros(len(walk), bool)
    ok[[s for s in pos if walk[s] and lo <= s < hi]] = True
    run_id = np.cumsum(~ok)                                  # same id <=> no gap in between
    pairs = [(a, b) for a, b in zip(kept[:-1], kept[1:]) if b - a <= MAX_GAP[kind] and run_id[a] == run_id[b] and ok[a] and ok[b]]
    if not pairs:
        return None
    seed = zlib.crc32(json.dumps(r["meta"], sort_keys=True).encode() + kind.encode())
    rng = np.random.default_rng(seed)
    cand = np.nonzero(ok)[0]
    gap_mean = float(np.mean([b - a for a, b in pairs]))
    out = {"kept": len(kept), "pairs": len(pairs), "gap_steps_mean": gap_mean}
    clocks = {"clock_same_rate": int(round(gap_mean)), "clock_2hz": CLOCK_2HZ}
    clock_pairs = {}
    for cname, P in clocks.items():
        cp = []
        for _ in range(N_CONTROL):
            for rid in np.unique(run_id[ok]):
                idx = np.nonzero(ok & (run_id == rid))[0]
                if len(idx) <= P:
                    continue
                s0 = int(idx[0] + rng.integers(P))
                ss = [x for x in range(s0, int(idx[-1]) + 1, P)]
                cp.extend(zip(ss[:-1], ss[1:]))
        clock_pairs[cname] = cp
    for stream in ("head", "eis"):
        F = r["f50"][stream].astype(np.float32)
        V = r["v50"]["eis"] if stream == "eis" else None

        def change(a, b):
            ia, ib = pos[a], pos[b]
            d = np.abs(F[ib] - F[ia])
            if V is None:
                return float(d.mean())
            m = V[ia] & V[ib]
            return float(d[m].mean()) if m.any() else np.nan
        locked = [change(a, b) for a, b in pairs]
        ctrl = []
        for a, b in pairs:
            g = b - a
            starts = [s for s in cand if s + g < len(ok) and ok[s + g] and run_id[s] == run_id[s + g]]
            if not starts:
                continue
            for s in rng.choice(starts, size=min(N_CONTROL, len(starts)), replace=False):
                ctrl.append(change(int(s), int(s) + g))
        out[f"{stream}_change_locked"] = float(np.nanmean(locked))
        out[f"{stream}_change_samegap_randphase"] = float(np.nanmean(ctrl)) if ctrl else None
        for cname, cp in clock_pairs.items():
            out[f"{stream}_change_{cname}"] = float(np.nanmean([change(a, b) for a, b in cp])) if cp else None
    # the sampled sequences' own wobble (std of the detrended angle at the sampled steps), per stream and axis
    for stream in ("head", "eis"):
        res = np.stack([S.detrended(r["ang"][stream][lo:hi, i]) for i in range(3)], 1)
        k_idx = np.array([s - lo for s in kept], int)
        out[f"{stream}_wobble_locked"] = np.degrees(res[k_idx].std(0)).round(3).tolist()
        for cname, cp in clock_pairs.items():
            steps = np.array(sorted(set([a for a, _ in cp] + [b for _, b in cp])), int) - lo
            out[f"{stream}_wobble_{cname}"] = np.degrees(res[steps].std(0)).round(3).tolist() if len(steps) > 2 else None
    return out


def pl_wobble(r, stream_ang, walk_steps, kind="pl", lo=0, hi=None, window=False):
    """std (deg) of the detrended angle at kept frames: how much the phase-locked views still differ."""
    hi = len(r["walk"]) if hi is None else hi
    walk = np.zeros(len(r["walk"]), bool)
    walk[walk_steps] = True
    want = (0,) if kind == "pl" else (0, 1)
    kept = np.array([int(s) for s, e in r["events"] if e in want and lo <= s < hi and walk[s]], int)
    if len(kept) < 3:
        return None
    if window:                                   # scripted: detrend inside the window like detrended_std
        res = np.stack([S.detrended(stream_ang[lo:hi, i]) for i in range(3)], 1)
        vals = res[kept - lo]
    else:
        res = np.stack([S.detrended(stream_ang[:, i]) for i in range(3)], 1)
        vals = res[kept]
    return [float(np.degrees(vals[:, i].std())) for i in range(3)]


def scripted_run(r):
    m = r["meta"]
    lo, hi = S.WINDOW
    out = {"wobble": {}, "frame_change": {}, "valid": {}}
    for n, A in r["ang"].items():
        out["wobble"][n] = [detrended_std(A[lo:hi, i]) for i in range(3)]
    # tick-end frames: frame j = end of tick j; window ticks 30..99
    j0, j1 = lo // 5, hi // 5
    for n, F in r["f10"].items():
        V = r["v10"].get(n)
        ch = frame_changes(F[j0:j1], None if V is None else V[j0:j1])
        out["frame_change"][n] = {"mean": float(np.nanmean(ch)), "median": float(np.nanmedian(ch))}
        if V is not None:
            vf = V[j0:j1].mean((1, 2))
            out["valid"][n] = {"mean": float(vf.mean()), "min": float(vf.min())}
    out["clamped"] = float(r["clamped"][lo:hi].mean())
    out["offset_p99"] = float(np.percentile(r["offset_deg"][lo:hi], 99))
    if m["mode"] == "off" and m["vx"] > 0 and "f50" in r and len(r.get("s50", [])):
        steps = np.arange(lo, hi)
        for kind in ("pl", "pl2"):
            out[kind] = phase_locked(r, steps, kind, lo, hi)
            out[f"{kind}_wobble"] = {s: pl_wobble(r, r["ang"][s], steps, kind, lo, hi, window=True) for s in ("head", "eis")}
    return out


def bot_run(r):
    walk = r["walk"].astype(bool)
    steps = np.nonzero(walk)[0]
    out = {"wobble": {}, "frame_change": {}, "valid": {}, "walk_frac": float(walk.mean()), "meta": r["meta"]}
    if walk.sum() < 50:                              # under 1 s of walking: nothing to score
        out["skipped"] = "under 1 s of walking"
        return out
    for n, A in r["ang"].items():
        res = np.stack([S.detrended(A[:, i]) for i in range(3)], 1)
        out["wobble"][n] = [float(np.degrees(res[walk, i].std())) for i in range(3)]
    te = np.nonzero(r["tick_end"])[0]                 # step index of each tick-end frame
    walk_tick = walk[te]                             # the command of the tick that ended there
    for n, F in r["f10"].items():
        V = r["v10"].get(n)
        ch = frame_changes(F, V)
        sel = walk_tick[1:]
        out["frame_change"][n] = {"mean": float(np.nanmean(ch[sel])), "median": float(np.nanmedian(ch[sel]))}
        if V is not None:
            vf = V.mean((1, 2))[walk_tick]
            out["valid"][n] = {"mean": float(vf.mean()), "min": float(vf.min())}
    out["clamped"] = float(r["clamped"][walk].mean())
    out["offset_p99"] = float(np.percentile(r["offset_deg"][walk], 99))
    if r["meta"]["mode"] == "off":
        for kind in ("pl", "pl2"):
            out[kind] = phase_locked(r, steps, kind)
            out[f"{kind}_wobble"] = {s: pl_wobble(r, r["ang"][s], steps, kind) for s in ("head", "eis")}
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return np.mean(xs, 0).round(3).tolist() if xs else None


def _agg(dicts):
    """Mean over runs of each field of the phase-locked dicts (lists elementwise; missing values skipped)."""
    out = {}
    for k in dicts[0]:
        vals = [d.get(k) for d in dicts if d.get(k) is not None]
        if not vals:
            out[k] = None
        elif isinstance(vals[0], list):
            out[k] = np.mean(vals, 0).round(3).tolist()
        else:
            out[k] = float(np.nanmean(vals))
    return out


def analyze(sub="runs", tag=""):
    runs = sorted((S.DATA / sub).glob("*.npz"))
    scripted, bots = {}, {}
    for p in runs:
        r = _load(p)
        m = r["meta"]
        if m["kind"] == "scripted":
            scripted.setdefault(f"vx{m['vx']}_wz{m['wz']}", {}).setdefault(m["mode"], []).append(scripted_run(r))
        else:
            b = bot_run(r)
            if "skipped" in b:
                print("skipped", p.name, b["skipped"])
            else:
                bots.setdefault(m["mode"], []).append(b)
    rep = {"source": f"data/r8_stabcam/{sub}", "gate": GATE, "scripted": {}, "bot": {}}
    # ---- scripted: per condition mean over configs
    for cond, modes in scripted.items():
        c = {}
        for mode, rs in modes.items():
            for n in rs[0]["wobble"]:
                c.setdefault("wobble", {})[n] = _mean([x["wobble"][n] for x in rs])
            for n in rs[0]["frame_change"]:
                c.setdefault("frame_change", {})[n] = {"mean": float(np.mean([x["frame_change"][n]["mean"] for x in rs])),
                                                       "median": float(np.mean([x["frame_change"][n]["median"] for x in rs]))}
            for n in rs[0]["valid"]:
                c.setdefault("valid", {})[n] = {"mean": float(np.mean([x["valid"][n]["mean"] for x in rs])),
                                                "min": float(np.min([x["valid"][n]["min"] for x in rs]))}
            c.setdefault("clamped", {})[mode] = float(np.mean([x["clamped"] for x in rs]))
            c.setdefault("offset_p99_max", {})[mode] = float(np.max([x["offset_p99"] for x in rs]))
            if mode == "off" and rs[0].get("pl"):
                for kind in ("pl", "pl2"):
                    pls = [x[kind] for x in rs if x.get(kind)]
                    c[kind] = _agg(pls)
                    c[f"{kind}_wobble"] = {s: _mean([x[f"{kind}_wobble"][s] for x in rs]) for s in ("head", "eis")}
            c.setdefault("n", {})[mode] = len(rs)
        rep["scripted"][cond] = c
    walk_conds = [k for k in rep["scripted"] if not k.startswith("vx0.0")]
    if walk_conds:
        wa = {"wobble": {}, "frame_change": {}, "valid": {}}
        for n in rep["scripted"][walk_conds[0]]["wobble"]:
            wa["wobble"][n] = _mean([rep["scripted"][k]["wobble"][n] for k in walk_conds])
        for n in rep["scripted"][walk_conds[0]]["frame_change"]:
            wa["frame_change"][n] = {s: float(np.mean([rep["scripted"][k]["frame_change"][n][s] for k in walk_conds])) for s in ("mean", "median")}
        for n in rep["scripted"][walk_conds[0]].get("valid", {}):
            wa["valid"][n] = {"mean": float(np.mean([rep["scripted"][k]["valid"][n]["mean"] for k in walk_conds])),
                              "min": float(np.min([rep["scripted"][k]["valid"][n]["min"] for k in walk_conds]))}
        for kind in ("pl", "pl2"):
            if all(rep["scripted"][k].get(kind) for k in walk_conds):
                wa[kind] = _agg([rep["scripted"][k][kind] for k in walk_conds])
                wa[f"{kind}_wobble"] = {s: _mean([rep["scripted"][k][f"{kind}_wobble"][s] for k in walk_conds]) for s in ("head", "eis")}
        wa["clamped_off"] = float(np.mean([rep["scripted"][k]["clamped"].get("off", 0) for k in walk_conds]))
        rep["scripted_walking_average"] = wa
    # ---- bot
    for mode, rs in bots.items():
        b = {"episodes": [x["meta"] for x in rs], "walk_frac": [x["walk_frac"] for x in rs], "wobble": {}, "frame_change": {}, "valid": {}}
        for n in rs[0]["wobble"]:
            b["wobble"][n] = _mean([x["wobble"][n] for x in rs])
        for n in rs[0]["frame_change"]:
            b["frame_change"][n] = {"median_of_medians": float(np.median([x["frame_change"][n]["median"] for x in rs])),
                                    "mean_of_means": float(np.mean([x["frame_change"][n]["mean"] for x in rs]))}
        for n in rs[0]["valid"]:
            b["valid"][n] = {"mean": float(np.mean([x["valid"][n]["mean"] for x in rs])), "min": float(np.min([x["valid"][n]["min"] for x in rs]))}
        b["clamped"] = float(np.mean([x["clamped"] for x in rs]))
        b["offset_p99_max"] = float(np.max([x["offset_p99"] for x in rs]))
        if mode == "off":
            for kind in ("pl", "pl2"):
                pls = [x[kind] for x in rs if x.get(kind)]
                if pls:
                    b[kind] = _agg(pls)
                    b[f"{kind}_wobble"] = {s: _mean([x[f"{kind}_wobble"][s] for x in rs if x[f"{kind}_wobble"][s] is not None]) for s in ("head", "eis")}
        b["successes"] = int(sum(x["meta"]["success"] for x in rs))
        b["falls"] = int(sum(x["meta"]["falls"] for x in rs))
        rep["bot"][mode] = b
    rep["gate_check"] = gate_check(rep)
    name = "wobble.json" if not tag else f"wobble_{tag}.json"
    (S.OUT / name).write_text(json.dumps(rep, indent=1))
    print_table(rep)
    return rep


def gate_check(rep):
    """Primary: scripted walking average, yaw <= 2.0 and pitch <= 1.0 deg, sensor-only streams. Bot: same bars, descriptive."""
    out = {}
    wa = rep.get("scripted_walking_average")
    if wa:
        rows = dict(wa["wobble"])
        for kind in ("pl", "pl2"):
            for s in ("head", "eis"):
                if wa.get(f"{kind}_wobble", {}).get(s):
                    rows[f"{kind}_{s}"] = wa[f"{kind}_wobble"][s]
        out["scripted"] = {n: {"yaw": v[0], "pitch": v[1], "roll": v[2], "sensor_only": n in SENSOR_ONLY,
                               "pass": bool(v[0] <= GATE["yaw"] and v[1] <= GATE["pitch"])} for n, v in rows.items() if v}
    for mode, b in rep.get("bot", {}).items():
        rows = dict(b["wobble"])
        for kind in ("pl", "pl2"):
            for s in ("head", "eis"):
                if b.get(f"{kind}_wobble", {}).get(s):
                    rows[f"{kind}_{s}"] = b[f"{kind}_wobble"][s]
        out[f"bot_{mode}"] = {n: {"yaw": v[0], "pitch": v[1], "roll": v[2], "sensor_only": n in SENSOR_ONLY,
                                  "pass": bool(v[0] <= GATE["yaw"] and v[1] <= GATE["pitch"])} for n, v in rows.items() if v}
    return out


def print_table(rep):
    wa = rep.get("scripted_walking_average")
    if wa:
        print("scripted walking average (train-0020..0025, 5 walking segments): wobble yaw/pitch/roll deg | frame change mean")
        for n, v in wa["wobble"].items():
            fc = wa["frame_change"].get(n, {}).get("mean")
            vf = wa["valid"].get(n, {}).get("mean")
            print(f"  {n:16s} {v[0]:5.2f} {v[1]:5.2f} {v[2]:5.2f} | {fc if fc is None else round(fc, 2)}" + (f" valid {vf:.3f}" if vf else ""))
        for kind in ("pl", "pl2"):
            if wa.get(kind):
                print(f"  {kind}: ", wa[kind])
    for cond, c in rep["scripted"].items():
        print(cond, {n: v for n, v in c["wobble"].items()})
    for mode, b in rep.get("bot", {}).items():
        print("bot", mode, "successes", b["successes"], "falls", b["falls"])
        for n, v in b["wobble"].items():
            fc = b["frame_change"].get(n, {})
            print(f"  {n:16s} {v[0]:5.2f} {v[1]:5.2f} {v[2]:5.2f} | fc median {fc.get('median_of_medians', float('nan')):.2f} mean {fc.get('mean_of_means', float('nan')):.2f}")
        for kind in ("pl", "pl2"):
            if b.get(kind):
                print(f"  {kind}: ", b[kind])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="runs")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    analyze(a.sub, a.tag)
