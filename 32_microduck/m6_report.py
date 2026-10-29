"""M6: turn the evaluation files into one report (no new episodes).

    python m6_report.py
Reads results/m6/eval_*.json (bot, oracle, searchers, policies) and results/m3/eval_final.json (the
development numbers), writes results/m6/report.json and results/m6/report.md:
  - success with Wilson 95% intervals, every controller x task x set
  - the bot: development (M3, dev splits) vs reserved test, per family
  - set_lamp done-at-once floor (episodes needing no press) per set
  - go_and_return homing: how often the bot recognised its mat, success with / without it
  - policies: paired with the bot on the same episodes, when they said done, latency under load
"""
import json
from pathlib import Path

import numpy as np

from m6_eval import wilson

HERE = Path(__file__).resolve().parent
M6 = HERE / "results" / "m6"
TASKS = ("go_and_return", "set_lamp", "find_ball")
SETS = ("train", "instances-test", "arrangements-test", "appearances-test", "distractor-test")


def load_rows():
    rows = []
    for f in sorted(M6.glob("eval_*.json")):
        rows += json.loads(f.read_text())["rows"]
    return rows


def key(r):
    return (r["config_id"], r["task"], r["set"])


def frac(k, n):
    return {"k": int(k), "n": int(n), "rate": round(k / n, 3) if n else None, "ci95": wilson(k, n) if n else None}


def count(x):
    return len(x) if isinstance(x, list) else int(x or 0)


def breakdowns(rows, whos):
    """Per controller x task x set: median sim time of successes, falls, episodes touching an obstacle,
    go_and_return by target colour and wrong-pad visits, find_ball by hider."""
    out = {}
    for who in whos:
        for t in TASKS:
            for s in SETS:
                rs = [r for r in rows if r["who"] == who and r["task"] == t and r["set"] == s]
                if not rs:
                    continue
                ok = [r["sim_time"] for r in rs if r["success"]]
                b = {"median_s_success": float(np.median(ok)) if ok else None,
                     "falls": int(sum(r["falls"] for r in rs)),
                     "contact_episodes": int(sum(count(r["collision_ticks"]) > 0 for r in rs))}
                if t == "go_and_return":
                    b["wrong_pad_episodes"] = int(sum(count(r.get("wrong_pad_visits")) > 0 for r in rs))
                    b["by_colour"] = {c: frac(sum(r["success"] for r in rs if r["target"] == c),
                                              sum(r["target"] == c for r in rs))
                                      for c in sorted({r["target"] for r in rs})}
                if t == "find_ball":
                    b["by_hider"] = {h: frac(sum(r["success"] for r in rs if r["hider"] == h),
                                             sum(r["hider"] == h for r in rs))
                                     for h in sorted({r["hider"] for r in rs})}
                out[f"{who}/{t}/{s}"] = b
    return out


def main():
    rows = load_rows()
    whos = sorted({r["who"] for r in rows})
    by = {}
    for r in rows:
        by.setdefault(r["who"], {})[key(r)] = r
    rep = {"controllers": whos, "success": {}, "bot_dev_vs_test": {}, "set_lamp_floor": {}, "homing": {},
           "policies": {}}
    for who in whos:
        for t in TASKS:
            for s in SETS:
                rs = [r for r in by[who].values() if r["task"] == t and r["set"] == s]
                if rs:
                    rep["success"][f"{who}/{t}/{s}"] = frac(sum(r["success"] for r in rs), len(rs))
    dev = json.loads((HERE / "results" / "m3" / "eval_final.json").read_text())["summary"]
    for t in TASKS:
        for fam in ("instances", "arrangements", "appearances", "distractor"):
            d = dev.get(f"bot/{t}/{fam}-dev")
            te = rep["success"].get(f"bot/{t}/{fam}-test")
            if d and te:
                rep["bot_dev_vs_test"][f"{t}/{fam}"] = {"dev": frac(d["success"], d["n"]), "test": te}
        d = dev.get(f"bot/{t}/train")
        rep["bot_dev_vs_test"][f"{t}/train"] = {"m3_train_0-59": frac(d["success"], d["n"]),
                                                "m6_train_0-119": rep["success"].get(f"bot/{t}/train")}
    for s in SETS:
        rs = [r for r in by.get("bot", {}).values() if r["task"] == "set_lamp" and r["set"] == s]
        if rs:
            nopress = sum(not r["press_needed"] for r in rs)
            pn = [r for r in rs if r["press_needed"]]
            rep["set_lamp_floor"][s] = {"done_at_once": frac(nopress, len(rs)),
                                        "bot_press_needed": frac(sum(r["success"] for r in pn), len(pn))}
        rs = [r for r in by.get("bot", {}).values() if r["task"] == "go_and_return" and r["set"] == s]
        if rs:
            um = [any(e.get("event") == "home" and e.get("used_mat") for e in r["log"]) for r in rs]
            rep["homing"][s] = {"mat_recognised": frac(sum(um), len(rs)),
                                "success_with_mat": frac(sum(u and r["success"] for u, r in zip(um, rs)), sum(um)),
                                "success_without_mat": frac(sum((not u) and r["success"] for u, r in zip(um, rs)),
                                                            len(rs) - sum(um)),
                                "reached_pad_then_failed": int(sum((not r["success"]) and r["reached_pad_tick"] is not None
                                                                   for r in rs))}
    for who in whos:
        if who in ("bot", "oracle") or who.startswith("searcher_"):
            continue
        pr = list(by[who].values())
        out = {"episodes": len(pr)}
        for t in TASKS:
            rs = [r for r in pr if r["task"] == t]
            paired = [(r, by["bot"].get(key(r))) for r in rs if key(r) in by.get("bot", {})]
            dd = [r for r in rs if r["declared_done"]]
            out[t] = {"success": frac(sum(r["success"] for r in rs), len(rs)),
                      "bot_same_episodes": frac(sum(b["success"] for _, b in paired), len(paired)),
                      "only_policy": int(sum(r["success"] and not b["success"] for r, b in paired)),
                      "only_bot": int(sum(b["success"] and not r["success"] for r, b in paired)),
                      "said_done": int(len(dd)), "said_done_wrongly": int(sum(not r["success"] for r in dd)),
                      "median_s_said_done": float(np.median([r["sim_time"] for r in dd])) if dd else None,
                      "falls": int(sum(r["falls"] for r in rs)),
                      "reached_pad": int(sum(r.get("reached_pad_tick") is not None for r in rs)) if t == "go_and_return" else None}
            if t == "set_lamp":
                pn = [r for r in rs if r["press_needed"]]
                out[t]["press_needed"] = frac(sum(r["success"] for r in pn), len(pn))
                out[t]["no_press_needed"] = frac(sum(r["success"] for r in rs if not r["press_needed"]), len(rs) - len(pn))
                out[t]["presses_made"] = int(sum((r.get("presses") or 0) > 0 for r in rs))
        lat = [r["latency_ms_median"] for r in pr if "latency_ms_median" in r]
        out["latency_ms_median"] = float(np.median(lat)) if lat else None
        out["latency_ms_p99_median"] = float(np.median([r["latency_ms_p99"] for r in pr])) if lat else None
        out["load_1min_median"] = float(np.median([r["load_1min"] for r in pr])) if lat else None
        rep["policies"][who] = out
    rep["breakdowns"] = breakdowns(rows, whos)
    (M6 / "report.json").write_text(json.dumps(rep, indent=1))
    lines = ["| controller / task | " + " | ".join(SETS) + " |", "|---|" + "---|" * len(SETS)]
    for who in whos:
        for t in TASKS:
            cells = []
            for s in SETS:
                v = rep["success"].get(f"{who}/{t}/{s}")
                cells.append(f"{v['k']}/{v['n']} ({max(0.0, v['ci95'][0]):.2f}-{v['ci95'][1]:.2f})" if v else "-")
            if any(c != "-" for c in cells):
                lines.append(f"| {who} / {t} | " + " | ".join(cells) + " |")
    (M6 / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
