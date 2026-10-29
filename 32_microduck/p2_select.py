"""Phase 2: closed-loop dev check and checkpoint pick (results/p2/protocol.json).

    python p2_select.py <run_dir | weights> [...]     # -> results/p2/select_<name>.json
Every checkpoint (or the given weights) drives set_lamp on instances/arrangements/appearances/distractor-dev
0000-0009 (k=0, 40 episodes), and gets the open-loop lamp diagnostic (p2_diag.py). Pick within a run:
most press-needed successes, then higher lamp AUC, then most no-press successes, then the earlier step.
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learner  # noqa: E402
from m6_eval import load_set, one  # noqa: E402

SETS = ("instances-dev", "arrangements-dev", "appearances-dev", "distractor-dev")
_PC = learner.PolicyController


def controller(weights):
    cfgp = Path(weights).parent / "p2_config.json"
    if cfgp.exists():
        from p2_learner import P2Controller
        return P2Controller(weights, json.loads(cfgp.read_text()))
    return _PC(weights)


def one_any(args):
    learner.PolicyController = controller          # m6_eval.one imports it at call time
    return one(args)


def summarise(rows):
    pn = [r for r in rows if r["press_needed"]]
    npn = [r for r in rows if not r["press_needed"]]
    return {"n": len(rows), "success": int(sum(r["success"] for r in rows)),
            "press_needed": [int(sum(r["success"] for r in pn)), len(pn)],
            "no_press": [int(sum(r["success"] for r in npn)), len(npn)],
            "presses_made": int(sum((r.get("presses") or 0) > 0 for r in rows)),
            "said_done_without_press_pn": int(sum(r["declared_done"] and not (r.get("presses") or 0) for r in pn)),
            "falls": int(sum(r["falls"] for r in rows))}


def scheduled_checkpoints(p):
    """{ckpt name: weights path} for a run directory's scheduled checkpoints only.

    A save made when the 45-min wall guard stops a run, or an extra save after --resume (which once fell back
    to --ckpt-every 2000), is not a candidate. Decided 2026-10-27 12:12, before any P2-E / P2-A' seed-1
    selection episode; written into amendment 6. Earlier runs only have scheduled checkpoints."""
    p = Path(p)
    every = json.loads((p / "p2_config.json").read_text()).get("ckpt_every") if (p / "p2_config.json").exists() else None
    if every is None:
        metas = sorted(p.glob("ckpt_*/meta.json"))
        every = json.loads(metas[0].read_text())["cfg"]["ckpt_every"] if metas else None
    return {c.name: str(c / "weights.safetensors") for c in sorted(p.glob("ckpt_*"))
            if (c / "weights.safetensors").exists() and (every is None or int(c.name[5:]) % every == 0)}


def main():
    import p2_diag
    for arg in sys.argv[1:]:
        p = HERE / arg
        if p.is_dir():
            policies = scheduled_checkpoints(p)
            name = p.name
        else:
            # A bare <run>/<ckpt>/weights.safetensors is named <run>_<ckpt>, so several don't overwrite one file.
            name = f"{p.parent.parent.name}_{p.parent.name}" if p.stem == "weights" else p.stem
            policies = {name: str(p)}
        jobs = [(c, s, "set_lamp", who, policies) for who in policies for s in SETS for c in load_set(s, 10)]
        with Pool(int(os.environ.get("P2_PROCS", "4"))) as pool:
            rows = [r for r in pool.map(one_any, jobs, chunksize=1) if r is not None]
        table = {}
        for who, w in policies.items():
            t = summarise([r for r in rows if r["who"] == who])
            d = p2_diag.run(w)
            t.update(auc=d["auc"], early_done_mismatch=[d["early_done_mismatch"], d["mismatch"]], aux_acc=d.get("aux_acc"))
            table[who] = t
            print(name, who, t, flush=True)
        pick = sorted(table, key=lambda k: (-table[k]["press_needed"][0], -(table[k]["auc"] or 0), -table[k]["no_press"][0], k))[0]
        (HERE / "results" / "p2" / f"select_{name}.json").write_text(json.dumps(
            {"run": name, "table": table, "chosen": pick, "weights": policies[pick], "rows": rows}, indent=1, default=float))
        print(name, "chosen:", pick, flush=True)


if __name__ == "__main__":
    main()
