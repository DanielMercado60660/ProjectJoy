"""Phase 2, amendment 6: training indexes for the recovery round.

    python p2_recovery_index.py         # -> episodes/{dagger,dagger2}_filtered_index.jsonl, episodes/recovery_index.jsonl,
                                        #    results/p2/recovery_index_report.json
Done-label filter: in a policy-driven episode, the bot's first done tick is flagged drop_done_tick (p2_train
removes it, never relabels it) when the simulator lamp at that tick differs from the set_lamp target. The
truth column is read only for this; it never becomes a label or an input.
Recovery set: check A's takeover_100/250/400 episodes (train configs 0-99, k=4) + p2_recovery.py's
episodes (train configs 100-299, k=4), each with its takeover_tick.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def done_lamp_wrong(path):
    """True when the bot's first done tick in a set_lamp episode comes with the simulator lamp off target."""
    with np.load(HERE / path) as z:
        H = json.loads(str(z["header"]))
        if H["task"] != "set_lamp":
            return False
        cb, tr = z["cmd_bot"], z["truth"]
    done = np.flatnonzero(cb[:, 7] > 0.5)
    if not len(done):
        return False
    return bool(tr[done[0], 5] > 0.5) != (H["instruction"]["target"] == "on")


def row_for(path):
    with np.load(path) as z:
        H = json.loads(str(z["header"]))
    rel = str(Path(path).resolve().relative_to(HERE))
    return {"path": rel, "config": H["config_id"], "split": H["split"], "task": H["task"], "k": H["k"],
            "takeover_tick": H["dagger"]["takeover_tick"], "success": H["result"]["success"], "ticks": H["ticks"]}


def main():
    report = {}
    for name in ("dagger", "dagger2"):
        rows = [json.loads(l) for l in open(HERE / "episodes" / f"{name}_index.jsonl") if l.strip()]
        n = 0
        with open(HERE / "episodes" / f"{name}_filtered_index.jsonl", "w") as f:
            for r in rows:
                if "error" not in r and done_lamp_wrong(r["path"]):
                    r["drop_done_tick"] = True
                    n += 1
                f.write(json.dumps(r) + "\n")
        report[name] = {"episodes": len(rows), "done_dropped": n}
    paths = []
    for arm in ("takeover_100", "takeover_250", "takeover_400"):
        paths += sorted(glob.glob(str(HERE / "episodes" / "p2_takeover" / arm / "set_lamp" / "*.npz")))
    paths += sorted(glob.glob(str(HERE / "episodes" / "p2_recovery" / "set_lamp" / "*.npz")))
    n, by_t = 0, {}
    with open(HERE / "episodes" / "recovery_index.jsonl", "w") as f:
        for p in paths:
            r = row_for(p)
            if done_lamp_wrong(r["path"]):
                r["drop_done_tick"] = True
                n += 1
            by_t[r["takeover_tick"]] = by_t.get(r["takeover_tick"], 0) + 1
            f.write(json.dumps(r) + "\n")
    report["recovery"] = {"episodes": len(paths), "done_dropped": n, "by_takeover_tick": by_t}
    (HERE / "results" / "p2" / "recovery_index_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
