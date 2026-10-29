"""Phase 2, amendment 3: checks on a DAgger round before training on it.

    python p2_dagger_check.py [--tag dagger3]      # -> results/p2/<tag>_check.json

- tick share: the round's kept ticks (trainer's label_rows: through the first done, finite rows) as a
  share of all training ticks (the trainer samples windows in proportion to episode length);
- how each episode ended: the shadow bot said done / the policy said done / the budget ran out;
- done-label sanity: where the bot said done, was the simulator lamp (truth, replay-only) at the target?
  Uses the judge's lamp_final and the truth column on the done tick;
- walk-label share (bot vx > 0.15) on kept ticks, this round vs the earlier data.
Also writes episodes/<tag>_train_index.jsonl (the training index, with drop_done_tick flags).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2_train import label_rows  # noqa: E402
from record import EP_DIR  # noqa: E402

BASE = ["results/m5/bc_index.jsonl", "episodes/dagger_index.jsonl", "episodes/dagger2_index.jsonl"]


def rows(index):
    for line in open(HERE / index):
        if line.strip():
            r = json.loads(line)
            if "error" not in r and r.get("split", "train") == "train":
                yield r


def kept(path):
    with np.load(HERE / path) as z:
        cb = z["cmd_bot"]
    idx, _ = label_rows(cb)
    return len(idx), int((cb[idx, 0] > 0.15).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dagger3")
    a = ap.parse_args()
    base_t = base_w = 0
    for ix in BASE:
        for r in rows(ix):
            t, w = kept(r["path"])
            base_t, base_w = base_t + t, base_w + w
    new_t = new_w = 0
    ends = {"bot_done": 0, "policy_done": 0, "budget": 0}
    lamp_wrong, lamp_wrong_eps, policy_done_right, presses = 0, [], 0, 0
    n = 0
    for r in rows(f"episodes/{a.tag}_index.jsonl"):
        n += 1
        t, w = kept(r["path"])
        new_t, new_w = new_t + t, new_w + w
        with np.load(HERE / r["path"]) as z:
            H = json.loads(str(z["header"]))
            tr = z["truth"]
        res = H["result"]
        presses += int(res["presses"])
        if H["dagger"]["bot_finished_tick"] is not None:
            ends["bot_done"] += 1
            truth_on_done_tick = bool(tr[-1, 5] > 0.5)
            if bool(res["lamp_final"]) != bool(res["target"]) or truth_on_done_tick != bool(res["target"]):
                lamp_wrong += 1
                lamp_wrong_eps.append(r["config"])
        elif res["declared_done"]:
            ends["policy_done"] += 1
            policy_done_right += bool(res["success"])
        else:
            ends["budget"] += 1
    out = {"tag": a.tag, "episodes": n, "ends": ends,
           "policy_done_successes": policy_done_right,
           "bot_done_with_lamp_wrong": [lamp_wrong, ends["bot_done"]], "bot_done_lamp_wrong_configs": lamp_wrong_eps,
           "exclude_done_labels": lamp_wrong > 0.05 * max(1, ends["bot_done"]),
           "presses_executed": presses,
           "kept_ticks": {"this_round": new_t, "earlier": base_t,
                          "share_of_training": round(new_t / (new_t + base_t), 4)},
           "walk_label_share": {"this_round": round(new_w / max(1, new_t), 4), "earlier": round(base_w / max(1, base_t), 4)}}
    (HERE / "results" / "p2" / f"{a.tag}_check.json").write_text(json.dumps(out, indent=1))
    # The training index: the same rows, with drop_done_tick on the flagged episodes when amendment 3's
    # 5% bar is exceeded (p2_train then removes that final done tick).
    bad = set(lamp_wrong_eps) if out["exclude_done_labels"] else set()
    with open(EP_DIR / f"{a.tag}_train_index.jsonl", "w") as f:
        for line in open(HERE / "episodes" / f"{a.tag}_index.jsonl"):
            r = json.loads(line)
            if r.get("config") in bad:
                r["drop_done_tick"] = True
            f.write(json.dumps(r) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
