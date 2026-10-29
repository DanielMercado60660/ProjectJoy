"""Claude reviewer's round-5 diagnostics. NumPy/JSON only (no MLX), ~1 minute, 4 threads.

Reads existing receipts only; trains and evaluates nothing new.
  OMP_NUM_THREADS=4 ../.venv/bin/python r5_review_analysis.py
writes results/r5/review_t2_diagnostics.json with:
  * per-run success / invalid / steps per method and per split (eval.json, secondary.json)
  * per-arm updates, rates, utilization and final validation metrics (logs/r5/<tag>/*.log)
  * reward-collection counts (results/r5/<tag>/reward/manifest.json)
  * closed-loop agreement with the teacher (the tuned planner, room order 0,1,2, which
    generated every demonstration) along each saved test trace: before vs after the first
    divergence, the verb at that divergence, invalid actions by verb, repeated-action runs
  * FP32 NumPy teacher-forced action accuracy on all 597 validation rows by target verb,
    for every turn-2 LM checkpoint (cross-checks the MLX-logged validation metric)
"""
import collections
import json
import time
from pathlib import Path
import numpy as np
from r5_world import ACTIONS, FEEDBACK, Planner
from r5_numpy_reference import Reference, score

TURN1, TURN2 = ("battery_s51", "ac_s52", "ac_s53"), ("t2_ac_s51", "t2_ac_s52", "t2_ac_s53")
METHODS = ("planner", "mlp", "scratch", "pretrained", "reward", "control")
LMS = ("pretrained", "scratch", "reward", "control")
SPLITS = ("seen", "pairs", "order", "long3", "long4")
VERBS = ("INSPECT", "GO", "TAKE", "OPEN", "SEND", "DROP")
ARMS = ("pretrained", "scratch", "mlp_final", "reward_fit", "sft_control")
TEACHER = Planner((0, 1, 2), False)  # selected in every run's data/manifest.json
# features(): ... seen(3), feedback one-hot(len(FEEDBACK)), clock(1); feedback block starts at 118
ILLEGAL_COL = 130 - 1 - len(FEEDBACK) + FEEDBACK.index("illegal")


def eval_summary(path):
    e = json.loads(Path(path).read_text())
    out = dict(complete=e["complete"], paired=e["paired_scenarios"], expected=e["expected_scenarios"],
               excluded_partial=len(e.get("excluded_partial_block") or []), methods={})
    for m in METHODS:
        eps = [x for x in e["episodes"] if x["method"] == m]
        row = dict(episodes=len(eps), success=float(np.mean([x["success"] for x in eps])),
                   invalid_rate=sum(x["invalid"] for x in eps) / sum(x["steps"] for x in eps),
                   mean_steps=float(np.mean([x["steps"] for x in eps])))
        row["by_split"] = {s: dict(n=sum(x["split"] == s for x in eps),
                                   success=float(np.mean([x["success"] for x in eps if x["split"] == s])))
                           for s in SPLITS}
        out["methods"][m] = row
    return out, e["episodes"]


def train_receipts(tag):
    out = {}
    for arm in ARMS:
        rows = [json.loads(l) for l in open(f"logs/r5/{tag}/{arm}.log") if l.startswith("{")]
        done = [r for r in rows if r.get("event") == "done"][-1]
        warm = [r for r in rows if r.get("event") == "train" and r.get("rate_warmed")]
        out[arm] = dict(updates=done["step"] - done["session_start_step"], final_step=done["step"],
                        steps_per_s_first=warm[0]["measured_steps_per_s"],
                        steps_per_s_last=warm[-1]["measured_steps_per_s"],
                        wall_utilization=done["wall_utilization"], peak_memory_gb=done["peak_memory_gb"],
                        final_val_nll=done["val_loss"], final_val_metrics=done.get("val_metrics"))
    return out


def closed_loop(episodes, method):
    eps = [e for e in episodes if e["method"] == method]
    on, off, first_verb, first_valid, inv_verb, runs, diverged = [], [], [], [], [], [], []
    first_split = []
    for e in eps:
        first = None
        for k, r in enumerate(e["trace"]):
            target = TEACHER(r["observation"])
            (on if first is None else off).append(r["action"] == target)
            if first is None and r["action"] != target:
                first = k
                first_verb.append(target.split()[0])
                first_split.append((e["split"], target.split()[0], bool(e["success"])))
                first_valid.append(bool(r["valid"]))
            if not r["valid"]:
                inv_verb.append(r["action"].split()[0])
        acts = [r["action"] for r in e["trace"]]
        best = cur = 1
        for a, b in zip(acts, acts[1:]):
            cur = cur + 1 if a == b else 1
            best = max(best, cur)
        runs.append(best)
        diverged.append((first is not None, bool(e["success"])))
    never = [s for d, s in diverged if not d]
    div = [s for d, s in diverged if d]
    return dict(episodes=len(eps), steps=len(on) + len(off),
                agreement_before_first_divergence=float(np.mean(on)), steps_before=len(on),
                agreement_after_first_divergence=float(np.mean(off)) if off else None, steps_after=len(off),
                never_diverged=len(never), never_diverged_success=float(np.mean(never)) if never else None,
                diverged=len(div), diverged_success=float(np.mean(div)) if div else None,
                first_divergence_teacher_verb=dict(collections.Counter(first_verb).most_common()),
                first_divergence_split_verb_success={f"{s}/{v}/{'success' if ok else 'fail'}": n for (s, v, ok), n in
                                                     collections.Counter(first_split).most_common()},
                first_divergence_action_valid=float(np.mean(first_valid)) if first_valid else None,
                invalid_actions_by_verb=dict(collections.Counter(inv_verb).most_common()),
                max_repeat_run_median=float(np.median(runs)),
                share_episodes_repeat_run_ge10=float(np.mean(np.array(runs) >= 10)))


def pooled_recovery(all_eps):
    out = {}
    for m in METHODS:
        c = collections.Counter()
        for eps in all_eps:
            for e in (x for x in eps if x["method"] == m):
                c["episodes"] += 1
                c["steps"] += e["steps"]
                c["assigned"] += e["mode"] != "none"
                if e["disruption"]:
                    c["exposed"] += 1
                    c["success_exposed"] += e["success"]
                    rs = e["disruption"]["recovered_step"]
                    if rs is None:
                        c["unresolved"] += 1
                    else:
                        c["recovered"] += 1
                        c["recovery_steps"] += rs - e["disruption"]["step"]
        out[m] = dict(mean_steps=c["steps"] / c["episodes"], assigned=c["assigned"], exposed=c["exposed"],
                      success_given_exposure=c["success_exposed"] / c["exposed"],
                      recovery_rate=c["recovered"] / c["exposed"],
                      recovery_steps_resolved_mean=c["recovery_steps"] / max(1, c["recovered"]),
                      unresolved=c["unresolved"])
    return out


def teacher_forced_by_verb(tag, arm):
    ref = Reference(f"checkpoints/r5_{tag}/{arm}/selected")
    exact = []
    with np.load(f"results/r5/{tag}/data/val.npz") as d:
        toks, masks, targets, deliveries = d["tokens"], d["masks"], d["targets"], d["deliveries"]
        for t, m in zip(toks, masks):
            pos = np.flatnonzero(m)
            lg = ref.logits(t[:pos[-1] + 1].astype(np.int64), pos)
            exact.append(score(lg, t[pos + 1].astype(np.int64))["exact"])
    exact, verbs = np.array(exact), np.array([ACTIONS[i].split()[0] for i in targets])
    return dict(rows=len(exact), all=float(exact.mean()),
                one_delivery=float(exact[deliveries == 1].mean()),
                two_deliveries=float(exact[deliveries == 2].mean()),
                by_verb={v: dict(n=int((verbs == v).sum()), exact=float(exact[verbs == v].mean()))
                         for v in VERBS if (verbs == v).any()})


def main():
    started = time.monotonic()
    out = dict(kind="claude_review_diagnostics_numpy_only", teacher="Planner((0,1,2), survey_first=False)",
               train_states_after_illegal_feedback={t: int(np.load(f"results/r5/{t}/data/train.npz")["features"][:, ILLEGAL_COL].sum())
                                                    for t in TURN2},
               note="Same 80 paired scenarios recur across the three training seeds; pooled counts are "
                    "descriptive, not 240 independent worlds.", runs={})
    t2_eps = []
    for tag in TURN1 + TURN2:
        run = out["runs"][tag] = {}
        run["eval"], eps = eval_summary(f"results/r5/{tag}/eval.json")
        run["secondary"], _ = eval_summary(f"results/r5/{tag}/secondary.json")
        run["train"] = train_receipts(tag)
        man = json.loads(Path(f"results/r5/{tag}/reward/manifest.json").read_text())
        run["reward_collection"] = {k: man.get(k) for k in ("attempted", "successes", "accepted_episodes",
                                    "accepted_steps", "accepted_illegal_actions", "accepted_failed_actions")}
        run["suite_elapsed_s"] = json.loads(Path(f"results/r5/{tag}/finished.json").read_text())["elapsed_s"]
        run["mlp_selected_lr"] = json.loads(Path(f"results/r5/{tag}/mlp_tuning.json").read_text())["selected"]["lr"]
        if tag in TURN2:
            t2_eps.append(eps)
            run["closed_loop"] = {m: closed_loop(eps, m) for m in METHODS if m != "planner"}
            run["teacher_forced_numpy_fp32"] = {arm: teacher_forced_by_verb(tag, arm) for arm in LMS}
    out["turn2_pooled_closed_loop"] = {m: closed_loop([e for eps in t2_eps for e in eps], m)
                                       for m in METHODS if m != "planner"}
    out["turn2_pooled_recovery"] = pooled_recovery(t2_eps)
    out["elapsed_s"] = time.monotonic() - started
    Path("results/r5/review_t2_diagnostics.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results/r5/review_t2_diagnostics.json in %.1f s" % out["elapsed_s"])


if __name__ == "__main__":
    main()
