"""Cross-review checks (added by the reviewer; see REVIEW.md). Reads the existing checkpoints and
results, runs a few cheap evaluations, and writes results/review_checks[_<pair>].json.

  ../bin/gpu_run ../.venv/bin/python review_checks.py                         # fwd vs bwd
  SKILL_PAIR=fwd_cruise ../bin/gpu_run ../.venv/bin/python review_checks.py   # fwd vs cruise

1. Single-specialist baseline, selected on VALIDATION seeds: run one specialist under both labels.
   Each specialist is scored on the other task on VAL_SEEDS; the pick per training seed uses
   (a) the merges' own criterion (summed raw validation return over both tasks) and
   (b) mean retained on validation (return / that task's specialist validation return).
   Eval-seed numbers come from the merge JSON (specialists were already scored on both tasks).
2. Exactly-zero update of the unused label column of a1 (and c1) during specialization.
3. Seed-bank disjointness (training / DAgger / CPG reset seeds vs VAL_SEEDS and EVAL_SEEDS).
4. Per-seed steps to 90% on both tasks for each adaptation run (no medians between grid points).
5. (fwd_bwd only) Demo reproduction: the saved demo merges reload, the TIES rebuild matches the
   saved npz, and one episode on eval seed 10000 matches media/demo_grid_returns.json.
"""
import json
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

from common import EVAL_SEEDS, TASKS, VAL_SEEDS, evaluate, load_params  # noqa: E402
import merge as M  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAIR = "_".join(TASKS)
CK = os.path.join(HERE, "checkpoints") if PAIR == "fwd_bwd" else os.path.join(HERE, "checkpoints", PAIR)
RES = os.path.join(HERE, "results") if PAIR == "fwd_bwd" else os.path.join(HERE, "results", PAIR)
A, B = TASKS
SEEDS = [0, 1, 2, 3]


def m(x):
    return float(np.mean(x))


def main():
    out = {"pair": PAIR}
    merge = {s: json.load(open(os.path.join(RES, f"merge_s{s}.json"))) for s in SEEDS}
    S = str(max(int(k) for k in merge[0]["pre"]["by_length"]))

    # 1. single specialist under both labels, picked on validation seeds
    single = {}
    for cond in ("pre", "init"):
        rows = []
        for s in SEEDS:
            d = os.path.join(CK, f"s{s}")
            _, norm = load_params(os.path.join(d, "base.npz"))
            b = merge[s][cond]["by_length"][S]
            sv = b["spec_val"]                       # own-task validation return of each specialist
            val, ev = {}, {}
            for who in TASKS:
                p, _ = load_params(os.path.join(d, f"{cond}_{who}.npz"))
                other = B if who == A else A
                val[who] = {who: sv[who], other: m(evaluate(p, norm, other, VAL_SEEDS))}
                ev[who] = {tk: m(b["specialist_" + who][tk]) for tk in TASKS}
            oracle_ev = {tk: ev[tk][tk] for tk in TASKS}
            ret_ev = {who: float(np.mean([ev[who][tk] / oracle_ev[tk] for tk in TASKS])) for who in TASKS}
            ret_val = {who: float(np.mean([val[who][tk] / sv[tk] for tk in TASKS])) for who in TASKS}
            sum_val = {who: float(sum(val[who].values())) for who in TASKS}
            pick_sum = max(TASKS, key=lambda w: sum_val[w])
            pick_ret = max(TASKS, key=lambda w: ret_val[w])
            th0 = {tk: m(merge[s][cond]["theta0"][tk]) for tk in TASKS}
            gain = {who: float(np.mean([(ev[who][tk] - th0[tk]) / (oracle_ev[tk] - th0[tk]) for tk in TASKS]))
                    for who in TASKS}
            rows.append(dict(seed=s, val=val, sum_val=sum_val, ret_val=ret_val, eval=ev,
                             eval_mean_retained=ret_ev, eval_gain_retained=gain,
                             pick_by_summed_val_return=pick_sum, pick_by_val_mean_retained=pick_ret,
                             picked_eval_mean_retained_sum=ret_ev[pick_sum],
                             picked_eval_mean_retained_ret=ret_ev[pick_ret]))
        agg = {}
        for crit in ("sum", "ret"):
            v = np.array([r[f"picked_eval_mean_retained_{crit}"] for r in rows])
            g = np.array([r["eval_gain_retained"][r["pick_by_summed_val_return" if crit == "sum"
                                                     else "pick_by_val_mean_retained"]] for r in rows])
            agg[crit] = dict(mean_retained=float(v.mean()), std=float(v.std()), per_seed=v.tolist(),
                             gain_retained=float(g.mean()), gain_std=float(g.std()),
                             picks=[r["pick_by_summed_val_return" if crit == "sum" else "pick_by_val_mean_retained"]
                                    for r in rows])
        single[cond] = dict(rows=rows, summary=agg)
        print(f"[single-spec {PAIR} {cond}] summed-val-return pick {agg['sum']['picks']} -> "
              f"{100 * agg['sum']['mean_retained']:.0f}% ± {100 * agg['sum']['std']:.0f} "
              f"(gain {100 * agg['sum']['gain_retained']:.0f}%);  val-mean-retained pick {agg['ret']['picks']} -> "
              f"{100 * agg['ret']['mean_retained']:.0f}% ± {100 * agg['ret']['std']:.0f} "
              f"(gain {100 * agg['ret']['gain_retained']:.0f}%)", flush=True)
    out["single_specialist"] = single

    # 2. unused label column: exactly zero update?
    zg = []
    for s in SEEDS:
        d = os.path.join(CK, f"s{s}")
        for cond, t0name in (("pre", "base.npz"), ("init", "init.npz")):
            t0, _ = load_params(os.path.join(d, t0name))
            for who in TASKS:
                p, _ = load_params(os.path.join(d, f"{cond}_{who}.npz"))
                own = 17 + TASKS.index(who)
                unused = 17 + (1 - TASKS.index(who))
                zg.append(dict(seed=s, cond=cond, spec=who,
                               a1_unused_maxabs=float(np.abs(p["a1.weight"][:, unused] - t0["a1.weight"][:, unused]).max()),
                               c1_unused_maxabs=float(np.abs(p["c1.weight"][:, unused] - t0["c1.weight"][:, unused]).max()),
                               a1_own_maxabs=float(np.abs(p["a1.weight"][:, own] - t0["a1.weight"][:, own]).max())))
    out["zero_grad_label_column"] = dict(rows=zg, all_exactly_zero=all(r["a1_unused_maxabs"] == 0 and
                                                                       r["c1_unused_maxabs"] == 0 for r in zg))
    print(f"[zero-grad] unused label column exactly unchanged in all specialists: "
          f"{out['zero_grad_label_column']['all_exactly_zero']}", flush=True)

    # 3. seed banks (first reset seed of every env; later resets continue each env's own RNG)
    train_seeds = set()
    for s in SEEDS:
        for base in (s, s + 100, s + 300):             # base, specialists, fine-tunes / transfer / scratch
            train_seeds |= {base * 1000 + i for i in range(16)}
        train_seeds |= {(s + 400) * 1000 + 50 + i for i in range(16)}   # DAgger
    train_seeds |= set(range(500, 1500)) | {900, 901, 902}           # CPG CEM + re-score
    hold = set(VAL_SEEDS) | set(EVAL_SEEDS)
    out["seed_overlap_with_val_or_eval"] = sorted(train_seeds & hold)
    out["val_eval_overlap"] = sorted(set(VAL_SEEDS) & set(EVAL_SEEDS))
    print(f"[seeds] training/val/eval overlap: {out['seed_overlap_with_val_or_eval']} {out['val_eval_overlap']}",
          flush=True)

    # 4. per-seed steps to 90% on both tasks
    adapt = {s: json.load(open(os.path.join(RES, f"adapt_s{s}.json"))) for s in SEEDS
             if os.path.exists(os.path.join(RES, f"adapt_s{s}.json"))}
    hits = {}
    for key in adapt.get(0, {}):
        if key in ("seed", "wall_s_last") or "transfer" in key:
            continue
        cond = "init" if key.startswith("init") else "pre"
        hs = []
        for s in adapt:
            cur = adapt[s][key].get("curve") or adapt[s][key]["evals"]
            b = merge[s][cond]["by_length"][S]
            ref = {tk: m(b["specialist_" + tk][tk]) for tk in TASKS}
            h = [c["step"] for c in cur if all(c[tk] >= 0.9 * ref[tk] for tk in TASKS)]
            hs.append(h[0] if h else None)
        hits[key] = hs
        print(f"[90% both] {key}: {hs}", flush=True)
    out["steps_to_90_both_per_seed"] = hits

    # 5. demo reproduction (fwd_bwd only)
    if PAIR == "fwd_bwd":
        d = os.path.join(CK, "s0")
        t0, norm = load_params(os.path.join(d, "base.npz"))
        Af, _ = load_params(os.path.join(d, "pre_fwd.npz"))
        Bb, _ = load_params(os.path.join(d, "pre_bwd.npz"))
        saved_best, _ = load_params(os.path.join(CK, "demo", "s0_pre_best_merge.npz"))
        saved_avg, _ = load_params(os.path.join(CK, "demo", "s0_pre_average.npz"))
        rebuilt = M.ties(t0, [Af, Bb], 0.3, k=0.2)
        same = all(np.allclose(rebuilt[k], saved_best[k]) for k in rebuilt)
        ref = json.load(open(os.path.join(HERE, "media", "demo_grid_returns.json")))
        rep = {}
        for name, p in (("average a=0.5", saved_avg), ("best merge: ties l=0.3", saved_best)):
            for tk in TASKS:
                r = m(evaluate(p, norm, tk, [10000]))
                rep[f"{name} | {tk}"] = dict(now=r, json=ref[f"{name} | {tk}"])
        out["demo"] = dict(ties_rebuild_matches_saved=same, returns=rep)
        print(f"[demo] TIES rebuild == saved npz: {same}; " +
              "; ".join(f"{k}: {v['now']:.0f} vs {v['json']:.0f}" for k, v in rep.items()), flush=True)

    path = os.path.join(RES, "review_checks.json" if PAIR == "fwd_bwd" else f"review_checks_{PAIR}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
