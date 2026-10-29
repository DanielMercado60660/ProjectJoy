"""Aggregate results/*.json over training seeds -> results/summary.md, results/summary.json, media/*.png.

  ../.venv/bin/python report.py
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("REPORT_RES", os.path.join(HERE, "results"))
MEDIA = os.environ.get("REPORT_MEDIA", os.path.join(HERE, "media"))
TASKS = os.environ.get("SKILL_PAIR", "fwd_bwd").split("_")
A, B = TASKS
TAG = "" if TASKS == ["fwd", "bwd"] else "_" + "_".join(TASKS)   # file-name suffix for other pairs
COND_NAME = {"pre": "shared pretraining (theta0 = 0.5M-step multitask base)",
             "init": "shared random init only (theta0 = init)"}
# reference categorical palette (dataviz skill), fixed order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def load(prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, f"{prefix}_s*.json"))):
        d = json.load(open(f))
        out[d["seed"]] = d
    return out


def m(x):
    return float(np.mean(x))


def ms(xs, fmt="{:.0f}", pct=False):
    xs = np.asarray(xs, float)
    if pct:
        return f"{100 * xs.mean():.0f}% ± {100 * xs.std():.0f}"
    return (fmt + " ± " + fmt).format(xs.mean(), xs.std())


def main():
    train, merge, adapt = load("train"), load("merge"), load("adapt")
    seeds = sorted(merge)
    S = max(int(k) for k in merge[seeds[0]]["pre"]["by_length"])
    lines, summary = [], {"seeds": seeds, "S": S}
    P = lines.append
    P(f"# Auto-generated summary (seeds {seeds}, specialist branch = {S} env steps)\n")
    P("Returns are deterministic (mean-action) 1000-step episode returns on 10 held-out eval seeds, "
      "averaged per training seed; tables show mean ± std over training seeds. "
      "'retained' = method return / oracle-switch (specialist) return on that task, computed per seed. "
      "'gain retained' = (method - theta0) / (specialist - theta0), averaged over the two tasks: the share of "
      "what each branch learned AFTER the shared start that survives (in `init`, theta0 returns ~0 so it equals retained).\n")

    cpg = json.load(open(os.path.join(RES, "cpg.json"))) if os.path.exists(os.path.join(RES, "cpg.json")) else None

    def spec_ref(seed, cond, L=S):
        b = merge[seed][cond]["by_length"][str(L)]
        return {tk: m(b["specialist_" + tk][tk]) for tk in TASKS}

    # ---------------------------------------------------------------- main table
    for cond in ("pre", "init"):
        P(f"\n## Condition `{cond}`: {COND_NAME[cond]}\n")
        P(f"| method | {A} return | {B} return | {A} retained | {B} retained | mean retained | gain retained | lambda |")
        P("|---|---|---|---|---|---|---|---|")
        fine = load("lambda_fine")
        rows = {}
        for meth in ("oracle_switch", "theta0", "average", "task_arith", "ties", "dare",
                     f"zero_shot_{A}_spec", f"zero_shot_{B}_spec", "distill_theta0", "distill_average",
                     "ft_theta0", "ft_average", "ft_task_arith",
                     "task_arith_fine", "ties_fine", "dare_fine"):
            R = {tk: [] for tk in TASKS}
            lam = []
            ok = True
            for s in seeds:
                b = merge[s][cond]["by_length"][str(S)]
                if meth == "oracle_switch":
                    v = {tk: m(b["specialist_" + tk][tk]) for tk in TASKS}
                elif meth == "theta0":
                    v = {tk: m(merge[s][cond]["theta0"][tk]) for tk in TASKS}
                elif meth.startswith("zero_shot"):
                    who = meth.split("_")[2]
                    v = {tk: m(b["specialist_" + who][tk]) for tk in TASKS}
                elif meth.endswith("_fine"):
                    if s not in fine:
                        ok = False
                        break
                    fb = fine[s][cond][meth[:-5]]
                    v = {tk: m(fb[tk]) for tk in TASKS}
                    lam.append(fb["lam"])
                elif meth.startswith("distill") or meth.startswith("ft_"):
                    key = f"{cond}_{meth}" if meth.startswith("distill") else f"{cond}_ft_{meth[3:]}"
                    if s not in adapt or key not in adapt[s]:
                        ok = False
                        break
                    e = adapt[s][key]
                    last = (e["curve"] if "curve" in e else e["evals"])[-1]
                    v = {tk: last[tk] for tk in TASKS}
                else:
                    v = {tk: m(b[meth][tk]) for tk in TASKS}
                    if "lam" in b[meth]:
                        lam.append(b[meth]["lam"])
                for tk in TASKS:
                    R[tk].append(v[tk])
            if not ok:
                continue
            ref = [spec_ref(s, cond) for s in seeds]
            fr = {tk: np.array(R[tk]) / np.array([r[tk] for r in ref]) for tk in TASKS}
            th0 = [{tk: m(merge[s][cond]["theta0"][tk]) for tk in TASKS} for s in seeds]
            gain = np.mean([(np.array(R[tk]) - np.array([t[tk] for t in th0])) /
                            (np.array([r[tk] for r in ref]) - np.array([t[tk] for t in th0])) for tk in TASKS], 0)
            rows[meth] = dict(ret={tk: R[tk] for tk in TASKS}, frac={tk: fr[tk].tolist() for tk in TASKS}, lam=lam,
                              gain=gain.tolist())
            P(f"| {meth} | {ms(R[A])} | {ms(R[B])} | {ms(fr[A], pct=True)} | {ms(fr[B], pct=True)} "
              f"| {ms((fr[A] + fr[B]) / 2, pct=True)} | {ms(gain, pct=True)} | {', '.join(str(l) for l in lam) if lam else ''} |")
        if cpg and TAG == "":
            ret = {tk: m(cpg[tk]["eval"][tk]) for tk in TASKS}
            P(f"| scripted CPG gait (per-task, label-switched) | {ret['fwd']:.0f} | {ret['bwd']:.0f} | "
              f"{100 * ret['fwd'] / np.mean([spec_ref(s, cond)['fwd'] for s in seeds]):.0f}% | "
              f"{100 * ret['bwd'] / np.mean([spec_ref(s, cond)['bwd'] for s in seeds]):.0f}% | | |")
        summary[cond] = rows

        # interference
        P("\n**Interference (task arithmetic, same lambda):** return on each task with only its own task "
          "vector added vs with both added.\n")
        P(f"| task | theta0 + lam*tau_own | theta0 + lam*(tau_{A}+tau_{B}) | drop from adding the other vector |")
        P("|---|---|---|---|")
        for tk in TASKS:
            own = [m(merge[s][cond]["by_length"][str(S)]["ta_single"][f"{tk}_only"][tk]) for s in seeds]
            both = [m(merge[s][cond]["by_length"][str(S)]["task_arith"][tk]) for s in seeds]
            drop = [o - b for o, b in zip(own, both)]
            P(f"| {tk} | {ms(own)} | {ms(both)} | {ms(drop)} |")
            summary.setdefault("interference", {}).setdefault(cond, {})[tk] = dict(own=own, both=both)

        # branch-length study
        P("\n**Branch length** (env steps each specialist trained after branching) vs merge quality, "
          "mean retained over both tasks:\n")
        P(f"| branch steps | specialist {A} / {B} return | average | task arith | TIES | DARE | TIES gain retained | cos(tau_{A}, tau_{B}) | |tau| / |theta0| |")
        P("|---|---|---|---|---|---|---|---|---|")
        Ls = sorted(int(k) for k in merge[seeds[0]][cond]["by_length"])
        for L in Ls:
            cells = []
            for meth in ("average", "task_arith", "ties", "dare"):
                v = []
                for s in seeds:
                    b = merge[s][cond]["by_length"][str(L)]
                    ref = {tk: m(b["specialist_" + tk][tk]) for tk in TASKS}
                    v.append(np.mean([m(b[meth][tk]) / ref[tk] for tk in TASKS]))
                cells.append(ms(v, pct=True))
            gv = []
            for s in seeds:
                b = merge[s][cond]["by_length"][str(L)]
                g = [(m(b["ties"][tk]) - m(merge[s][cond]["theta0"][tk])) /
                     (m(b["specialist_" + tk][tk]) - m(merge[s][cond]["theta0"][tk])) for tk in TASKS]
                gv.append(np.mean(g))
            cells.append(ms(gv, pct=True))
            st = [merge[s][cond]["by_length"][str(L)]["stats"] for s in seeds]
            sp = [spec_ref(s, cond, L) for s in seeds]
            P(f"| {L} | {ms([x[A] for x in sp])} / {ms([x[B] for x in sp])} | " + " | ".join(cells) +
              f" | {ms([x['cos_tau'] for x in st], '{:.2f}')} | "
              f"{ms([(x['norm_tau_fwd'] + x['norm_tau_bwd']) / 2 / x['norm_theta0'] for x in st], '{:.2f}')} |")

    # ---------------------------------------------------------------- adaptation cost
    if adapt:
        P(f"\n## Adaptation cost: env steps of multitask PPO (8 {A} + 8 {B} envs) or DAgger distillation until "
          "BOTH tasks reach a threshold fraction of the specialists (first eval at or above; '>' = not reached)\n")
        P(f"| run | 50% | 80% | 90% | final {A} / {B} retained |")
        P("|---|---|---|---|---|")
        keys = [k for k in adapt[seeds[0]] if k not in ("seed", "wall_s_last")]
        for key in keys:
            cond = "pre" if key.startswith("pre") or key == "multitask_scratch" else "init"
            res_th = {th: [] for th in (0.5, 0.8, 0.9)}
            finals = []
            for s in seeds:
                if s not in adapt or key not in adapt[s]:
                    continue
                e = adapt[s][key]
                cur = e["curve"] if "curve" in e else e["evals"]
                ref = spec_ref(s, cond)
                tasks = TASKS
                if "transfer" in key:
                    tasks = [B] if key.endswith("2" + B) else [A]
                for th in res_th:
                    hit = [c["step"] for c in cur if all(c[tk] >= th * ref[tk] for tk in tasks)]
                    res_th[th].append(hit[0] if hit else np.inf)
                finals.append([cur[-1][tk] / ref[tk] for tk in TASKS])
            if not finals:
                continue

            def fmt(v):
                v = np.array(v)
                n_inf = int(np.isinf(v).sum())
                fin = v[np.isfinite(v)]
                if len(fin) == 0:
                    return f"> budget (0/{len(v)})"
                s_ = f"{np.median(v) / 1e3:.0f}k median" if n_inf < len(v) / 2 else f"> budget (median)"
                return s_ + f" ({len(fin)}/{len(v)} reached)"

            fa = np.array(finals)
            P(f"| {key} | {fmt(res_th[0.5])} | {fmt(res_th[0.8])} | {fmt(res_th[0.9])} | "
              f"{ms(fa[:, 0], pct=True)} / {ms(fa[:, 1], pct=True)} |")
            summary.setdefault("adapt", {})[key] = {str(k): v for k, v in res_th.items()}

    # sequential transfer vs training the same task from theta0 (same PPO settings, same steps)
    tkeys = [(f"pre_transfer_{A}2{B}", f"pre_{B}", B), (f"pre_transfer_{B}2{A}", f"pre_{A}", A)]
    if adapt and all(k in adapt[s] for s in seeds for k, _, _ in tkeys):
        P("\n## Sequential transfer (`pre`): PPO on a task starting from the OTHER specialist vs from theta0\n")
        P("| target task | start | return at 0 | at ~0.25M | at ~0.5M | at ~0.88M |")
        P("|---|---|---|---|---|---|")
        for key, spec, tk in tkeys:
            for lab, get in (("other task's specialist", lambda s: adapt[s][key]["evals"]),
                             ("theta0 (= the specialist's own training curve)", lambda s: train[s]["spec"][spec]["evals"])):
                cur = [{e["step"]: e[tk] for e in get(s)} for s in seeds]
                steps = sorted(set.intersection(*[set(c) for c in cur]))
                pick = [steps[0]] + [min(steps, key=lambda x: abs(x - t)) for t in (250_000, 500_000, 880_000)]
                P(f"| {tk} | {lab} | " + " | ".join(ms([c[p] for c in cur]) for p in pick) + " |")

    open(os.path.join(RES, f"summary{TAG}.md"), "w").write("\n".join(lines) + "\n")
    json.dump(summary, open(os.path.join(RES, f"summary{TAG}.json"), "w"), indent=1, default=float)
    print("\n".join(lines))
    plots(train, merge, adapt, seeds, S, spec_ref, cpg)


def band(ax, x, Y, color, label, ls="-"):
    Y = np.array(Y)
    mu, sd = Y.mean(0), Y.std(0)
    ax.plot(x, mu, color=color, label=label, ls=ls)
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)


def plots(train, merge, adapt, seeds, S, spec_ref, cpg):
    os.makedirs(MEDIA, exist_ok=True)
    # interpolation + task-arithmetic curves
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for row, cond in enumerate(("pre", "init")):
        ax = axes[row, 0]
        al = sorted(float(a) for a in merge[seeds[0]][cond]["by_length"][str(S)]["interp"])
        for j, tk in enumerate(TASKS):
            Y = [[m(merge[s][cond]["by_length"][str(S)]["interp"][str(a)][tk]) for a in al] for s in seeds]
            band(ax, al, Y, C[j], f"return on {tk}")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"[{cond}] (1-α)·θ_{A} + α·θ_{B}", fontsize=10)
        ax.set_xlabel(f"α (0 = {A} specialist, 1 = {B} specialist)")
        ax.set_ylabel("episode return")
        ax.legend(frameon=False, fontsize=8)
        ax = axes[row, 1]
        lams = sorted(float(l) for l in merge[seeds[0]][cond]["by_length"][str(S)]["ta_curve"])
        for j, tk in enumerate(TASKS):
            Y = [[m(merge[s][cond]["by_length"][str(S)]["ta_curve"][str(l)][tk]) for l in lams] for s in seeds]
            band(ax, lams, Y, C[j], f"return on {tk}")
            ref = np.mean([spec_ref(s, cond)[tk] for s in seeds])
            ax.axhline(ref, color=C[j], lw=1, ls=":")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"[{cond}] θ0 + λ(τ_{A} + τ_{B})   (dotted = oracle switch)", fontsize=10)
        ax.set_xlabel("λ")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(f"Merging two HalfCheetah specialists ({A}/{B}), mean ± std over {len(seeds)} training seeds",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, f"interpolation{TAG}.png"), dpi=130)
    plt.close(fig)

    # branch length (absolute returns: ratios blow up when a short branch's specialist is weak)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, cond in zip(axes, ("pre", "init")):
        Ls = sorted(int(k) for k in merge[seeds[0]][cond]["by_length"])
        X = [L / 1e6 for L in Ls]
        series = [("oracle switch (specialists)", lambda b: {tk: m(b["specialist_" + tk][tk]) for tk in TASKS}),
                  ("average", lambda b: {tk: m(b["average"][tk]) for tk in TASKS}),
                  ("task arith", lambda b: {tk: m(b["task_arith"][tk]) for tk in TASKS}),
                  ("TIES", lambda b: {tk: m(b["ties"][tk]) for tk in TASKS}),
                  ("DARE", lambda b: {tk: m(b["dare"][tk]) for tk in TASKS})]
        for j, (lab, f) in enumerate(series):
            Y = [[np.mean(list(f(merge[s][cond]["by_length"][str(L)]).values())) for L in Ls] for s in seeds]
            band(ax, X, Y, C[j], lab, ls=":" if j == 0 else "-")
        th = np.mean([np.mean([m(merge[s][cond]["theta0"][tk]) for tk in TASKS]) for s in seeds])
        ax.axhline(th, color=INK2, lw=1.2, ls="--", label="θ0 (shared start)")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(X)
        ax.set_xticklabels([f"{x:g}" for x in X])
        ax.minorticks_off()
        ax.set_title(f"[{cond}] merge quality vs specialist branch length", fontsize=10)
        ax.set_xlabel("specialist branch length (M env steps)")
    axes[0].set_ylabel(f"mean return over {A} and {B} tasks")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, f"branch_length{TAG}.png"), dpi=130)
    plt.close(fig)

    # adaptation curves: min over tasks of retained fraction
    if adapt:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, cond in zip(axes, ("pre", "init")):
            j = 0
            for key, lab in ((f"{cond}_ft_theta0", "PPO from θ0"), (f"{cond}_ft_average", "PPO from average"),
                             (f"{cond}_ft_task_arith", "PPO from task-arith"),
                             (f"{cond}_distill_theta0", "DAgger from θ0"),
                             (f"{cond}_distill_average", "DAgger from average"),
                             ("multitask_scratch", "multitask PPO from random init")):
                ss = [s for s in seeds if s in adapt and key in adapt[s]]
                if not ss:
                    j += 1
                    continue
                Y, X = [], None
                for s in ss:
                    e = adapt[s][key]
                    cur = e["curve"] if "curve" in e else e["evals"]
                    ref = spec_ref(s, cond)
                    X = [c["step"] / 1e6 for c in cur]
                    Y.append([min(c[tk] / ref[tk] for tk in TASKS) for c in cur])
                n = min(len(y) for y in Y)
                band(ax, X[:n], [y[:n] for y in Y], C[j], lab, ls="--" if "DAgger" in lab else "-")
                j += 1
            ax.axhline(0.9, color=INK2, lw=1, ls=":")
            ax.set_ylim(-0.6, 1.15)
            ax.set_title(f"[{cond}] worst-task retained vs adaptation env steps", fontsize=10)
            ax.set_xlabel("env steps (M)")
        axes[0].set_ylabel("min over tasks of return / specialist")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(MEDIA, f"adaptation{TAG}.png"), dpi=130)
        plt.close(fig)

    # training curves of the specialists
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, cond in zip(axes, ("pre", "init")):
        for j, tk in enumerate(TASKS):
            Y, X = [], None
            for s in seeds:
                ev = train[s]["spec"][f"{cond}_{tk}"]["evals"]
                X = [e["step"] / 1e6 for e in ev]
                Y.append([e[tk] for e in ev])
            band(ax, X, Y, C[j], f"{tk} specialist on {tk}")
            Y = [[e[TASKS[1 - j]] for e in train[s]["spec"][f"{cond}_{tk}"]["evals"]] for s in seeds]
            band(ax, X, Y, C[j], f"{tk} specialist on {TASKS[1 - j]}", ls="--")
        if adapt and cond == "pre":
            for j, (key, tk, lab) in enumerate(((f"pre_transfer_{A}2{B}", B, f"{A} spec → PPO on {B}"),
                                                (f"pre_transfer_{B}2{A}", A, f"{B} spec → PPO on {A}"))):
                ss = [s for s in seeds if s in adapt and key in adapt[s]]
                if ss:
                    Y = [[e[tk] for e in adapt[s][key]["evals"]] for s in ss]
                    X = [e["step"] / 1e6 for e in adapt[ss[0]][key]["evals"]]
                    band(ax, X, Y, C[2 + j], lab, ls="-.")
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_title(f"[{cond}] specialist training (eval return)", fontsize=10)
        ax.set_xlabel("env steps after branching (M)")
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("episode return")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, f"training_curves{TAG}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
