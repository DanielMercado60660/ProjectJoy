"""Evaluation driver. Every match result is cached in results/matches.json, so runs
can be interrupted and resumed.

  python evaluate.py suite  --run main --gens 0,5,10,...   # each gen vs the fixed suite
  python evaluate.py rr     --run main --gens 0,10,20,...  # generation round robin
  python evaluate.py budget --run main --gen 80            # matched decision-time budgets
  python evaluate.py elo                                   # fit Elo over every cached match
  python evaluate.py plots                                 # figures in media/
"""
import argparse
import json
import os
import time

import numpy as np

import arena

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# each evaluate process writes its own cache file (EVAL_TAG), all files are merged on read,
# so two evaluation jobs can run at the same time without clobbering each other
CACHE = os.path.join(RES, f"matches_{os.environ.get('EVAL_TAG', 'main')}.json")
SUITE = ["random", "greedy", "uct200", "uct1000", "uct5000", "ab2", "ab4", "ab6", "ab8"]
EVAL_SIMS = 100
# training phases of the main run (see RESULTS.md): (first gen, last gen, label)
PHASES = [(1, 100, "100 sims, lr 2e-3"), (101, 136, "200 sims, lr 1e-3"),
          (137, 999, "200 sims + random openings")]


def shade_phases(ax, gmax):
    cols = ["#00000000", "#8888ff18", "#88cc8820"]
    for (g0, g1, lab), c in zip(PHASES, cols):
        if g0 > gmax:
            break
        ax.axvspan(g0 - 0.5, min(g1, gmax) + 0.5, color=c, lw=0)
        ax.text(g0, 1.0, " " + lab, transform=ax.get_xaxis_transform(), fontsize=7, va="top", color="#555")


def load_cache():
    out = {}
    if os.path.isdir(RES):
        for fn in sorted(os.listdir(RES)):
            if fn.startswith("matches") and fn.endswith(".json"):
                with open(os.path.join(RES, fn)) as f:
                    out.update(json.load(f))
    return out


def save_cache(c):
    os.makedirs(RES, exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(c, f, indent=0)
    os.replace(tmp, CACHE)


def net_spec(run, gen, sims=EVAL_SIMS):
    return f"net:checkpoints/{run}/gen_{gen:03d}:{sims}"


def short(spec):
    if spec.startswith("net:"):
        _, p, s = spec.split(":")
        return f"{p.split('/')[-2]}/g{int(p.split('_')[-1])}@{s}"
    return spec


DEADLINE = [None]


class OutOfTime(Exception):
    pass


def run_match(cache, a, b, seed=0):
    key = f"{a} | {b}"
    if key in cache:
        return cache[key]
    if DEADLINE[0] is not None and time.time() > DEADLINE[0]:
        raise OutOfTime()
    t = time.time()
    A = arena.make_agent(a, seed=seed)
    B = arena.make_agent(b, seed=seed + 1)
    r = arena.play_match(A, B, seed=seed)
    r["ms_per_move_a"] = 1000 * A.think_time / max(1, A.n_moves)
    r["ms_per_move_b"] = 1000 * B.think_time / max(1, B.n_moves)
    r["seconds"] = time.time() - t
    cache[key] = r
    save_cache(cache)
    print(f"{short(a):>22} vs {short(b):<22} score {r['score']:.3f}  "
          f"(W{r['wins']} D{r['draws']} L{r['losses']}; first {r['score_as_first']:.2f} second {r['score_as_second']:.2f})"
          f"  {r['ms_per_move_a']:.1f} ms/move  [{r['seconds']:.0f}s]", flush=True)
    return r


def parse_gens(s, run):
    if s.startswith("every"):          # e.g. every5 -> 0,5,10,... plus the latest generation
        k = int(s[5:])
        allg = parse_gens("all", run)
        return sorted(set([g for g in allg if g % k == 0] + allg[-1:]), reverse=True)
    if s == "all":
        d = os.path.join(HERE, "checkpoints", run)
        return sorted(int(f[4:7]) for f in os.listdir(d) if f.startswith("gen_") and f.endswith(".json"))
    return [int(x) for x in s.split(",") if x]


def cmd_suite(a):
    cache = load_cache()
    for g in parse_gens(a.gens, a.run):
        for opp in SUITE:
            run_match(cache, net_spec(a.run, g, a.sims), opp)


def cmd_rr(a):
    cache = load_cache()
    gens = parse_gens(a.gens, a.run)
    for i, g1 in enumerate(gens):
        for g2 in gens[i + 1:]:
            run_match(cache, net_spec(a.run, g2, a.sims), net_spec(a.run, g1, a.sims))


def cmd_budget(a):
    """Policy-only vs search without a trained net vs the combined system."""
    cache = load_cache()
    budgets = [int(x) for x in a.budgets.split(",")]
    opps = a.opps.split(",")
    net0 = net_spec(a.run, a.gen, 0)
    for opp in opps:
        run_match(cache, net0, opp)
    for n in budgets:
        for opp in opps:
            run_match(cache, net_spec(a.run, a.gen, n), opp)
            run_match(cache, f"uct{n}", opp)
            run_match(cache, f"mcts0:{n}", opp)
        # head to head at equal simulation count
        run_match(cache, net_spec(a.run, a.gen, n), f"uct{n}")
    # the raw policy against rollout MCTS at every budget
    for n in budgets:
        run_match(cache, net0, f"uct{n}")
    # wall-time matched: rollout MCTS gets as many simulations as fit in the net's single-game time
    lp = os.path.join(RES, "latency.json")
    if os.path.exists(lp):
        with open(lp) as f:
            matched = json.load(f)["time_matched_uct"]
        for n, u in matched.items():
            for opp in opps:
                run_match(cache, f"uct{u}", opp)
            run_match(cache, net_spec(a.run, a.gen, int(n)), f"uct{u}")


def single_game_latency(spec, positions, reps=1):
    """Median ms per move for ONE game at a time (no batching across games): the honest
    decision-time cost of an agent."""
    ag = arena.make_agent(spec, max_games=1)
    ag.moves(np.array(positions[:1, 0]), np.array(positions[:1, 1]))   # warm-up / compile
    ts = []
    for p, m in positions:
        for _ in range(reps):
            t = time.perf_counter()
            ag.moves(np.array([p]), np.array([m]))
            ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def latency_positions(n=20, seed=0):
    import c4
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        p, m = np.int64(0), np.int64(0)
        ok = True
        for _ in range(6):
            c = int(rng.integers(7))
            if c4.winning_move(p, m, c):
                ok = False
                break
            p, m = c4.play(p, m, c)
        if ok:
            out.append((p, m))
    return np.array(out, np.int64)


def cmd_latency(a):
    """Single-game ms/move for net@n, uct@n, mcts0@n; then the rollout budget that matches
    the net's wall time at each n."""
    budgets = [int(x) for x in a.budgets.split(",")]
    pos = latency_positions()
    lat = {}
    lat[net_spec(a.run, a.gen, 0)] = single_game_latency(net_spec(a.run, a.gen, 0), pos)
    for n in budgets:
        for spec in (net_spec(a.run, a.gen, n), f"uct{n}", f"mcts0:{n}"):
            lat[spec] = single_game_latency(spec, pos)
            print(f"{short(spec):>24}: {lat[spec]:8.2f} ms/move (single game)", flush=True)
    per_sim_uct = np.median([lat[f"uct{n}"] / n for n in budgets if n >= 256] or [lat[f"uct{budgets[-1]}"] / budgets[-1]])
    # only the small budgets: at 256+ sims the time-matched rollout budget runs into the millions
    matched = {n: min(200_000, int(round(lat[net_spec(a.run, a.gen, n)] / per_sim_uct / 10) * 10))
               for n in budgets if n <= 64}
    for n, u in matched.items():
        lat[f"uct{u}"] = single_game_latency(f"uct{u}", pos)
        print(f"net@{n} ~ {lat[net_spec(a.run, a.gen, n)]:.1f} ms  ->  uct{u}: {lat[f'uct{u}']:.1f} ms", flush=True)
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "latency.json"), "w") as f:
        json.dump(dict(gen=a.gen, ms_per_move=lat, time_matched_uct=matched), f, indent=1)


def cmd_baselines(a):
    """Round robin among the fixed opponents themselves (anchors the Elo scale)."""
    cache = load_cache()
    for i, x in enumerate(SUITE):
        for y in SUITE[i + 1:]:
            run_match(cache, y, x)


def cmd_elo(a):
    cache = load_cache()
    pairs = [(r["a"], r["b"], r["score"], r["games"]) for r in cache.values()]
    elo = arena.fit_elo(pairs, anchor="random")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "elo.json"), "w") as f:
        json.dump(elo, f, indent=1)
    for k, v in sorted(elo.items(), key=lambda kv: -kv[1]):
        print(f"{v:8.0f}  {short(k)}")


def cmd_report(a):
    """Markdown tables from the cache -> results/report.md"""
    cache = load_cache()
    elo = {}
    ep = os.path.join(RES, "elo.json")
    if os.path.exists(ep):
        elo = json.load(open(ep))
    get = lambda x, y: cache.get(f"{x} | {y}")
    out = []
    # baseline round robin (row player's score)
    out.append("### Baseline round robin (row score vs column, 98 games each)\n")
    out.append("| | " + " | ".join(SUITE) + " | Elo |")
    out.append("|---" * (len(SUITE) + 2) + "|")
    for x in SUITE:
        cells = []
        for y in SUITE:
            r = get(x, y)
            if r:
                cells.append(f"{r['score']:.2f}")
            elif get(y, x):
                cells.append(f"{1 - get(y, x)['score']:.2f}")
            else:
                cells.append("–" if x == y else "n/m")
        out.append(f"| **{x}** | " + " | ".join(cells) + f" | {elo.get(x, float('nan')):.0f} |")
    # generations vs suite
    gens = sorted({int(r["a"].split("gen_")[1][:3]) for r in cache.values()
                   if r["a"].startswith(f"net:checkpoints/{a.run}/") and r["a"].endswith(f":{EVAL_SIMS}") and r["b"] in SUITE})
    out.append(f"\n### Generations vs the suite (net + {EVAL_SIMS} sims; score, 98 games each)\n")
    out.append("| gen | " + " | ".join(SUITE) + " | Elo |")
    out.append("|---" * (len(SUITE) + 2) + "|")
    for g in gens:
        spec = net_spec(a.run, g)
        cells = [f"{get(spec, o)['score']:.2f}" if get(spec, o) else "n/m" for o in SUITE]
        out.append(f"| {g} | " + " | ".join(cells) + f" | {elo.get(spec, float('nan')):.0f} |")
    # budget table
    rows = []
    for r in cache.values():
        A = r["a"]
        if r["b"] not in ("greedy", "ab4", "ab8", "uct1000", "uct5000"):
            continue
        if A.startswith("net:") and not A.endswith(f":{EVAL_SIMS}") or A.startswith("mcts0:") or (A.startswith("uct") and A not in SUITE):
            rows.append(r)
    if rows:
        out.append("\n### Matched-budget comparison (score vs opponent, 98 games each; ms/move is batch-amortised)\n")
        out.append("| agent | opponent | score | W/D/L | ms/move |")
        out.append("|---|---|---|---|---|")
        for r in sorted(rows, key=lambda r: (r["b"], r["a"])):
            out.append(f"| {short(r['a'])} | {r['b']} | {r['score']:.2f} | {r['wins']}/{r['draws']}/{r['losses']} | {r['ms_per_move_a']:.1f} |")
    h2h = [r for r in cache.values() if r["a"].startswith("net:") and r["b"].startswith("uct") and r["b"] not in SUITE]
    if h2h:
        out.append("\n### Head to head: net + MCTS vs rollout MCTS\n")
        out.append("| net agent | rollout agent | net score | W/D/L |")
        out.append("|---|---|---|---|")
        for r in sorted(h2h, key=lambda r: r["a"]):
            out.append(f"| {short(r['a'])} | {r['b']} | {r['score']:.2f} | {r['wins']}/{r['draws']}/{r['losses']} |")
    rr = [r for r in cache.values() if r["a"].startswith("net:") and r["b"].startswith("net:")]
    if rr:
        out.append("\n### Generation round robin (row = later generation, its score)\n")
        out.append("| later | earlier | score | W/D/L |")
        out.append("|---|---|---|---|")
        for r in sorted(rr, key=lambda r: (r["a"], r["b"])):
            out.append(f"| {short(r['a'])} | {short(r['b'])} | {r['score']:.2f} | {r['wins']}/{r['draws']}/{r['losses']} |")
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "report.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


def cmd_plots(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cache = load_cache()
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
    # win rate vs generation for each suite opponent
    rows = {}
    for r in cache.values():
        if r["a"].startswith(f"net:checkpoints/{a.run}/") and r["a"].endswith(f":{EVAL_SIMS}") and r["b"] in SUITE:
            g = int(r["a"].split("gen_")[1][:3])
            rows.setdefault(r["b"], []).append((g, r["score"]))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab10")
    for i, opp in enumerate(SUITE):
        if opp in rows:
            pts = sorted(rows[opp])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3, lw=1.2, label=opp, color=cmap(i))
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("generation")
    ax.set_ylabel(f"score vs opponent (net + {EVAL_SIMS}-sim MCTS)")
    ax.set_ylim(-0.02, 1.1)
    ax.set_title("Tiny AlphaZero vs the fixed opponent suite (98 games each, both colours)")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    if rows:
        shade_phases(ax, max(p[0] for v in rows.values() for p in v))
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "media", "suite_vs_generation.png"), dpi=130)
    # Elo vs generation
    ep = os.path.join(RES, "elo.json")
    if os.path.exists(ep):
        with open(ep) as f:
            elo = json.load(f)
        gens = sorted((int(k.split("gen_")[1][:3]), v) for k, v in elo.items()
                      if k.startswith(f"net:checkpoints/{a.run}/") and k.endswith(f":{EVAL_SIMS}"))
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot([g for g, _ in gens], [v for _, v in gens], "o-", color="#2a6f97", label=f"net + {EVAL_SIMS} sims")
        last_y = -1e9
        for opp in sorted((o for o in SUITE if o in elo), key=lambda o: elo[o]):
            ax.axhline(elo[opp], color="gray", lw=0.6, ls=":")
            y = max(elo[opp], last_y + 28)
            last_y = y
            ax.text(ax.get_xlim()[1] if gens else 0, y, " " + opp, fontsize=7, va="center", color="gray")
        if gens:
            shade_phases(ax, gens[-1][0])
        ax.set_xlabel("generation")
        ax.set_ylabel("Elo (random = 0)")
        ax.set_title("Generation-to-generation progress (Bradley-Terry fit over all matches)")
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "media", "elo_vs_generation.png"), dpi=130)
    # training curves from the per-generation log
    lp = os.path.join(HERE, "logs", f"{a.run}.jsonl")
    if os.path.exists(lp):
        recs = [json.loads(l) for l in open(lp)]
        g = [r["gen"] for r in recs]
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
        axs[0].plot(g, [r["policy_loss"] for r in recs], label="policy CE")
        axs[0].plot(g, [r["value_loss"] for r in recs], label="value MSE")
        axs[0].set_title("training loss"); axs[0].legend(); axs[0].set_xlabel("generation")
        axs[1].plot(g, [r["first_win"] for r in recs], label="first player wins")
        axs[1].plot(g, [r["draws"] for r in recs], label="draws")
        axs[1].set_title("self-play outcomes"); axs[1].legend(); axs[1].set_xlabel("generation"); axs[1].set_ylim(0, 1)
        axs[2].plot(g, [r["avg_len"] for r in recs])
        axs[2].set_title("self-play game length (recorded plies)"); axs[2].set_xlabel("generation")
        for ax in axs:
            shade_phases(ax, g[-1])
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "media", "training_curves.png"), dpi=120)
        epath = os.path.join(HERE, "logs", f"{a.run}_eval.jsonl")
        ev = []
        if os.path.exists(epath):
            ev = [(e["gen"], {k: v for k, v in e.items() if k != "gen"}) for e in map(json.loads, open(epath))]
        if ev:
            fig, ax = plt.subplots(figsize=(7, 4))
            for opp in [k for k in ev[0][1] if k != "eval_s"]:
                ax.plot([e[0] for e in ev], [e[1][opp] for e in ev], "o-", label=opp)
            ax.axhline(0.5, color="gray", ls="--", lw=0.8)
            ax.set_ylim(-0.02, 1.02); ax.set_xlabel("generation"); ax.set_ylabel("score (98 games)")
            ax.set_title("inline eval during training (net + 100 sims)"); ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(HERE, "media", "inline_eval.png"), dpi=120)
    # symmetry-augmentation ablation: inline-eval curves at equal generation
    pm, pn = (os.path.join(HERE, "logs", f) for f in ("main_eval.jsonl", "noflip_eval.jsonl"))
    if os.path.exists(pm) and os.path.exists(pn):
        em = [json.loads(l) for l in open(pm)]
        en = [json.loads(l) for l in open(pn)]
        gmax = max(e["gen"] for e in en)
        em = [e for e in em if e["gen"] <= gmax]
        opps = [k for k in en[0] if k not in ("gen", "eval_s")]
        fig, axs = plt.subplots(1, len(opps), figsize=(4.2 * len(opps), 3.6), squeeze=False)
        for ax, o in zip(axs[0], opps):
            ax.plot([e["gen"] for e in em], [e[o] for e in em], "o-", ms=3, label="with left-right flip (main run)")
            ax.plot([e["gen"] for e in en], [e[o] for e in en], "s-", ms=3, label="no flip augmentation")
            ax.set_title(f"vs {o}"); ax.set_xlabel("generation"); ax.set_ylim(-0.02, 1.02)
            ax.axhline(0.5, color="gray", lw=0.6, ls="--")
        axs[0][0].set_ylabel("score (98 games, net + 100 sims)")
        axs[0][0].legend(fontsize=8, loc="lower right")
        fig.suptitle("Ablation: symmetry augmentation (same config otherwise, one seed each)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "media", "ablation_flip.png"), dpi=120)
    # matched-budget comparison
    rows = {}
    lp = os.path.join(RES, "latency.json")
    lat = json.load(open(lp)) if os.path.exists(lp) else None
    bgen = lat["gen"] if lat else None
    for r in cache.values():
        b = r["b"]
        if b not in ("greedy", "ab4", "ab8"):
            continue
        A = r["a"]
        if A.startswith("net:") and (bgen is None or f"gen_{bgen:03d}:" not in A):
            continue
        if A.startswith("net:"):
            n = int(A.split(":")[-1]); kind = "policy only" if n == 0 else "net + MCTS (AlphaZero)"
        elif A.startswith("uct"):
            n = int(A[3:]); kind = "rollout MCTS (no net)"
        elif A.startswith("mcts0:"):
            n = int(A.split(":")[1]); kind = "PUCT, uniform prior, no value (no net)"
        else:
            continue
        rows.setdefault(b, {}).setdefault(kind, []).append((n, r["score"]))
    if rows:
        opps = [o for o in ("greedy", "ab4", "ab8") if o in rows]
        fig, axs = plt.subplots(1, len(opps) + (1 if lat else 0), figsize=(5 * (len(opps) + 1), 4), squeeze=False)
        for ax, o in zip(axs[0], opps):
            for kind, pts in sorted(rows[o].items()):
                pts = sorted(set(pts))
                if kind == "policy only":
                    ax.axhline(pts[0][1], color="k", ls=":", label="policy only (1 net call)")
                    continue
                ax.plot([p[0] for p in pts if p[0] > 0], [p[1] for p in pts if p[0] > 0], "o-", label=kind)
            ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.axhline(0.5, color="gray", lw=0.6)
            ax.set_xlabel("simulations per move"); ax.set_ylabel(f"score vs {o}")
            ax.set_title(f"vs {o} (98 games per point)")
        axs[0][0].legend(fontsize=8)
        if lat:
            # same data vs ab8, but on a wall-clock axis (single-game ms per move, measured)
            ax = axs[0][len(opps)]
            ms = lat["ms_per_move"]
            for kind, pts in sorted(rows.get("ab8", {}).items()):
                xs, ys = [], []
                for n, sc in sorted(set(pts)):
                    key = (net_spec(a.run, bgen, n) if kind in ("policy only", "net + MCTS (AlphaZero)")
                           else f"uct{n}" if kind.startswith("rollout") else f"mcts0:{n}")
                    if key in ms:
                        xs.append(ms[key]); ys.append(sc)
                if xs:
                    ax.plot(xs, ys, "o-" if len(xs) > 1 else "*", ms=6 if len(xs) > 1 else 12, label=kind)
            ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02); ax.axhline(0.5, color="gray", lw=0.6)
            ax.set_xlabel("measured ms per move, one game at a time"); ax.set_ylabel("score vs ab8")
            ax.set_title("vs ab8, matched on wall-clock time"); ax.legend(fontsize=7)
        fig.suptitle(f"Generation {bgen}: policy only vs search without a net vs net + search", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "media", "budget_comparison.png"), dpi=120)
    print("plots written")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["suite", "rr", "budget", "latency", "baselines", "elo", "plots", "report"])
    ap.add_argument("--run", default="main")
    ap.add_argument("--gens", default="all")
    ap.add_argument("--gen", type=int, default=0)
    ap.add_argument("--sims", type=int, default=EVAL_SIMS)
    ap.add_argument("--budgets", default="16,64,256,1024")
    ap.add_argument("--opps", default="greedy,ab4,ab8")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop starting new matches after this")
    a = ap.parse_args()
    if a.max_minutes:
        DEADLINE[0] = time.time() + 60 * a.max_minutes
    try:
        dict(report=cmd_report, latency=cmd_latency, suite=cmd_suite, rr=cmd_rr, budget=cmd_budget, baselines=cmd_baselines, elo=cmd_elo, plots=cmd_plots)[a.cmd](a)
    except OutOfTime:
        print("time budget reached; partial results cached", flush=True)


if __name__ == "__main__":
    main()
