"""Move accuracy against perfect play, and the game-theoretic value of the 49 two-ply openings.

  python eval_solver.py positions   # held-out test positions (fixed seeds) -> results/solver/positions.json
  python eval_solver.py agents      # moves of policy argmax, net+MCTS, ab8 on them -> results/solver/agent_moves.json
  python eval_solver.py solve       # 4-process exact (weak = win/draw/loss) solve of every legal child of every
                                    # test position, the 25 mirror-unique 2-ply openings and the 1-ply positions
                                    # -> results/solver/solved.jsonl (appended as results arrive, resumable)
  python eval_solver.py verify      # solver vs plain full minimax on 1,300 positions with 4-16 empty cells
  python eval_solver.py protocol    # replay gen191@100 vs ab8 / uct5000 (98 games) and cross results with opening values
  python eval_solver.py report      # tables -> results/solver/report.md + summary.json, media/solver_*.png

A move is game-theoretically optimal if it keeps the position's value: in a won position any
move that still wins, in a drawn one any move that still draws; in a lost position every move
is "optimal", so lost positions are reported separately.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

import c4
from c4 import W
from arena import ALL_2PLY

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results", "solver")
CKPT = os.path.join(HERE, "checkpoints", "main", "gen_191")
PLY_RANGE = (8, 30)        # inclusive; plies already played in a test position
N_PER_SET = 500
AGENTS = ["policy", "mcts100", "mcts800", "ab8"]


def p(*a):
    print(*a, flush=True)


def canon(pos, mask):
    k = int(pos) + int(mask)
    m = 0
    for c in range(W):
        m |= ((k >> (c * 7)) & 127) << ((W - 1 - c) * 7)
    return min(k, m)


# ------------------------------------------------------------------ test positions
def gen_games(n, seed, policy_eval=None):
    """n non-terminal positions. Each game plays from the empty board to a target ply drawn
    uniformly from PLY_RANGE, choosing moves uniformly at random (policy_eval=None) or by
    sampling the net's policy at temperature 1. Games that end before the target are
    discarded and replayed. Duplicates (up to mirroring) are dropped."""
    rng = np.random.default_rng(seed)
    out, seen = [], set()
    while len(out) < n:
        k = 2 * (n - len(out)) + 16
        targets = rng.integers(PLY_RANGE[0], PLY_RANGE[1] + 1, size=k)
        pos = np.zeros(k, np.int64)
        mask = np.zeros(k, np.int64)
        moves = [[] for _ in range(k)]
        alive = np.ones(k, bool)
        for ply in range(PLY_RANGE[1]):
            act = np.nonzero(alive & (targets > ply))[0]
            if len(act) == 0:
                break
            if policy_eval is not None:
                logits, _ = policy_eval(c4.features_batch(pos[act], mask[act]))
            for j, i in enumerate(act):
                legal = c4.legal_moves(mask[i])
                if policy_eval is None:
                    c = int(rng.choice(np.flatnonzero(legal)))
                else:
                    lg = np.where(legal, logits[j].astype(np.float64), -np.inf)
                    pr = np.exp(lg - lg.max())
                    c = int(rng.choice(W, p=pr / pr.sum()))
                if c4.winning_move(pos[i], mask[i], c):
                    alive[i] = False
                    continue
                pos[i], mask[i] = c4.play(pos[i], mask[i], c)
                moves[i].append(c)
                if c4.is_full(mask[i]):
                    alive[i] = False
        for i in range(k):
            if alive[i] and len(out) < n:
                key = canon(pos[i], mask[i])
                if key not in seen:
                    seen.add(key)
                    out.append(moves[i])
    return out


def cmd_positions(a):
    from arena import load_evaluator
    os.makedirs(OUT, exist_ok=True)
    rnd = gen_games(N_PER_SET, seed=1001)
    net = gen_games(N_PER_SET, seed=1002, policy_eval=load_evaluator(CKPT))
    data = dict(ply_range=PLY_RANGE, n_per_set=N_PER_SET,
                note="random: uniform random moves, seed 1001; net: gen_191 policy sampled at T=1, seed 1002; "
                     "target ply uniform in ply_range, games ending earlier discarded, mirror duplicates dropped",
                sets=dict(random=rnd, net=net))
    with open(os.path.join(OUT, "positions.json"), "w") as f:
        json.dump(data, f)
    for k, v in data["sets"].items():
        pl = [len(m) for m in v]
        p(k, len(v), "positions, ply mean %.1f min %d max %d" % (np.mean(pl), min(pl), max(pl)))


def load_positions():
    with open(os.path.join(OUT, "positions.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------ agent moves
def cmd_agents(a):
    from arena import make_agent
    data = load_positions()
    specs = {"policy": f"net:{CKPT}:0", "mcts100": f"net:{CKPT}:100", "mcts800": f"net:{CKPT}:800", "ab8": "ab8"}
    res = {}
    for set_name, games in data["sets"].items():
        pos = np.zeros(len(games), np.int64)
        mask = np.zeros(len(games), np.int64)
        for i, mv in enumerate(games):
            pos[i], mask[i] = c4.from_moves(mv)
        res[set_name] = {}
        for name, spec in specs.items():
            c4.seed_numba(0)
            ag = make_agent(spec, max_games=256, seed=0)
            t = time.perf_counter()
            res[set_name][name] = [int(x) for x in ag.moves(pos, mask)]
            p(set_name, name, "%.1fs" % (time.perf_counter() - t))
    with open(os.path.join(OUT, "agent_moves.json"), "w") as f:
        json.dump(dict(specs=specs, moves=res, note="MCTS without noise, ties broken by rng seed 0; "
                       "ab8 random tie-breaks with numba seed 0"), f)


# ------------------------------------------------------------------ solving (worker pool)
_S = None


def _work(task):
    global _S
    import solver
    if _S is None:
        _S = solver.Solver(25)
    kind, tid, moves, node_limit, deadline = task
    if time.time() > deadline:
        return dict(kind=kind, id=tid, moves=moves, skipped=True)
    pos, mask = c4.from_moves(moves)
    _S.reset()
    t = time.perf_counter()
    if kind == "position":
        vals = _S.move_values(pos, mask, weak=True, node_limit=node_limit)
        nodes = _S.nodes
        out = dict(values=[int(v) for v in vals])
    else:   # value of the position itself, from the view of the player to move
        r = _S.solve(pos, mask, weak=(kind != "strong"), node_limit=node_limit)
        out = dict(value=int(r))
    _S.nodes = 0
    out.update(kind=kind, id=tid, moves=moves, seconds=time.perf_counter() - t, node_limit=node_limit,
               nodes=nodes if kind == "position" else _S.last_nodes)
    return out


def cmd_solve(a):
    from multiprocessing import get_context
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "solved.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if not r.get("skipped"):
                    done.add((r["kind"], r["id"]))
    deadline = time.time() + a.minutes * 60
    tasks = []
    # 28 mirror-unique 2-ply openings (heaviest first: edge columns are slowest)
    for o in ALL_2PLY:
        if tuple(o) <= (W - 1 - o[0], W - 1 - o[1]):
            tasks.append(("opening", "%d%d" % tuple(o), list(o), a.opening_nodes, deadline))
    data = load_positions()
    for set_name, games in data["sets"].items():
        for i, mv in enumerate(games):
            tasks.append(("position", "%s/%d" % (set_name, i), mv, a.position_nodes, deadline))
    for c in range(4):    # 1-ply positions (mirror-unique), last: most expensive
        tasks.append(("ply1", str(c), [c], a.ply1_nodes, deadline))
    if a.strong:     # exact scores (win distance) of the 4 mirror-unique replies to the centre opening
        tasks = [("strong", "3%d" % c, [3, c], a.strong_nodes, deadline) for c in range(4)]
    if a.strong1:    # exact scores of the 4 mirror-unique first moves
        tasks = [("strong", "%d" % c, [c], a.strong_nodes, deadline) for c in range(4)]
    tasks = [t for t in tasks if (t[0], t[1]) not in done]
    p(len(tasks), "tasks to do,", len(done), "already done; deadline in", a.minutes, "min")
    t0 = time.time()
    with get_context("spawn").Pool(a.procs) as pool, open(path, "a") as f:
        for k, r in enumerate(pool.imap_unordered(_work, tasks, chunksize=1)):
            f.write(json.dumps(r) + "\n")
            f.flush()
            if r["kind"] != "position" or k % 100 == 0:
                p("[%6.0fs] %d/%d %s %s %s" % (time.time() - t0, k + 1, len(tasks), r["kind"], r["id"],
                                               "skipped" if r.get("skipped") else
                                               (str(r.get("value", "")) if r["kind"] != "position" else "")
                                               + " %.1fs %.2e nodes" % (r["seconds"], r["nodes"])))
    p("done in %.0fs" % (time.time() - t0))


# ------------------------------------------------------------------ verification vs brute force
def cmd_verify(a):
    """Exact score and per-move win/draw/loss vs plain full minimax on positions with few empty cells."""
    import solver
    S = solver.Solver(22)
    rng = np.random.default_rng(2024)
    rec = dict(per_empties={}, mismatches=[])
    t0 = time.time()
    for empties in range(4, a.max_empties + 1):
        n_ok = n_nontriv = 0
        bf_max = 0.0
        for k in range(a.per_empties):
            greedy = k % 2 == 0
            if greedy:
                c4.seed_numba(int(rng.integers(1 << 30)))
            while True:      # random or greedy play to 42 - empties stones, nobody has won yet
                pos, mask, ok = np.int64(0), np.int64(0), True
                for _ in range(c4.NCELLS - empties):
                    c = int(c4.greedy_move(pos, mask)) if greedy else int(rng.choice(np.flatnonzero(c4.legal_moves(mask))))
                    if c4.winning_move(pos, mask, c):
                        ok = False
                        break
                    pos, mask = c4.play(pos, mask, c)
                if ok:
                    break
            n = c4.popcount(mask)
            t = time.perf_counter()
            ref = int(solver.brute_force(pos, mask, n))
            bf_max = max(bf_max, time.perf_counter() - t)
            S.reset()
            got = S.solve(pos, mask)
            n_nontriv += S.last_nodes > 10
            ok = got == ref and S.solve(pos, mask, weak=True) == np.sign(ref)
            vals = S.move_values(pos, mask, weak=True)
            for c in range(W):
                if not c4.can_play(mask, c):
                    continue
                if c4.winning_move(pos, mask, c):
                    want = 1
                else:
                    p2, m2 = c4.play(pos, mask, c)
                    want = 0 if c4.is_full(m2) else -int(np.sign(solver.brute_force(p2, m2, n + 1)))
                ok &= int(vals[c]) == want
            n_ok += bool(ok)
            if not ok:
                rec["mismatches"].append(dict(pos=int(pos), mask=int(mask), ref=ref, got=got))
        rec["per_empties"][empties] = dict(n=a.per_empties, agree=n_ok, solver_search_gt10_nodes=n_nontriv,
                                           brute_force_max_s=bf_max)
        p(empties, rec["per_empties"][empties])
    rec["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "verify_bruteforce.json"), "w") as f:
        json.dump(rec, f, indent=1)
    p("mismatches:", len(rec["mismatches"]), "in %.0fs" % rec["seconds"])


# ------------------------------------------------------------------ protocol vs opening values
def cmd_protocol(a):
    """Replays the headline match (gen 191 @100 sims vs ab8, 98 games, seed 0, same agent seeds as
    evaluate.py) with move records and crosses each game's result with its opening's solved value."""
    import arena
    solved = load_solved()
    lab = {}
    for o in ALL_2PLY:
        rep = min(tuple(o), (W - 1 - o[0], W - 1 - o[1]))
        r = solved.get(("opening", "%d%d" % rep))
        lab[tuple(o)] = None if (r is None or r.get("skipped") or abs(r["value"]) > 1) else r["value"]
    out = {}
    for spec_b in a.opponents.split(","):
        A = arena.make_agent(f"net:{CKPT}:100", seed=0)
        B = arena.make_agent(spec_b, seed=1)
        r = arena.play_match(A, B, seed=0, record=True)
        table = {}
        for i, mv in enumerate(r["moves"]):
            v = lab[tuple(mv[:2])]                         # value for the first player
            net_first = r["a_first"][i]
            theo = None if v is None else (v if net_first else -v)
            res = r["result"][i]                          # +1 first player won
            actual = 0 if res == 0 else (1 if (res == 1) == net_first else -1)
            k = f"{theo}->{actual}"
            table[k] = table.get(k, 0) + 1
        out[spec_b] = dict(score=r["score"], wins=r["wins"], draws=r["draws"], losses=r["losses"],
                           theoretical_to_actual=table,
                           note="keys: game-theoretic value of the net's side after the 2-ply opening -> actual result "
                                "(1 win, 0 draw, -1 loss); None = opening unsolved")
        p(spec_b, out[spec_b])
    with open(os.path.join(OUT, "protocol.json"), "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------ report
def load_solved():
    res = {}
    with open(os.path.join(OUT, "solved.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            key = (r["kind"], r["id"])
            if r.get("skipped") and key in res:
                continue
            res[key] = r
    return res


def cmd_report(a):
    import solver
    data = load_positions()
    am = json.load(open(os.path.join(OUT, "agent_moves.json")))["moves"]
    solved = load_solved()
    lines = ["# Solver evaluation (generated by `python eval_solver.py report`)", ""]
    summary = dict(sets={}, openings={}, ply1={})
    buckets = [(8, 14), (15, 21), (22, 30)]
    for set_name, games in data["sets"].items():
        rows = []
        for i, mv in enumerate(games):
            r = solved.get(("position", "%s/%d" % (set_name, i)))
            if r is None or r.get("skipped") or any(v == solver.UNSOLVED for v in r["values"]):
                rows.append(None)
                continue
            vals = np.array(r["values"])
            legal = vals != -999
            best = vals[legal].max()
            rows.append(dict(ply=len(mv), value=int(best), opt=(vals == best) & legal, legal=legal,
                             vals=vals, nodes=r["nodes"], sec=r["seconds"]))
        n_uns = sum(x is None for x in rows)
        ok = [x for x in rows if x is not None]
        vcount = {v: sum(x["value"] == v for x in ok) for v in (1, 0, -1)}
        S = dict(n=len(games), unsolved=n_uns, won=vcount[1], drawn=vcount[0], lost=vcount[-1],
                 solve_seconds_total=float(sum(x["sec"] for x in ok)),
                 solve_seconds_max=float(max(x["sec"] for x in ok)), agents={})
        lines += [f"## {set_name} set: {len(games)} positions, {n_uns} unsolved within the node budget",
                  f"Value for the player to move: won {vcount[1]}, drawn {vcount[0]}, lost {vcount[-1]}.", "",
                  "| agent | accuracy (won+drawn) | won: keeps the win | drawn: keeps the draw | "
                  + " | ".join(f"ply {lo}-{hi}" for lo, hi in buckets) + " |",
                  "|---|---|---|---|" + "---|" * len(buckets)]
        nl = [x for x in ok if x["value"] >= 0]
        idx = [i for i, x in enumerate(rows) if x is not None and x["value"] >= 0]

        def acc(sel):
            return (float(np.mean(sel)), len(sel)) if len(sel) else (float("nan"), 0)

        def fmt(m, n):
            se = np.sqrt(m * (1 - m) / n) if n else float("nan")
            return f"{m:.3f} ± {se:.3f} (n={n})"
        # uniform-random legal move: expected accuracy, exact from the labels
        rnd_all = [x["opt"].sum() / x["legal"].sum() for x in nl]
        agent_hits = {"random move (expected)": rnd_all}
        for ag in AGENTS:
            agent_hits[ag] = [float(rows[i]["opt"][am[set_name][ag][i]]) for i in idx]
        for ag, hits in agent_hits.items():
            hits = np.array(hits)
            vals = np.array([x["value"] for x in nl])
            plies = np.array([x["ply"] for x in nl])
            cells = [fmt(*acc(hits)), fmt(*acc(hits[vals == 1])), fmt(*acc(hits[vals == 0]))]
            cells += ["%.3f (n=%d)" % acc(hits[(plies >= lo) & (plies <= hi)]) for lo, hi in buckets]
            lines.append(f"| {ag} | " + " | ".join(cells) + " |")
            S["agents"][ag] = dict(acc=acc(hits)[0], n=len(hits), acc_won=acc(hits[vals == 1])[0],
                                   acc_drawn=acc(hits[vals == 0])[0],
                                   by_ply={f"{lo}-{hi}": acc(hits[(plies >= lo) & (plies <= hi)])
                                           for lo, hi in buckets})
            if ag != "random move (expected)":
                # what kind of mistakes: win -> draw / win -> loss / draw -> loss
                mist = {"win->draw": 0, "win->loss": 0, "draw->loss": 0}
                for j, i in enumerate(idx):
                    x = rows[i]
                    v2 = int(x["vals"][am[set_name][ag][i]])
                    if v2 < x["value"]:
                        mist[{(1, 0): "win->draw", (1, -1): "win->loss", (0, -1): "draw->loss"}[(x["value"], v2)]] += 1
                S["agents"][ag]["mistakes"] = mist
        lines += ["", "Mistake types (count): " + "; ".join(
            f"{ag}: " + ", ".join(f"{k} {v}" for k, v in S["agents"][ag]["mistakes"].items()) for ag in AGENTS),
            f"Lost positions ({vcount[-1]}) are excluded above: every move is optimal there.",
            f"Solve time (weak solve of every legal child, summed over positions, busy machine): {S['solve_seconds_total']:.0f} s, "
            f"max {S['solve_seconds_max']:.1f} s per position.", ""]
        summary["sets"][set_name] = S
    # openings
    lab = {}
    for o in ALL_2PLY:
        rep = min(tuple(o), (W - 1 - o[0], W - 1 - o[1]))
        r = solved.get(("opening", "%d%d" % rep))
        lab[tuple(o)] = None if (r is None or r.get("skipped") or r["value"] == solver.UNSOLVED) else r["value"]
    lines += ["## The 49 two-ply openings (value for the first player; rows = first move, cols = reply)", "",
              "| first \\ reply | " + " | ".join("abcdefg") + " |", "|---|" + "---|" * W]
    sym = {1: "W", 0: "D", -1: "L", None: "?"}
    for c1 in range(W):
        lines.append(f"| {'abcdefg'[c1]} | " + " | ".join(sym[lab[(c1, c2)]] for c2 in range(W)) + " |")
    cnt = {s: sum(1 for v in lab.values() if v == k) for k, s in sym.items()}
    lines += ["", "Counts: " + ", ".join(f"{k} {v}" for k, v in cnt.items()), ""]
    summary["openings"] = {"%d%d" % k: v for k, v in lab.items()}
    osec = [r["seconds"] for (k, _), r in solved.items() if k == "opening" and not r.get("skipped")]
    if osec:
        lines.append(f"Opening solve time (busy machine): {len(osec)} solves, summed {sum(osec):.0f} s, max {max(osec):.0f} s.")
    # 1-ply
    lines += ["", "## 1-ply positions (value for the first player after their first move)", ""]
    for c in range(W):
        rep = min(c, W - 1 - c)
        r = solved.get(("ply1", str(rep)))
        if r is None or r.get("skipped"):
            v = "not run"
        elif r["value"] == solver.UNSOLVED:
            v = "not solved within %.1e nodes (%.0f s)" % (r["node_limit"], r["seconds"])
        else:
            v = {1: "first player wins", 0: "draw", -1: "first player loses"}[-r["value"]] + \
                " (%.0f s, %.2e nodes)" % (r["seconds"], r["nodes"])
        summary["ply1"]["abcdefg"[c]] = v
        lines.append(f"- {'abcdefg'[c]}: {v}")
    # exact scores after the centre opening
    lines += ["", "## Exact scores after 1. d (strong solve; value for the first player, Pons score)", ""]
    summary["strong_d"] = {}
    for c in range(W):
        rep = min(c, W - 1 - c)
        r = solved.get(("strong", "3%d" % rep))
        if r is None or r.get("skipped"):
            v = "not run"
        elif r["value"] == solver.UNSOLVED:
            v = "not solved within %.1e nodes (%.0f s)" % (r["node_limit"], r["seconds"])
        else:
            sc = r["value"]                      # first player is to move after 2 plies
            v = "score %+d (first player wins with stone %d, ply %s; %.0f s, %.2e nodes)" % (
                sc, 22 - abs(sc), solver.win_ply(2, sc), r["seconds"], r["nodes"]) if sc else "draw"
            summary["strong_d"]["d" + "abcdefg"[c]] = sc
        lines.append(f"- d{'abcdefg'[c]}: {v}")
    lines += ["", "## Exact scores of the first moves (strong solve; value for the first player, Pons score)", ""]
    summary["strong_ply1"] = {}
    for c in range(W):
        rep = min(c, W - 1 - c)
        r = solved.get(("strong", "%d" % rep))
        if r is None or r.get("skipped"):
            v = "not run"
        elif r["value"] == solver.UNSOLVED:
            v = "not solved within %.1e nodes (%.0f s)" % (r["node_limit"], r["seconds"])
        else:
            sc = -r["value"]                     # second player is to move after 1 ply
            v = ("score %+d (%s wins, game ends at ply %d; %.0f s, %.2e nodes)" % (
                sc, "first player" if sc > 0 else "second player", solver.win_ply(1, r["value"]),
                r["seconds"], r["nodes"])) if sc else "draw (%.0f s, %.2e nodes)" % (r["seconds"], r["nodes"])
            summary["strong_ply1"]["abcdefg"[c]] = sc
        lines.append(f"- {'abcdefg'[c]}: {v}")
    pp = os.path.join(OUT, "protocol.json")
    if os.path.exists(pp):
        pr = json.load(open(pp))
        lines += ["", "## Headline matches crossed with opening values (gen 191 @100 sims, 98 games, seed 0)", "",
                  "| opponent | score | net's side won → result W/D/L | drawn → W/D/L | lost → W/D/L |", "|---|---|---|---|---|"]
        for opp, r in pr.items():
            t = r["theoretical_to_actual"]
            cell = lambda th: "/".join(str(t.get(f"{th}->{x}", 0)) for x in (1, 0, -1))
            lines.append(f"| {opp} | {r['score']:.3f} | {cell(1)} | {cell(0)} | {cell(-1)} |")
        summary["protocol"] = pr
    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\n".join(lines))
    plot_openings(lab)


def plot_openings(lab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {1: "#2a9d8f", 0: "#e9c46a", -1: "#e76f51", None: "#cccccc"}
    txt = {1: "W", 0: "D", -1: "L", None: "?"}
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for c1 in range(W):
        for c2 in range(W):
            v = lab[(c1, c2)]
            ax.add_patch(plt.Rectangle((c2, W - 1 - c1), 0.95, 0.95, color=col[v]))
            ax.text(c2 + 0.475, W - 1 - c1 + 0.475, txt[v], ha="center", va="center", fontsize=12)
    ax.set_xlim(0, W)
    ax.set_ylim(0, W)
    ax.set_xticks(np.arange(W) + 0.475, list("abcdefg"))
    ax.set_yticks(np.arange(W) + 0.475, list("gfedcba"))
    ax.set_xlabel("second player's reply")
    ax.set_ylabel("first player's move")
    ax.set_title("2-ply openings, perfect play from here\n(W = first player wins, D = draw, L = first player loses)",
                 fontsize=10)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "media", "solver_openings.png"), dpi=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["positions", "agents", "solve", "verify", "protocol", "report"])
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--opening-nodes", type=float, default=3e9)
    ap.add_argument("--position-nodes", type=float, default=3e8)
    ap.add_argument("--ply1-nodes", type=float, default=6e9)
    ap.add_argument("--strong", action="store_true", help="solve: only the exact scores of d-a..d-d")
    ap.add_argument("--strong1", action="store_true", help="solve: only the exact scores of the first moves a..d")
    ap.add_argument("--strong-nodes", type=float, default=2.5e9)
    ap.add_argument("--opponents", default="ab8,uct5000")
    ap.add_argument("--max-empties", type=int, default=16)
    ap.add_argument("--per-empties", type=int, default=100)
    a = ap.parse_args()
    a.opening_nodes, a.position_nodes, a.ply1_nodes = int(a.opening_nodes), int(a.position_nodes), int(a.ply1_nodes)
    a.strong_nodes = int(a.strong_nodes)
    {"positions": cmd_positions, "agents": cmd_agents, "solve": cmd_solve, "verify": cmd_verify, "protocol": cmd_protocol, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
