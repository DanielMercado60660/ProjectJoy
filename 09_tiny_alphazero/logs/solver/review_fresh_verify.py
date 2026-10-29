# Reviewer: fresh brute-force check (new seeds) + consistency checks on held-out positions.
import sys, json, time
sys.path.insert(0, "/Users/danielmercado/ProjectJoy/09_tiny_alphazero")
import numpy as np, c4, solver
S = solver.Solver(21)
rng = np.random.default_rng(int(sys.argv[4]))
c4.seed_numba(4242)
n = agree = nontriv = 0; mism = []
t0 = time.time()
for empties in range(int(sys.argv[1]), int(sys.argv[2])):
    for k in range(int(sys.argv[3])):
        greedy = k % 3 == 0
        while True:
            pos, mask, ok = np.int64(0), np.int64(0), True
            for _ in range(42 - empties):
                if greedy and rng.random() < 0.7:
                    c = int(c4.greedy_move(pos, mask))
                else:
                    c = int(rng.choice(np.flatnonzero(c4.legal_moves(mask))))
                if c4.winning_move(pos, mask, c): ok = False; break
                pos, mask = c4.play(pos, mask, c)
            if ok: break
        nm = c4.popcount(mask)
        ref = int(solver.brute_force(pos, mask, nm))
        S.reset(); got = S.solve(pos, mask); nontriv += S.last_nodes > 10
        # do NOT reset: second solves reuse a warm table (tests stale-entry handling too)
        ok = got == ref and S.solve(pos, mask, weak=True) == np.sign(ref)
        vals = S.move_values(pos, mask, weak=True)
        for c in range(7):
            if not c4.can_play(mask, c): continue
            if c4.winning_move(pos, mask, c): want = 1
            else:
                p2, m2 = c4.play(pos, mask, c)
                want = 0 if c4.is_full(m2) else -int(np.sign(solver.brute_force(p2, m2, nm + 1)))
            ok &= int(vals[c]) == want
        # exact child scores too (strong), vs brute force
        svals = S.move_values(pos, mask, weak=False)
        for c in range(7):
            if not c4.can_play(mask, c): continue
            if c4.winning_move(pos, mask, c): continue
            p2, m2 = c4.play(pos, mask, c)
            want = 0 if c4.is_full(m2) else -int(solver.brute_force(p2, m2, nm + 1))
            ok &= int(svals[c]) == want
        n += 1; agree += bool(ok)
        if not ok: mism.append((int(pos), int(mask), ref, got))
print("fresh brute force: %d/%d agree, %d nontrivial (>10 nodes), %.0fs" % (agree, n, nontriv, time.time() - t0), mism[:5])

# consistency on held-out positions: weak solve of the position itself (fresh table) == max of stored child labels,
# and sign(strong solve) == weak value, on a sample of late (cheap) positions
D = json.load(open("/Users/danielmercado/ProjectJoy/09_tiny_alphazero/results/solver/positions.json"))
sol = {}
for line in open("/Users/danielmercado/ProjectJoy/09_tiny_alphazero/results/solver/solved.jsonl"):
    r = json.loads(line)
    if r["kind"] == "position" and not r.get("skipped"): sol[r["id"]] = r["values"]
S2 = solver.Solver(24)
cnt = bad = 0
for set_name, games in D["sets"].items():
    idx = [i for i, m in enumerate(games) if len(m) >= 18][:25]
    for i in idx:
        pos, mask = c4.from_moves(games[i])
        vals = np.array(sol["%s/%d" % (set_name, i)]); lab = vals[vals != -999].max()
        S2.reset(); w = S2.solve(pos, mask, weak=True)
        S2.reset(); s = S2.solve(pos, mask, weak=False, node_limit=int(3e8))
        mp, mm = c4.mirror(int(pos), int(mask)); S2.reset(); wm = S2.solve(np.int64(mp), np.int64(mm), weak=True)
        cnt += 1
        if not (w == lab == wm and (s == solver.UNSOLVED or np.sign(s) == lab)): bad += 1; print("BAD", set_name, i, lab, w, wm, s)
print("held-out consistency: %d positions, %d inconsistent, %.0fs" % (cnt, bad, time.time() - t0))
