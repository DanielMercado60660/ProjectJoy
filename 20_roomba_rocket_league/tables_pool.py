"""Follow-up (round 3): side-by-side table, run 4 vs opponent-pool PPO vs scripted attackers.

    ../.venv/bin/python tables_pool.py results/pool/eval_pool_final.json [results/pool/eval_pool_best.json]
Sources: results/eval_run4_final.json (run 4 + scripted attacker), results/review_checks.json (aim attacker),
and the pool eval json(s) given. All on the same held-out seeds 1001-1003, 500 matches per cell.
"""
import json, sys
r4 = json.load(open("results/eval_run4_final.json"))
rc = json.load(open("results/review_checks.json"))
pools = [(f"pool PPO ({p.split('/')[-1].replace('.json','').replace('eval_','')})", json.load(open(p))) for p in sys.argv[1:]]
rows = [("run 4 PPO (goalie only)", r4, "ppo"), *[(n, d, "ppo") for n, d in pools],
        ("scripted attacker", r4, "scripted_attacker"), ("aim attacker (0.65)", rc, "aim_attacker")]
seen = {"goalie": "seen by both", "keeper": "UNSEEN by both", "chaser": "seen by pool only"}
for opp in ["goalie", "keeper", "chaser"]:
    print(f"\n**vs {opp}** ({seen[opp]}): goal % / conceded+own-goal %\n")
    print("| policy | heldout | corner | kickoff |\n|---|---|---|---|")
    for name, d, key in rows:
        cells = []
        for s in ["heldout", "corner", "kickoff"]:
            m = d[f"{key}|{opp}|{s}"]
            cells.append(f"{100*m['goal']:.1f} / {100*(m['conceded']+m['own_goal']):.1f}")
        print(f"| {name} | " + " | ".join(cells) + " |")
