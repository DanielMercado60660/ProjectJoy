"""Print markdown tables from results/*.json (used to write RESULTS.md; no numbers by hand).

  ../.venv/bin/python summarize_results.py
"""
import json, os


def heuristic(path, title):
    if not os.path.exists(path):
        print(f"\n({path} not found: not measured)\n"); return
    S = json.load(open(path))["summary"]
    print(f"\n### {title}\n")
    print(f"{S['queries']} held-out queries; network inference median {S['infer_ms_median']:.1f} ms "
          f"(one forward pass per map+goal, MLX on the shared GPU)\n")
    print("| method | expanded / A* (mean) | expanded (median) | cost above optimal: mean | p95 | max | optimal paths | search ms (median) | total ms incl. inference |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, v in S.items():
        if isinstance(v, dict):
            print(f"| {k} | {v['expanded_vs_astar_mean_ratio']:.3f} | {v['expanded_median']:.0f} | {v['subopt_mean_pct']:.2f}% | "
                  f"{v['subopt_p95_pct']:.2f}% | {v['subopt_max_pct']:.2f}% | {100 * v['frac_optimal']:.0f}% | "
                  f"{v['search_ms_median']:.2f} | {v['total_ms_median_incl_inference']:.2f} |")


def bridge(path="results/bridge.json"):
    if not os.path.exists(path):
        print("\n(bridge: not measured)\n"); return
    S = json.load(open(path))["summary"]
    print("\n### Bridge collapse\n")
    print(f"{S['queries']} queries; goal unreachable after collapse: {S['failed_no_path_after_collapse']}; "
          f"D* Lite cost == A* cost in every case: {S['dstar_astar_consistent']}\n")
    print("| planner | initial expanded (median) | initial ms (median) | replan expanded (median / mean) | replan ms (median / mean) |")
    print("|---|---|---|---|---|")
    for m, name in (("dstar", "D* Lite (incremental)"), ("astar", "A* from scratch"), ("dijkstra", "Dijkstra from scratch")):
        print(f"| {name} | {S[m + '_init_expanded_median']:.0f} | {S[m + '_init_ms_median']:.2f} | "
              f"{S[m + '_replan_expanded_median']:.0f} / {S[m + '_replan_expanded_mean']:.0f} | "
              f"{S[m + '_replan_ms_median']:.2f} / {S[m + '_replan_ms_mean']:.2f} |")
    print(f"\n- D* Lite expanded fewer nodes than A* in {100 * S['frac_dstar_fewer_expansions_than_astar']:.0f}% of replans, "
          f"and was faster in wall-clock in {100 * S['frac_dstar_faster_than_astar']:.0f}%.")
    print(f"- Cost per expansion: D* Lite {S['dstar_us_per_expansion']:.1f} µs vs A* {S['astar_us_per_expansion']:.1f} µs.")
    print(f"- Small detours (below-median detour ratio): D* Lite median {S['dstar_replan_expanded_median_small_detour']:.0f} "
          f"vs A* {S['astar_replan_expanded_median_small_detour']:.0f} expansions. Large detours: D* Lite "
          f"{S['dstar_replan_expanded_median_large_detour']:.0f} vs A* {S['astar_replan_expanded_median_large_detour']:.0f}.")
    print(f"- Travel cost: planned {S['mean_planned_cost']:.1f} → realized {S['mean_travel_cost']:.1f} "
          f"vs omniscient {S['mean_omniscient_cost']:.1f} (mean realized/omniscient {S['mean_travel_over_omniscient']:.3f}).")


def fogres(path="results/fog.json"):
    if not os.path.exists(path):
        print("\n(fog: not measured)\n"); return
    D = json.load(open(path)); S = D["summary"]
    print(f"\n### Fog of war (sensing radius {S['R']})\n")
    print("| prior | queries | failure rate | travel / optimal: mean | median | p95 | cells observed (mean) | replans (mean) | D* Lite expanded total | A* expanded total | D* Lite ms total (median) | A* ms total (median) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for pr in ("optimistic", "learned", "hydrology"):
        v = S[pr]
        print(f"| {pr} | {v['queries']} | {100 * v['failure_rate']:.1f}% | {v['travel_over_opt_mean']:.3f} | {v['travel_over_opt_median']:.3f} | "
              f"{v['travel_over_opt_p95']:.3f} | {v['observed_cells_mean']:.0f} | {v['replans_mean']:.1f} | "
              f"{v.get('dstar_expanded_total_mean', float('nan')):.0f} | {v.get('astar_expanded_total_mean', float('nan')):.0f} | "
              f"{v.get('dstar_ms_total_median', float('nan')):.1f} | {v.get('astar_ms_total_median', float('nan')):.1f} |")
    print(f"\n- learned prior better than optimistic on {100 * S['learned_better_than_optimistic_frac']:.0f}% of queries, "
          f"worse on {100 * S['learned_worse_than_optimistic_frac']:.0f}%.")
    sw = S.get("radius_sweep_travel_over_opt_mean", {})
    if sw:
        print("\n| sensing radius | n | optimistic | learned | hydrology |\n|---|---|---|---|---|")
        for R, v in sw.items():
            print(f"| {R} | {v['n']} | {v['optimistic']:.3f} | {v['learned']:.3f} | {v['hydrology']:.3f} |")
    print(f"\nprior checkpoint: {json.dumps({k: D['prior_cfg'].get(k) for k in ('step', 'val_water_iou', 'val_water_acc', 'val_mland_mae')})}")


if __name__ == "__main__":
    heuristic("results/heuristic.json", "Learned heuristic (hnet, base 32) vs A* / weighted A*")
    heuristic("results/heuristic_hnet16.json", "Browser-sized learned heuristic (hnet16, base 16)")
    bridge()
    fogres()
