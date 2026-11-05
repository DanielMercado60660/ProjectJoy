"""Print the zero-shot tables (markdown) from results/r8/nav/zeroshot.json, for REPORT.md.

    ../.venv/bin/python r8_nav/report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
ZS = HERE / "results" / "r8" / "nav" / "zeroshot.json"


def f(x, nd=2):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{nd}f}"


def p(st):
    return "–" if st["p"] is None else f"{st['pos']}/{st['n']}, p={st['p']:.2g}"


def main():
    z = json.loads(ZS.read_text())
    C = z["conditions"]
    print(f"episodes: {z['episodes']}\n")
    print("| model | camera | rho median (wrong-room) | rho pooled | err to plate, pooled median deg (wrong / masked / "
          "straight-ahead) | err vs expert motion | G_a | G_b | G_c dist | G_c bearing | beats straight | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model in ("vint", "nomad"):
        g = z["gates"].get(model, {})
        for cam in ("normal", "wide"):
            if cam not in g:
                continue
            a = C[f"{model}|{cam}|G1|0.25s"]["all"]
            w = C[f"{model}|{cam}|wrong|0.25s"]["all"]
            mk = C.get(f"{model}|{cam}|masked|0.25s", {}).get("all", {})
            gg = g[cam]
            print(f"| {model} | {cam} | {f(a['rho_median'])} ({f(w['rho_median'])}) | {f(a.get('rho_pooled'))} | "
                  f"{f(a['err_plate_pooled_median'], 1)} ({f(w['err_plate_pooled_median'], 1)} / "
                  f"{f(mk.get('err_plate_pooled_median'), 1)} / {f(a['err_straight_pooled_median'], 1)}) | "
                  f"{f(a['err_motion_pooled_median'], 1)} | {f(gg['G_a'])} | {f(gg['G_b'])} | {f(gg['G_c_distance'])} "
                  f"(d={f(gg['vs_wrong_rho']['median_diff'])}, {p(gg['vs_wrong_rho']['sign_test'])}) | "
                  f"{f(gg['G_c_bearing'])} (d={f(gg['vs_wrong_err']['median_diff'], 1)}, "
                  f"{p(gg['vs_wrong_err']['sign_test'])}) | {f(gg['straight_ahead_reading'])} "
                  f"(d={f(gg['vs_straight_err']['median_diff'], 1)}, {p(gg['vs_straight_err']['sign_test'])}) | "
                  f"{g.get('verdict', '–')} |")
    print("\nAll conditions (all episodes; per expert in zeroshot.json):\n")
    print("| condition | n ep | frames | rho med | rho last 5 s | rho vs metric dist | dist pred med | err plate | "
          "err wp2 | err motion | err last 5 s | err straight | err sample-mean |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in C.items():
        a = s["all"]
        print(f"| {name} | {a['n_episodes']} | {a['n_query_frames']} | {f(a['rho_median'])} | {f(a['rho_last5s_median'])} | "
              f"{f(a['rho_vs_metric_dist_median'])} | {f(a['dist_pred_median'], 1)} | {f(a.get('err_plate_pooled_median'), 1)} | "
              f"{f(a.get('err_plate_wp2_pooled_median'), 1)} | {f(a.get('err_motion_pooled_median'), 1)} | "
              f"{f(a.get('err_plate_last5s_pooled_median'), 1)} | {f(a.get('err_straight_pooled_median'), 1)} | "
              f"{f(a.get('err_plate_sample_mean_pooled_median'), 1)} |")
    print("\nPer expert (primary and wide, G1, 0.25 s):\n")
    print("| condition | expert | n ep | rho med | err plate | err straight | err motion |")
    print("|---|---|---|---|---|---|---|")
    for name, s in C.items():
        if not name.endswith("|G1|0.25s"):
            continue
        for x in ("servo", "bot"):
            a = s[x]
            print(f"| {name} | {x} | {a['n_episodes']} | {f(a['rho_median'])} | {f(a.get('err_plate_pooled_median'), 1)} | "
                  f"{f(a.get('err_straight_pooled_median'), 1)} | {f(a.get('err_motion_pooled_median'), 1)} |")
    print("\ntiming:", z.get("timing_s"))


if __name__ == "__main__":
    sys.exit(main())
