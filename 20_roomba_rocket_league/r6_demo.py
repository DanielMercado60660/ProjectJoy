"""Round 6 demo: GIF/MP4 grids of the round-6 attacker vs the held-out opponents, plus an HTML page.

    ../.venv/bin/python r6_demo.py --att r6_s0=checkpoints/r6_s0.pkl --seed-base <held-out seed> --outdir media

Clips use the FIRST 4 held-out starts (unselected). Blue = attacker (attacks the right goal), red = opponent.
Also renders the scripted attacker on the same starts for comparison, and the round-4 policy if given.
"""
import argparse, json, os
import numpy as np
from PIL import Image
import render
from env import OUT_NAMES
from r6_eval import make_attacker, run_matches


def grid(pol, opp, seed, caption, path, n=64, start="heldout", idx=(0, 1, 2, 3)):
    res = run_matches(pol, opp, start, n, seed, record=list(idx))
    seqs = [[render.draw_frame(s, f"{caption} #{i}: {OUT_NAMES.get(int(res['outcome'][i]), '?')}")
             for s in res["frames"][i]] for i in idx]
    L = max(len(s) for s in seqs)
    w, h = seqs[0][0].size
    out = []
    for t in range(L):
        g = Image.new("RGB", (2 * w, 2 * h))
        for k, s in enumerate(seqs[:4]):
            g.paste(s[min(t, len(s) - 1)], ((k % 2) * w, (k // 2) * h))
        out.append(g)
    out += [out[-1]] * 20
    render.save_clip(out, path + ".mp4", fps=20)
    render.save_clip([f.resize((w, h)) for f in out[::2]], path + ".gif", fps=10)
    oc = [OUT_NAMES.get(int(res["outcome"][i])) for i in idx]
    print("wrote", path, oc, flush=True)
    return oc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+", required=True, help="first = featured r6 attacker; others compared")
    ap.add_argument("--opps", default="interceptor,zone,presser")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--outdir", default="media")
    ap.add_argument("--prefix", default="r6_")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    pols = [make_attacker(s) for s in args.att]
    rows = []
    for opp in args.opps.split(","):
        for name, pol in pols:
            fn = f"{args.prefix}{name}_vs_{opp}"
            oc = grid(pol, opp, args.seed_base, f"{name} vs {opp}", os.path.join(args.outdir, fn))
            rows.append((opp, name, fn, oc))
            if args.quick:
                break
        if args.quick:
            break
    side = os.path.join(args.outdir, f"{args.prefix}demo_outcomes.json")
    known = json.load(open(side)) if os.path.exists(side) else {}
    for o, name, fn, oc in rows:
        known[fn] = dict(opp=o, att=name, outcomes=oc)
    json.dump(known, open(side, "w"), indent=1)
    order = lambda k: (known[k]["opp"], not known[k]["att"].startswith("r6_s"), known[k]["att"])
    html = ["<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Roomba Rocket League R6</title>",
            "<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:16px}"
            "img{max-width:100%;border:1px solid #444}.row{display:flex;flex-wrap:wrap;gap:12px}"
            ".c{flex:1 1 360px}h2{margin-top:28px}</style>",
            "<h1>Round 6: population-trained attacker vs held-out opponents</h1>",
            f"<p>Blue = attacker (scores in the right goal), red = opponent. First 4 held-out starts (seed {args.seed_base}), "
            "not selected. interceptor, zone and presser were specified before training and never trained or tuned "
            "against. r6_s0/1/2 = the three training seeds of the final recipe; scripted = hand-written striker; "
            "run4 = round-4 policy (trained only vs the goalie). Numbers: RESULTS_R6.md.</p>"]
    opps_all = list(dict.fromkeys(known[k]["opp"] for k in sorted(known, key=order)))
    for opp in opps_all:
        html.append(f"<h2>vs {opp}</h2><div class=row>")
        for fn in sorted([k for k in known if known[k]["opp"] == opp], key=order):
            html.append(f"<div class=c><b>{known[fn]['att']}</b> ({', '.join(map(str, known[fn]['outcomes']))})"
                        f"<br><img src='{fn}.gif'></div>")
        html.append("</div>")
    open(os.path.join(args.outdir, f"{args.prefix}demo.html"), "w").write("\n".join(html))
    print("wrote", os.path.join(args.outdir, f"{args.prefix}demo.html"))


if __name__ == "__main__":
    main()
