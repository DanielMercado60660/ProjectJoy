"""Listening page for a flow run (f1, f2, ...): real recording, Mimi-latent round trip (ceiling), r5 (codes), f1 (flow head), same val clips.

  python3 make_listen_f1.py --f1_tag val_t0.6_cfg2.5_last

Writes media/listen_f1.html with wavs copied to media/listen_f1/.
"""
import argparse, html, json, os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")


def judged(d):
    return {c["id"]: c for c in json.load(open(os.path.join(d, "judge.json")))["clips"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1_tag", required=True)
    ap.add_argument("--run", default="f1")
    ap.add_argument("--label", default="flow head, 1 h")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prev", default="", help="earlier run to add as a column, e.g. f2")
    ap.add_argument("--prev_label", default="")
    a = ap.parse_args()
    r5d = os.path.join(HERE, "runs", "r5", "eval_val100_t0.4_s16000")
    f1d = os.path.join(HERE, "runs", a.run, f"eval_{a.f1_tag}")
    r5, f1 = judged(r5d), judged(f1d)
    if a.prev:
        pvd = os.path.join(HERE, "runs", a.prev, f"eval_{a.f1_tag.replace('_last', '')}_last")
        pv = judged(pvd)
    f1s = json.load(open(os.path.join(HERE, "results", f"eval_{a.run}_{a.f1_tag}.json")))
    ids = sorted(set(r5) & set(f1))
    # half: where r5 struggled most; half: evenly spread, longest sentences first among the rest
    worst = sorted(ids, key=lambda i: -r5[i]["cer"])[: a.n // 2]
    rest = sorted([i for i in ids if i not in worst], key=lambda i: -len(r5[i]["ref"]))
    pick = worst + rest[:: max(1, len(rest) // (a.n - len(worst)))][: a.n - len(worst)]
    out = os.path.join(HERE, "media", f"listen_{a.run}")
    os.makedirs(out, exist_ok=True)
    rows = []
    for cid in pick:
        srcs = [("real recording", os.path.join(LJ, f"{cid}.wav"), None),
                ("Mimi latent round trip (ceiling)", os.path.join(HERE, "runs", "ceiling_lat", "val", "proj", f"{cid}.wav"), None),
                ("r5: codes, 3 h", os.path.join(r5d, "wavs", f"{cid}.wav"), r5[cid]),
                (f"{a.run}: {a.label}", os.path.join(f1d, "wavs", f"{cid}.wav"), f1[cid])]
        if a.prev:
            srcs.insert(3, (f"{a.prev}: {a.prev_label}", os.path.join(pvd, "wavs", f"{cid}.wav"), pv.get(cid)))
        cells = []
        for k, (lab, p, j) in enumerate(srcs):
            if not os.path.exists(p):
                cells.append(f"<div><div class='lab'>{lab}</div><div class='na'>missing</div></div>")
                continue
            dst = f"{cid}_{k}.wav"
            shutil.copy(p, os.path.join(out, dst))
            heard = f"<div class='heard'>heard: {html.escape(j['hyp'])} (CER {j['cer']:.1f}%)</div>" if j else ""
            cells.append(f"<div><div class='lab'>{lab}</div><audio controls preload='none' src='listen_{a.run}/{dst}'></audio>{heard}</div>")
        rows.append(f"<section><h3>“{html.escape(r5[cid]['ref'])}”</h3><div class='grid'>{''.join(cells)}</div></section>")
    sub = (f"Val sentences (LJ). Half are the clips r5 got most wrong. r5 = Gemma 270M predicting 4 Mimi codebooks, t0.4, "
           f"3 h. {a.run} = same backbone plus a flow-matching head predicting Mimi's continuous latent ({a.label}), temp "
           f"{f1s['temp']}, guidance {f1s['cfg']}. Val-100: r5 CER 2.00% UTMOS 2.30; {a.run} CER {f1s['whisper_cer']}% "
           f"UTMOS {f1s['utmos']}. "
           f"\"Heard\" is Whisper large-v3-turbo.")
    page = open(os.path.join(HERE, "media", "demo.html")).read().split("</style>")[0] + "</style>"
    page = page.replace("<title>Gemma Voice</title>", f"<title>Gemma Flow {a.run}</title>")
    page += f"</head><body><main><h1>Gemma Voice: flow head ({a.run})</h1><p class='sub'>{html.escape(sub)}</p>{''.join(rows)}</main></body></html>"
    dst = os.path.join(HERE, "media", f"listen_{a.run}.html")
    open(dst, "w").write(page)
    print(dst, len(pick), "clips")


if __name__ == "__main__":
    main()
