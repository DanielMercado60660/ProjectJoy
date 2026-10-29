"""Build every demo artifact from a saved checkpoint (no training).

  ../.venv/bin/python make_demo.py                 # uses results/eval.json's selected VAE
  ../.venv/bin/python make_demo.py --vae checkpoints/vae_cond_b0.5

Writes to media/:
  grooves/<style>_vae.wav, compare/<style>_{real,vae,markov_bar,bernoulli}.wav
  interp/<A>_to_<B>.wav  (9-point latent walk, 2 bars per point, tempo glides A->B)
  pattern_grid.png, interpolation_grid.png
  listening_ab/pairNN_{A,B}.wav + listening_sheet.csv, answer key in listening_ab/_answer_key/
  demo.html (self-contained step sequencer with the decoder running in JS)
"""
import argparse
import base64
import csv
import json
import os
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import baselines as B
import data
import synth
from model import load_vae, decode_numpy, encode_numpy, one_hot

STYLES, TEMPO = data.STYLES, data.TEMPO
INTERP_PAIRS = [("rock", "funk"), ("house", "breakbeat"), ("hiphop", "bossa"), ("halftime", "reggaeton")]


def parse_mode(mode):
    """'threshold' | 'sample' | 'sample_T0.7' -> (mode, temperature)."""
    if mode.startswith("sample_T"):
        return "sample", float(mode[len("sample_T"):])
    return mode, 1.0


def vae_samples(model, cfg, style_idx, n, seed, mode):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, cfg["latent"])).astype(np.float32)
    md, T = parse_mode(mode)
    h, v, o, _ = decode_numpy(model, z, np.full(n, style_idx), mode=md, rng=np.random.default_rng(seed + 1), temperature=T)
    return h, v, o, z


def pack(h, v, o):
    """Grid -> list of [step, inst, vel*100, off*100] for the HTML."""
    return [[int(s), int(i), int(round(100 * v[s, i])), int(round(100 * o[s, i]))] for s, i in zip(*np.nonzero(h))]


def draw_grid(ax, h, v, title=None):
    img = np.where(h > 0, 0.25 + 0.75 * v, 0).T
    ax.imshow(img, aspect="auto", cmap="magma", vmin=0, vmax=1, interpolation="nearest")
    for x in range(4, 32, 4):
        ax.axvline(x - 0.5, color="#555" if x % 16 else "#ddd", lw=0.5 if x % 16 else 1.2)
    ax.set_yticks(range(9))
    ax.set_yticklabels(data.SHORT, fontsize=5)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=7)


def _q(W, bias):
    """int8 weights with one scale per output row, base64 encoded."""
    W = np.array(W, dtype=np.float32)  # [out, in]
    scale = np.maximum(np.abs(W).max(1), 1e-8) / 127.0
    q = np.clip(np.round(W / scale[:, None]), -127, 127).astype(np.int8)
    return {"n_in": int(W.shape[1]), "n_out": int(W.shape[0]), "q": base64.b64encode(q.tobytes()).decode(),
            "scale": [float(f"{s:.6g}") for s in scale],
            "bias": [float(f"{b:.6g}") for b in (np.zeros(W.shape[0]) if bias is None else np.array(bias))]}


def export_decoder(model):
    if model.kind == "ar":
        g = model.rnn
        return {"kind": "ar", "latent": model.latent, "n_styles": len(STYLES), "rnn": model.rnn_size,
                "init_h": _q(model.init_h.weight, model.init_h.bias), "Wx": _q(g.Wx, g.b), "Wh": _q(g.Wh, None),
                "bhn": [float(f"{b:.6g}") for b in np.array(g.bhn)], "out": _q(model.out.weight, model.out.bias),
                "layers": [_q(model.out.weight, model.out.bias)]}
    layers = [_q(L.weight, L.bias) for L in [*model.dec, model.out]]
    return {"kind": "mlp", "latent": model.latent, "n_styles": len(STYLES), "layers": layers}


def build_html(payload, out_path, web_dir="web"):
    tpl = open(os.path.join(web_dir, "template.html")).read()
    core = open(os.path.join(web_dir, "core.js")).read()
    ui = open(os.path.join(web_dir, "ui.js")).read()
    html = (tpl.replace("/*__CORE__*/", core).replace("/*__UI__*/", ui)
            .replace("/*__DATA__*/", "const DATA = " + json.dumps(payload, separators=(",", ":")) + ";"))
    with open(out_path, "w") as f:
        f.write(html)
    return len(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default=None)
    ap.add_argument("--mode", default=None, help="threshold|sample (default: from results/eval.json demo_mode or 'sample')")
    ap.add_argument("--data", default="data/drums.npz")
    ap.add_argument("--media", default="media")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n_ab_per_style", type=int, default=1)
    a = ap.parse_args()
    ev = json.load(open("results/eval.json")) if os.path.exists("results/eval.json") else {}
    vae_path = a.vae or ev.get("demo_vae") or ev.get("selected_vae")
    mode = a.mode or ev.get("demo_mode", "sample")
    model, cfg = load_vae(vae_path)
    print("VAE", vae_path, cfg, "mode", mode)
    d = data.load(a.data)
    tr, te = data.subset(d, "train"), data.subset(d, "test")
    rng = np.random.default_rng(a.seed)
    bern, mbar = B.BernoulliModel().fit(tr), B.BarMarkovModel().fit(tr)
    M = a.media
    for sub in ("grooves", "compare", "interp", "listening_ab"):
        os.makedirs(os.path.join(M, sub), exist_ok=True)

    # ---- per-style sets ----
    sets = {"real": {}, "vae": {}, "bernoulli": {}, "markov_bar": {}}
    for si, st in enumerate(STYLES):
        pool = np.flatnonzero(te["style"] == si)
        idx = rng.choice(pool, 6, replace=len(pool) < 6)
        sets["real"][st] = [(te["hits"][i], te["vel"][i], te["off"][i]) for i in idx]
        h, v, o, _ = vae_samples(model, cfg, si, 6, a.seed + 100 * si, mode)
        sets["vae"][st] = list(zip(h, v, o))
        for name, m in (("bernoulli", bern), ("markov_bar", mbar)):
            h, v, o = m.sample(np.full(4, si), np.random.default_rng(a.seed + 10 * si))
            sets[name][st] = list(zip(h, v, o))
        synth.write(f"{M}/grooves/{st}_vae.wav", synth.render_sequence(sets["vae"][st][:4], TEMPO[st], loops_each=1))
        for name in sets:
            synth.write(f"{M}/compare/{st}_{name}.wav", synth.render(*sets[name][st][0], bpm=TEMPO[st], loops=2))
    # also render grooves from the validation-NLL winner when it is not the demo model (e.g. the AR VAE)
    other = ev.get("selected_vae")
    if not a.vae and other and other != vae_path:
        om, ocfg = load_vae(other)
        oname = other.rstrip("/").split("/")[-2 if other.rstrip("/").endswith("best") else -1]
        omode = ev.get("sets") and min((k.split(":", 1)[1] for k in ev["sets"] if k.startswith(oname + ":")),
                                       key=lambda md: ev["sets"][f"{oname}:{md}"]["c2st_acc"])
        for si, st in enumerate(STYLES):
            h, v, o, _ = vae_samples(om, ocfg, si, 4, a.seed + 100 * si, omode)
            synth.write(f"{M}/grooves/{st}_{oname}.wav", synth.render_sequence(list(zip(h, v, o)), TEMPO[st], loops_each=1))
        print(f"also rendered {oname} ({omode}) grooves")
    print("grooves + compare WAVs written")

    # ---- interpolation walks between real test patterns of two styles ----
    K = 9
    alphas = np.linspace(0, 1, K)
    interps = []
    fig, axes = plt.subplots(len(INTERP_PAIRS), K, figsize=(K * 1.6, len(INTERP_PAIRS) * 1.2))
    for r, (sa, sb) in enumerate(INTERP_PAIRS):
        ia = rng.choice(np.flatnonzero(te["style"] == STYLES.index(sa)))
        ib = rng.choice(np.flatnonzero(te["style"] == STYLES.index(sb)))
        mu, _ = encode_numpy(model, te["hits"][[ia, ib]], te["vel"][[ia, ib]], te["off"][[ia, ib]], te["style"][[ia, ib]])
        z = (1 - alphas)[:, None] * mu[0] + alphas[:, None] * mu[1]
        sv = (1 - alphas)[:, None] * one_hot(np.array([STYLES.index(sa)]))[0] + alphas[:, None] * one_hot(np.array([STYLES.index(sb)]))[0]
        h, v, o, _ = decode_numpy(model, z, sv, mode="threshold")  # deterministic path for smooth walks
        bpms = (1 - alphas) * TEMPO[sa] + alphas * TEMPO[sb]
        synth.write(f"{M}/interp/{sa}_to_{sb}.wav", synth.render_sequence(list(zip(h, v, o)), bpms, loops_each=1))
        interps.append({"a": sa, "b": sb, "bpm": [float(x) for x in bpms], "points": [pack(*g) for g in zip(h, v, o)],
                        "z": [[float(f"{x:.4f}") for x in row] for row in z],
                        "endpoints": [pack(te["hits"][ia], te["vel"][ia], te["off"][ia]), pack(te["hits"][ib], te["vel"][ib], te["off"][ib])]})
        for k in range(K):
            draw_grid(axes[r, k], h[k], v[k], f"{sa}→{sb} α={alphas[k]:.2f}" if k in (0, K - 1) else f"α={alphas[k]:.2f}")
    fig.tight_layout()
    fig.savefig(f"{M}/interpolation_grid.png", dpi=150)
    plt.close(fig)
    print("interp WAVs + PNG written")

    # ---- pattern grid PNG: style rows x (real | VAE | bar-Markov | Bernoulli) ----
    cols = [("real", "real (test set)"), ("vae", f"VAE ({mode})"), ("markov_bar", "bar-Markov baseline"), ("bernoulli", "Bernoulli baseline")]
    fig, axes = plt.subplots(len(STYLES), len(cols), figsize=(11, 12))
    for r, st in enumerate(STYLES):
        for c, (k, lab) in enumerate(cols):
            h, v, _ = sets[k][st][0]
            draw_grid(axes[r, c], h, v, f"{st} — {lab}")
    fig.tight_layout()
    fig.savefig(f"{M}/pattern_grid.png", dpi=150)
    plt.close(fig)

    # ---- blind A/B listening folder ----
    ab = os.path.join(M, "listening_ab")
    shutil.rmtree(ab, ignore_errors=True)
    os.makedirs(os.path.join(ab, "_answer_key"))
    rows, key, pair = [], [], 0
    for r_i, rival in enumerate(("real", "markov_bar", "bernoulli")):
        for st in STYLES:
            pair += 1
            vae_g = sets["vae"][st][1 + r_i]  # a different VAE groove against each rival, none repeated
            riv_g = sets[rival][st][1 + r_i]
            vae_first = bool(rng.random() < 0.5)
            A, Bg = (vae_g, riv_g) if vae_first else (riv_g, vae_g)
            synth.write(f"{ab}/pair{pair:02d}_A.wav", synth.render(*A, bpm=TEMPO[st], loops=2))
            synth.write(f"{ab}/pair{pair:02d}_B.wav", synth.render(*Bg, bpm=TEMPO[st], loops=2))
            rows.append([f"pair{pair:02d}", st, "", ""])
            key.append({"pair": f"pair{pair:02d}", "style": st, "A": "vae" if vae_first else rival, "B": rival if vae_first else "vae"})
    order = rng.permutation(len(rows))  # shuffle listening order so rivals are not blocked together
    with open(f"{ab}/listening_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "style", "preferred (A/B)", "notes"])
        for i in order:
            w.writerow(rows[i])
    with open(f"{ab}/_answer_key/answer_key.json", "w") as f:
        json.dump({"note": "Do not open before listening. 'vae' = the model.", "pairs": key}, f, indent=1)
    with open(f"{ab}/README.txt", "w") as f:
        f.write("Do this test BEFORE opening demo.html or media/grooves|compare (they contain some of the same grooves).\n"
                "Blind A/B test. For each pair (in the order listed in listening_sheet.csv), listen to pairNN_A.wav and "
                "pairNN_B.wav and write which groove you prefer (A or B). Then score with:\n"
                "  ../.venv/bin/python score_listening.py media/listening_ab/listening_sheet.csv\n"
                "The answer key is in _answer_key/ - don't peek.\n")
    print(f"listening_ab: {pair} pairs")

    # ---- demo.html ----
    tr_mu, _ = encode_numpy(model, tr["hits"], tr["vel"], tr["off"], tr["style"])
    c = tr_mu - tr_mu.mean(0)
    _, S, Vt = np.linalg.svd(c, full_matrices=False)
    pcs = (Vt[:2] * (S[:2, None] / np.sqrt(len(c)))).astype(np.float32)  # scaled by std along each axis
    payload = {
        "styles": STYLES, "tempo": TEMPO, "instruments": data.INSTRUMENTS, "short": data.SHORT, "mode": parse_mode(mode)[0], "temperature": parse_mode(mode)[1],
        "sets": {k: {st: [pack(*g) for g in v] for st, v in s.items()} for k, s in sets.items()},
        "interps": interps, "decoder": export_decoder(model), "pcs": pcs.round(4).tolist(),
        "check": None,
    }
    zc = np.random.default_rng(0).standard_normal((1, cfg["latent"])).astype(np.float32).round(5)
    hc, _, _, pc = decode_numpy(model, zc, np.array([0]), mode="threshold")
    payload["check"] = {"z": zc[0].tolist(), "style": 0, "p": pc[0].reshape(-1).round(5).tolist(),
                        "h": hc[0].reshape(-1).astype(int).tolist()}
    n = build_html(payload, f"{M}/demo.html")
    print(f"demo.html {n / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
