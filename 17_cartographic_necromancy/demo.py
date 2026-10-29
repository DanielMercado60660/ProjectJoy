"""Render PNG strips from results/samples_<set>.npz (written by evaluate.py).

Columns: ruin | diffusion | stub_lines | U-Net | ground truth. Inside the hole (blue outline) restored road
is white; known road is grey. Each boundary stub is drawn green if the set of stubs it connects to through
the hole matches ground truth exactly, red if not (missed or false connection).
Outputs media/strip_<set>.png and media/plausible_but_wrong_<method>.png.
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
from metrics import score, stubs

METHODS = ["diffusion", "stub_lines", "unet"]
Z = 4


def tile(known, hole, restored, gt, show_stubs=True):
    rgb = np.zeros(gt.shape + (3,), np.uint8)
    rgb[known.astype(bool)] = (110, 110, 110)
    hb = hole.astype(bool)
    rgb[hb] = (18, 22, 40)
    if restored is not None:
        rgb[hb & restored.astype(bool)] = (245, 245, 245)
    if show_stubs and restored is not None:
        lab, n = stubs(gt, hole)
        ok = score(gt, hole, restored)["stub_ok"]
        for i in range(n):
            rgb[lab == i + 1] = (40, 210, 80) if ok[i] else (235, 50, 50)
    im = Image.fromarray(rgb).resize((gt.shape[1] * Z, gt.shape[0] * Z), Image.NEAREST)
    ys, xs = np.nonzero(hb)
    ImageDraw.Draw(im).rectangle([xs.min() * Z - 1, ys.min() * Z - 1, (xs.max() + 1) * Z, (ys.max() + 1) * Z],
                                 outline=(70, 140, 255))
    return im


def strip(d, idx, path, title):
    cols = ["ruin"] + METHODS + ["ground truth"]
    W, H = 64 * Z, 64 * Z
    pad, top = 6, 34
    canvas = Image.new("RGB", (len(cols) * (W + pad) + pad, top + len(idx) * (H + pad) + pad), (250, 250, 250))
    dr = ImageDraw.Draw(canvas)
    dr.text((pad, 4), title, fill=(0, 0, 0))
    for c, name in enumerate(cols):
        dr.text((pad + c * (W + pad), 18), name, fill=(0, 0, 0))
    for r, k in enumerate(idx):
        gt, h = d["ims"][k], d["holes"][k]
        known = gt * (1 - h)
        tiles = [tile(known, h, None, gt, False)] + [tile(known, h, d[f"rest_{m}"][k], gt) for m in METHODS] + \
                [tile(known, h, gt, gt)]
        for c, t in enumerate(tiles):
            canvas.paste(t, (pad + c * (W + pad), top + r * (H + pad)))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--media", default="media")
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    for s in ["test_id", "test_bighole", "test_delaunay"]:
        d = dict(np.load(f"{a.results}/samples_{s}.npz"))
        idx = list(range(a.n))  # first n maps of the fixed test set: not cherry-picked
        strip(d, idx, f"{a.media}/strip_{s}.png", f"{s}: first {a.n} held-out maps (green stub = correct connectivity, red = wrong)")
    d = dict(np.load(f"{a.results}/samples_test_id.npz"))
    for m in ["unet", "stub_lines"]:
        pbw = np.nonzero((d[f"rf1_{m}"] >= 0.8) & ~d[f"ok_{m}"])[0][:a.n]
        if len(pbw):
            strip(d, list(pbw), f"{a.media}/plausible_but_wrong_{m}.png",
                  f"test_id: {m} restorations with good pixels (relaxed F1>=0.8) but wrong connectivity")
    print("wrote media/")


if __name__ == "__main__":
    main()
