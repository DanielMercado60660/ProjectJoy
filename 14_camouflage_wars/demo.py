"""Render demo strips from the saved checkpoint: scene / ground truth / U-Net / baseline.
Writes media/ood_random.png, media/ood_hardest.png, media/id_random.png and media/ood_hardest.gif.
Error colouring on prediction tiles: green = correct creature pixels, red = false positive,
blue = missed creature pixels."""
import argparse, json, os, pickle
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion
import imageio.v2 as imageio
import model as M
import baseline as B
from metrics import stat_scale
from train import predict

S = 2  # upscale


def tile(img, label=None):
    t = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((96 * S, 96 * S), Image.NEAREST)
    if label:
        d = ImageDraw.Draw(t)
        d.rectangle([0, 0, 96 * S, 13], fill=(0, 0, 0))
        d.text((3, 1), label, fill=(255, 255, 255))
    return np.array(t)


def gt_tile(x, y):
    g = y > 0
    edge = g & ~binary_erosion(g, iterations=1)
    o = x.copy()
    o[edge] = [1, 1, 1]
    o[binary_erosion(g, iterations=1) & ~binary_erosion(g, iterations=2)] = [0, 0, 0]
    return o


def err_tile(x, y, p):
    g = y > 0
    o = x * 0.45
    o[p & g] = 0.35 * o[p & g] + 0.65 * np.array([0.1, 0.95, 0.2])
    o[p & ~g] = 0.35 * o[p & ~g] + 0.65 * np.array([1.0, 0.1, 0.1])
    o[~p & g] = 0.35 * o[~p & g] + 0.65 * np.array([0.2, 0.4, 1.0])
    return o


def iou1(p, y):
    g = y > 0
    u = (p | g).sum()
    return (p & g).sum() / u if u else 1.0


def strip(X, Y, Pn, Pb, idxs, notes, title):
    cols = []
    for j, i in enumerate(idxs):
        x = X[i].astype(np.float32) / 255
        col = [tile(x, notes[j]), tile(gt_tile(x, Y[i]), "ground truth"),
               tile(err_tile(x, Y[i], Pn[j]), f"U-Net IoU {iou1(Pn[j], Y[i]):.2f}"),
               tile(err_tile(x, Y[i], Pb[j]), f"baseline IoU {iou1(Pb[j], Y[i]):.2f}")]
        cols.append(np.concatenate([np.pad(c, ((0, 4), (0, 0), (0, 0)), constant_values=255) for c in col], 0))
    body = np.concatenate([np.pad(c, ((0, 0), (0, 4), (0, 0)), constant_values=255) for c in cols], 1)
    head = Image.new("RGB", (body.shape[1], 22), (255, 255, 255))
    ImageDraw.Draw(head).text((6, 5), title, fill=(0, 0, 0))
    return np.concatenate([np.array(head), body], 0), cols


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints/unet")
    ap.add_argument("--out", default="media")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net = M.load(a.ckpt)
    choice = json.load(open("baseline_choice.json"))["best"]
    scale = stat_scale([pickle.load(open(os.path.join(a.data, "train_meta.pkl"), "rb"))])
    rng = np.random.default_rng(7)

    def load(split):
        d = np.load(os.path.join(a.data, f"{split}.npz"))
        return d["X"], d["Y"], pickle.load(open(os.path.join(a.data, f"{split}_meta.pkl"), "rb"))

    def note(m):
        cs = list(m["creatures"].values())
        kinds = ",".join("other" if c["kind"] == "other" else f"d{c['delta']:.2f}" for c in cs)
        return f"{m['bg_family']} | {kinds}"

    for split, tag in (("test_ood", "ood"), ("test_id", "id")):
        X, Y, metas = load(split)
        idx = rng.choice(len(X), a.k, replace=False)
        Pn = predict(net, X[idx]) > 0.5
        Pb = B.run(X[idx], choice["method"], choice["sigma"])
        fams = "held-out families (marble/ripples/streaks)" if tag == "ood" else "seen families, new scenes"
        img, _ = strip(X, Y, Pn, Pb, idx, [note(metas[i]) for i in idx],
                       f"Random test scenes, {fams}. rows: scene | ground truth | U-Net | classical baseline. "
                       f"green=hit red=false alarm blue=miss")
        Image.fromarray(img).save(os.path.join(a.out, f"{tag}_random.png"))
        if tag != "ood":
            continue
        # hardest held-out cases: scenes whose best-camouflaged creature has the smallest
        # texture-statistics distance to its background (model-independent difficulty)
        d = np.array([min(float(np.sqrt(((np.array(c["stat_dist_raw"]) / scale) ** 2).mean()))
                          for c in m["creatures"].values()) if m["creatures"] else np.inf for m in metas])
        hard = np.argsort(d)[:a.k]
        Pn = predict(net, X[hard]) > 0.5
        Pb = B.run(X[hard], choice["method"], choice["sigma"])
        img, cols = strip(X, Y, Pn, Pb, hard, [f"{note(metas[i])} s={d[i]:.2f}" for i in hard],
                          "HARDEST held-out scenes (closest skin/background texture statistics). "
                          "rows: scene | ground truth | U-Net | baseline. label: bg family | per creature d=camouflage knob delta or other-family | s=stat distance")
        Image.fromarray(img).save(os.path.join(a.out, "ood_hardest.png"))
        # GIF: the hardest strip, cycling scene -> truth -> U-Net -> baseline
        h = 96 * S + 4
        frames = []
        for r in range(4):
            row = np.concatenate([np.pad(c[r * h:(r + 1) * h], ((0, 0), (0, 4), (0, 0)), constant_values=255)
                                  for c in cols], 1)
            frames.append(row)
        imageio.mimsave(os.path.join(a.out, "ood_hardest.gif"), frames, duration=1.2, loop=0)
    print("wrote", sorted(os.listdir(a.out)))
