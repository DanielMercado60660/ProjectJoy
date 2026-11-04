"""R6 tracker demo: side-by-side GIF of the R6 classical tracker (left) and the R6 learned tracker
(right) on held-out shifts, from saved eval tracks (no re-running). Green dot = true duck centre,
marker = estimate (cyan square classical, red ring CNN; drawn yellow when the frame is lost, > 5 px).

  ../.venv/bin/python demo_tracker_r6.py [--tag r6_tracker_eval_fresh] [--conds bg light distract bg2 light2 distract2]
  -> media/r6_tracker_demo.gif, media/r6_tracker_demo_strip.png
  round 2: --tag r6_tracker_eval_r2 --conds all2 bg3 light3 distract3 all3 --out media/r6_tracker_demo_r2
"""
import argparse
import numpy as np
from PIL import Image, ImageDraw
import scene_r6 as R

UP = 4


def panel(frame, gt, est, ok, kind, label):
    im = Image.fromarray(frame).resize((80 * UP, 80 * UP), Image.NEAREST)
    d = ImageDraw.Draw(im)
    gx, gy = (gt + 0.5) * UP; ex, ey = (est + 0.5) * UP
    col = (255, 230, 0) if not ok else ((0, 230, 255) if kind == "c" else (255, 40, 40))
    if kind == "c":
        d.rectangle([ex - 14, ey - 14, ex + 14, ey + 14], outline=col, width=3)
    else:
        d.ellipse([ex - 14, ey - 14, ex + 14, ey + 14], outline=col, width=3)
    d.ellipse([gx - 4, gy - 4, gx + 4, gy + 4], fill=(0, 255, 0))
    d.rectangle([0, 0, 80 * UP, 16], fill=(0, 0, 0)); d.text((4, 2), label, fill=(255, 255, 255))
    return np.array(im)


def load(cond, tag):
    T = np.load(f"results/{tag}_tracks_{cond}.npz")
    X = (R.fresh3_set(cond) if cond in R.FRESH3_CONDS else R.fresh_set(cond) if cond in R.FRESH_CONDS
         else R.heldout(cond) if cond in R.OLD_CONDS else R.dr_set("drval", 40))[0]
    return X, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="r6_tracker_eval_fresh")
    ap.add_argument("--conds", nargs="*", default=["bg", "light", "distract", "bg2", "light2", "distract2"])
    ap.add_argument("--scene", type=int, default=3)
    ap.add_argument("--classical", default="classical_r6")
    ap.add_argument("--cnn", default="cnn_r6_kf")
    ap.add_argument("--out", default="media/r6_tracker_demo")
    a = ap.parse_args()
    rows = []
    for cond in a.conds:
        X, T = load(cond, a.tag)
        i = a.scene; fr = []
        for f in range(X.shape[1]):
            cols = []
            for kind, nm in (("c", a.classical), ("n", a.cnn)):
                e = T[nm][i, f]; ok = np.linalg.norm(e - T["gt"][i, f]) <= 5
                lost = (np.linalg.norm(T[nm][i] - T["gt"][i], axis=-1) > 5)[T["vis"][i]].mean()
                cols.append(panel(X[i, f], T["gt"][i, f], e, ok, kind, f"{cond} {nm}  scene lost {lost*100:.0f}%"))
            fr.append(np.concatenate([cols[0], np.zeros((80 * UP, 6, 3), np.uint8), cols[1]], 1))
        rows.append(fr)
    frames = [np.concatenate([r[f] for r in rows], 0) for f in range(len(rows[0]))]
    small = [Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2), Image.BILINEAR) for f in frames]
    small[0].save(a.out + ".gif", save_all=True, append_images=small[1:], duration=150, loop=0)
    strip = np.concatenate([np.concatenate([r[f] for f in (5, 15, 25, 35)], 1) for r in rows], 0)
    Image.fromarray(strip).resize((strip.shape[1] // 2, strip.shape[0] // 2), Image.BILINEAR).save(a.out + "_strip.png")
    print("wrote", a.out + ".gif", a.out + "_strip.png")


if __name__ == "__main__":
    main()
