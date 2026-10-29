"""Gallery GIF: several organisms (one regeneration-trained NCA per target), each grown from a seed,
cut in half at step 200 (held-out damage) and hit with random blobs at step 450.

Usage: ../.venv/bin/python make_gallery.py --runs lizard_regen:lizard butterfly_regen:butterfly octopus_regen:octopus
"""
import argparse, os
import numpy as np
import mlx.core as mx
import imageio.v2 as imageio

import nca
import damage
from make_demo import tile, labelled_row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="checkpoint_name:target")
    p.add_argument("--out", default="media/gallery.gif")
    p.add_argument("--seed", type=int, default=5)
    a = p.parse_args()
    runs = []
    for r in a.runs:
        ck, tgt = r.split(":")
        path = os.path.join("checkpoints", ck, "model.safetensors")
        if os.path.exists(path):
            runs.append((ck, nca.load_model(path), nca.load_target(tgt), tgt))
        else:
            print("skip (no checkpoint):", ck)
    H, W = runs[0][2].shape[:2]
    rng = np.random.default_rng(a.seed)
    top = np.zeros((H, W), bool); top[: H // 2] = True
    sched = {200: ("top half removed (held-out)", top), 450: ("random blobs (held-out)", damage.blobs(H, W, rng))}
    xs = [mx.array(nca.make_seed(H, W)) for _ in runs]
    frames, banner = [], "growing from one cell"
    for t in range(701):
        if t in sched:
            banner = f"step {t}: {sched[t][0]}"
            xs = [mx.array(damage.apply(np.array(x), sched[t][1])) for x in xs]
        if t % 3 == 0 or t in sched:
            frames.append(np.asarray(labelled_row([tile(np.array(x)[0]) for x in xs], [r[3] for r in runs],
                                                  f"{banner}   (step {t})")))
        for i, r in enumerate(runs):
            mx.random.seed(a.seed * 10000 + t)
            xs[i] = r[1](xs[i])
        mx.eval(*xs)
    imageio.mimsave(a.out, frames, duration=0.04, loop=0)
    print("wrote", a.out, len(frames), "frames")


if __name__ == "__main__":
    main()
