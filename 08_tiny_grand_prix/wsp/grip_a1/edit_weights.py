"""Deterministic weight edits; never imports or runs the simulator."""
import argparse
import numpy as np


def load(path="start.npz"):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def forward(d, x):
    h1 = np.tanh(x @ d["W1"] + d["b1"])
    h2 = np.tanh(h1 @ d["W2"] + d["b2"])
    z = h2 @ d["W3"] + d["b3"]
    return h1, h2, z, np.tanh(z)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("output")
    p.add_argument("--speed-scale", type=float, default=1.0)
    p.add_argument("--pedal-bias", type=float, default=0.0)
    p.add_argument("--steer-scale", type=float, default=1.0)
    p.add_argument("--side-ray-gain", type=float, default=0.0)
    args = p.parse_args()
    d = load()
    d["W1"][10] *= args.speed_scale
    d["b3"][1] += args.pedal_bias
    d["W3"][:, 0] *= args.steer_scale
    d["b3"][0] *= args.steer_scale
    # Virtual oblique-ray contrast: left clearance > right clearance -> turn left.
    contrast = args.side_ray_gain * (d["W1"][6] - d["W1"][3])
    d["W1"][9] += contrast
    d["W1"][0] -= contrast
    np.savez(args.output, **d)
    original = load()
    counts = {k: int(np.count_nonzero(d[k] != original[k])) for k in d}
    print({"output": args.output, "changed": counts, "total_changed": sum(counts.values())})
