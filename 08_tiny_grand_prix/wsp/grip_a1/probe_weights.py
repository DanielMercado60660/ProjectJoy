"""Handmade observations and exact network derivatives, no environment code."""
import json
import numpy as np
from edit_weights import load, forward

ray_sets = {
    "straight": [.10, .115, .174, .324, .95, .95, .324, .174, .115, .10],
    "left_gentle": [.12, .14, .20, .32, .50, .75, .45, .20, .10, .08],
    "left_tight": [.13, .14, .18, .24, .30, .40, .50, .30, .10, .08],
}
ray_sets["right_gentle"] = ray_sets["left_gentle"][::-1]
ray_sets["right_tight"] = ray_sets["left_tight"][::-1]
d = load()
xs = []
labels = []
for name, rays in ray_sets.items():
    for speed in [9, 15, 21, 27]:
        x = np.array(rays + [speed / 30, 0, 0], np.float32)
        xs.append(x)
        labels.append((name, speed))
x = np.stack(xs)
h1, h2, z, a = forward(d, x)
# Derivative of each output pre-tanh logit with respect to h1 preactivation.
d_h1 = (1-h1*h1)[:, :, None] * np.einsum("ij,nj,jk->nik", d["W2"], 1-h2*h2, d["W3"])
speed_paths = d_h1 * d["W1"][10, :, None]
h2_speed = (1-h2*h2) * (((1-h1*h1)*d["W1"][10]) @ d["W2"])
h2_paths = h2_speed[:, :, None] * d["W3"]
for i, (name, speed) in enumerate(labels):
    print(name, speed, "action", np.round(a[i], 4).tolist(), "dlogit/dspeed_normalized", np.round(speed_paths[i].sum(0),4).tolist())
for paths, name in [(speed_paths, "h1"), (h2_paths, "h2")]:
    mean = paths.mean(0)
    order = np.argsort(-np.abs(mean[:, 1]))
    print(name, "speed->pedal paths, with steer collateral")
    for j in order[:12]:
        print(int(j), "mean", np.round(mean[j], 4).tolist(), "min/max pedal", np.round([paths[:,j,1].min(), paths[:,j,1].max()],4).tolist())
