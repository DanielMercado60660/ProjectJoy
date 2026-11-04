"""Closed-form readout editing against an explicit function on handmade inputs.

No simulator code, traces, learned data, or optimisation over simulator returns.
"""
import itertools
import json
import argparse
import numpy as np
from edit_weights import load, forward

templates = [
    [.10, .115, .174, .324, .95, .95, .324, .174, .115, .10],
    [.12, .14, .20, .32, .50, .75, .45, .20, .10, .08],
    [.13, .14, .18, .24, .30, .40, .50, .30, .10, .08],
    [.12, .12, .14, .17, .20, .25, .34, .60, .20, .09],
    [.06, .05, .06, .07, .09, .12, .18, .50, .80, .20],
    [.18, .20, .24, .28, .33, .41, .55, .04, .03, .025],
]
templates += [r[::-1] for r in templates[1:]]


def make_inputs(validation=False):
    speeds = [2.5, 7.5, 12.5, 17.5, 22.5, 27.5] if validation else [0, 5, 10, 15, 20, 25, 30]
    previous = [-.6, .2, .6] if validation else [-.8, -.4, 0, .4, .8]
    pedals = [-.5, .5] if validation else [-1, 0, 1]
    rays = [(np.asarray(a)+np.asarray(b))/2 for a,b in zip(templates, templates[1:])] if validation else templates
    return np.asarray([list(r)+[v/30, st, pd] for r,v,st,pd in itertools.product(rays,speeds,previous,pedals)], np.float64)


def targets(d, x):
    h1, h2, z, a = forward(d,x)
    original_z = z[:,0] / .5
    gain = .5 + .5*np.clip(1-x[:,10]*30/15, 0, 1)
    return np.column_stack([h2,np.ones(len(x))]), original_z*gain


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="c07_speed_adaptive_steering.npz")
    p.add_argument("--speed-scale", type=float, default=float(np.sqrt(2)))
    p.add_argument("--speed-scale-precision", choices=["float64", "float32"], default="float64",
                   help="Call 7 used a NumPy float64 scale; call 8 used Python scalar promotion to float32.")
    args = p.parse_args()
    d = load()
    speed_scale = np.float64(args.speed_scale) if args.speed_scale_precision == "float64" else args.speed_scale
    d["W1"][10] *= speed_scale
    d["W3"][:,0] *= .5
    d["b3"][0] *= .5
    base = np.r_[d["W3"][:,0],d["b3"][0]].astype(np.float64)
    x = make_inputs()
    a, y = targets(d,x)
    # Explicit regularized linear least squares, anchored to the half-gain readout.
    # Objective: mean squared logit error + 0.001*||readout-base||^2.
    ridge = .001
    delta = np.linalg.solve(a.T@a/len(a)+ridge*np.eye(33), a.T@(y-a@base)/len(a))
    w = base+delta
    stats = {"formula":"gain(v)=0.5+0.5*clip(1-v/15,0,1)", "ridge":ridge, "speed_scale":args.speed_scale,
             "speed_scale_precision":args.speed_scale_precision}
    for name, xx in [("fit",x),("handmade_validation",make_inputs(True))]:
        aa, yy = targets(d,xx)
        error = aa@w-yy
        stats[name] = {"n":len(xx),"logit_rmse":float(np.sqrt(np.mean(error**2))),"logit_max_error":float(abs(error).max()),"action_rmse":float(np.sqrt(np.mean((np.tanh(aa@w)-np.tanh(yy))**2)))}
    d["W3"][:,0] = w[:32]
    d["b3"][0] = w[32]
    np.savez(args.output,**d)
    stats["changed"] = {k:int(np.count_nonzero(d[k]!=load()[k])) for k in d}
    print(json.dumps(stats,indent=2))
