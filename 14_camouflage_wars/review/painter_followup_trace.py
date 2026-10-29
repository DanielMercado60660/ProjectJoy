"""Reviewer (follow-up 1): trace RESULTS numbers, single-random-draw control, paired SE fine-tune vs control."""
import json, numpy as np
o=np.load("results/painter/scores.npz"); f=np.load("results/painter_round1/scores.npz"); c=np.load("results/painter_round1_control/scores.npz")
s=json.load(open("results/painter/summary.json"))["table"]
t=s["all"]
for D in ["unet0","unet1","base","base_er"]:
    print(D, "mean %.3f cand0 %.3f"%(t[f"{D}/random_mean"],t[f"{D}/random_cand0"]), {P:round(t[f"{D}/painted_by_{P}"],3) for P in ["unet0","unet1","base","base_er","statdist"]}, "det rnd %.2f u0 %.2f"%(t[f"{D}/random_det"],t[f"{D}/painted_by_unet0_det"]))
for bn in ["delta=0.25","delta=0.5","delta=0.75","delta=1.0"]:
    t=s[bn]; print(bn, {k:round(t[k],3) for k in ["unet0/random_mean","unet0/random_cand0","unet0/painted_by_unet0","unet0/random_det","unet0/painted_by_unet0_det","unet1/random_mean","unet1/painted_by_unet0","unet1/painted_by_unet0_det","base_er/random_mean","base_er/painted_by_unet0"]})
r=np.arange(400)
# random single draw per scene (seeded random index) vs mean
rng=np.random.default_rng(123); ri=rng.integers(0,64,400)
for D in ["unet0","unet1","base","base_er"]:
    x=o[D][r,ri]; print("single-random-draw",D,round(x.mean(),3),"det",round((x>=.5).mean(),2))
# paired SE
pf=f["unet0"][r,f["unet0_soft"].argmin(1)]; pc=c["unet0"][r,c["unet0_soft"].argmin(1)]
d=pf-pc; print("paired diff %.4f se %.4f"%(d.mean(), d.std(ddof=1)/20), "ft %.3f ctrl %.3f"%(pf.mean(),pc.mean()))
for dd in [0.25,0.5,0.75,1.0]:
    m=o["delta"]==dd; print(dd, round(d[m].mean(),3), round(d[m].std(ddof=1)/10,3))
# candidate scores identical across runs for detectors not changed
for k in ["unet1","base","base_er"]: print(k,"same across runs", np.array_equal(o[k],f[k]), np.array_equal(o[k],c[k]))
print("mask same", np.array_equal(o["mask"],f["mask"]))
# soft-vs-hard pick: does hard IoU mean of argmin-soft equal?
print("families", set(o["family"]))
# selection noise reference: min over 64 for unet0 on random? max-gap
a=json.load(open("results/painter/analysis.json")); print({k:round(v,3) for k,v in a.items() if "all" not in k or True} if False else "")
for k,v in a.items():
    if not isinstance(v,dict): print(k, round(v,3))
print(a["all"])
print(json.load(open("results/painter_round1/compare.json")))
