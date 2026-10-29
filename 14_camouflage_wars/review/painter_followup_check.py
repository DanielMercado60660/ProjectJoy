"""Reviewer (follow-up 1): background pixel identity across candidates + share of the U-Net IoU drop due to FPs >3 px from the creature. Run from 14_camouflage_wars/."""
import numpy as np, json, sys
sys.path.insert(0,".")
import painter as P, scenes as S
from scipy.ndimage import binary_dilation
# 1) pixel identity of background across candidates
mx_far=mx_ring=0; ndiff_r=[]
for i in range(12):
    seed=9_000_000+1000*(i%4)+i
    X,m,_,_=P.render_scene((seed,P.DELTAS[i%4],8,S.HELDOUT_FAMILIES))
    for r,name in [(2,"far>2px"),(3,"far>3px")]:
        far=~binary_dilation(m,iterations=r)
        d=np.abs(X[1:].astype(int)-X[0:1].astype(int))[:,far].max()
        print(seed,name,"max abs diff across candidates",d)
    far1=~binary_dilation(m,iterations=1)
    ndiff_r.append(int((np.abs(X[1:].astype(int)-X[0:1].astype(int)).max(-1)[:,far1]>0).sum()))
print("diff pixels in 1-px..: ",ndiff_r)
# same candidate render as stored?
im=np.load("results/painter/images.npz")
X,m,_,_=P.render_scene((9_000_000,0.25,64,S.HELDOUT_FAMILIES))
print("matches stored random0:",np.array_equal(X[0],im["random0"][0]),"pick:",np.array_equal(X[im["pick_unet0"][0]],im["painted_by_unet0"][0]))
# 2) decomposition: IoU of painted pick with far FPs removed
import mlx.core as mx, model as UM
sc=np.load("results/painter/scores.npz"); M=sc["mask"]
far=np.stack([~binary_dilation(mm,iterations=3) for mm in M])
out={}
for ck_name,ck in (("unet0","checkpoints/unet"),("unet1","checkpoints/unet_seed1")):
    net=UM.load(ck)
    for key in ("random0","painted_by_unet0"):
        Xk=im[key]
        Pr=np.concatenate([np.array(mx.sigmoid(net(mx.array(Xk[i:i+200].astype(np.float32)/255))))>0.5 for i in range(0,len(Xk),200)])
        iou=lambda p:((p&M).sum((1,2))/np.maximum((p|M).sum((1,2)),1))
        out[f"{ck_name}/{key}/iou"]=float(iou(Pr).mean())
        out[f"{ck_name}/{key}/iou_far_fp_removed"]=float(iou(Pr&~far).mean())
for k,v in out.items(): print(k, round(v,3))
for n in ("unet0","unet1"):
    tot=out[f"{n}/random0/iou"]-out[f"{n}/painted_by_unet0/iou"]
    loc=out[f"{n}/random0/iou_far_fp_removed"]-out[f"{n}/painted_by_unet0/iou_far_fp_removed"]
    print(n,"total drop %.3f, drop with far FPs removed %.3f -> far-FP share %.0f%%"%(tot,loc,100*(1-loc/tot)))
json.dump(out,open("results/painter/review_nonlocal_decomp.json","w"),indent=1)
