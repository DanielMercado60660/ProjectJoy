"""Structural scores computed only through the hole.

Stubs = 8-connected road components on the 1-px ring just outside the hole.
Connectivity is computed on (hole ∪ ring) for both ground truth and restoration, so two stubs joined
by a path *outside* the hole never count as reconnected. No skeletonization (skimage not installed):
a thresholded binary mask + 8-connectivity gives the same component structure.
"""
import numpy as np
from scipy.ndimage import label, binary_dilation
from maps import ring_of

K8 = np.ones((3, 3), bool)


def stubs(gt, hole):
    ring = ring_of(hole).astype(bool)
    lab, n = label(gt.astype(bool) & ring, K8)
    return lab, n


def stub_partition(raster, hole, stub_lab, n):
    """For each stub, the id of its component in (raster∩(hole∪ring))."""
    region = (hole.astype(bool) | (stub_lab > 0)) & raster.astype(bool)
    region |= stub_lab > 0
    comp, _ = label(region, K8)
    return np.array([comp[stub_lab == i + 1][0] for i in range(n)])


def score(gt, hole, restored):
    """restored: full binary raster (known pixels already composited). Returns dict of per-sample stats."""
    hole_b = hole.astype(bool)
    lab, n = stubs(gt, hole)
    pg = stub_partition(gt, hole, lab, n)
    pp = stub_partition(restored, hole, lab, n)
    iu = np.triu_indices(n, 1)
    sg = (pg[:, None] == pg[None, :])[iu]
    sp = (pp[:, None] == pp[None, :])[iu]
    tp = int((sg & sp).sum()); fn = int((sg & ~sp).sum()); fp = int((~sg & sp).sum())
    # per-stub correctness: its connected stub-set matches GT exactly
    stub_ok = np.array([np.array_equal(pg == pg[i], pp == pp[i]) for i in range(n)], bool)
    # boundary continuity: stub that continues into the hole in GT also continues in restoration
    cont_g, cont_p = [], []
    for i in range(n):
        nb = binary_dilation(lab == i + 1, K8) & hole_b
        cont_g.append(bool((gt.astype(bool) & nb).any()))
        cont_p.append(bool((restored.astype(bool) & nb).any()))
    cont_g = np.array(cont_g, bool); cont_p = np.array(cont_p, bool)
    g_h = gt.astype(bool) & hole_b; p_h = restored.astype(bool) & hole_b
    inter = (g_h & p_h).sum(); uni = (g_h | p_h).sum()
    # relaxed (1-px tolerance) precision/recall inside the hole, standard for thin road masks
    g_t = binary_dilation(gt.astype(bool), K8) & hole_b; p_t = binary_dilation(restored.astype(bool), K8) & hole_b
    rp = (p_h & g_t).sum() / max(p_h.sum(), 1) if p_h.sum() else float(g_h.sum() == 0)
    rr = (g_h & p_t).sum() / max(g_h.sum(), 1) if g_h.sum() else 1.0
    rf1 = 2 * rp * rr / max(rp + rr, 1e-9)
    # dangling restored components inside the hole touching no stub
    comp, nc = label(p_h, K8)
    touched = set(np.unique(comp[binary_dilation(lab > 0, K8)]).tolist()) - {0}
    return dict(n_stubs=n, tp=tp, fn=fn, fp=fp, partition_ok=bool(stub_ok.all()), stub_ok=stub_ok,
                cont_need=int(cont_g.sum()), cont_hit=int((cont_g & cont_p).sum()),
                cont_spurious=int((~cont_g & cont_p).sum()),
                iou=float(inter / uni) if uni else 1.0, relaxed_f1=float(rf1), relaxed_p=float(rp), relaxed_r=float(rr), dangling=int(nc - len(touched)))


def aggregate(rows):
    tp = sum(r["tp"] for r in rows); fn = sum(r["fn"] for r in rows); fp = sum(r["fp"] for r in rows)
    rec = tp / max(tp + fn, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    ious = np.array([r["iou"] for r in rows]); ok = np.array([r["partition_ok"] for r in rows])
    rf = np.array([r["relaxed_f1"] for r in rows])
    faithful = (np.array([r["relaxed_p"] for r in rows]) >= 0.8) & (np.array([r["relaxed_r"] for r in rows]) >= 0.8)
    return dict(pair_recall=rec, pair_precision=prec, pair_f1=f1,
                partition_exact=float(ok.mean()),
                stub_correct=float(np.concatenate([r["stub_ok"] for r in rows]).mean()),
                continuity=sum(r["cont_hit"] for r in rows) / max(sum(r["cont_need"] for r in rows), 1),
                hole_iou=float(ious.mean()),
                dangling_per_map=float(np.mean([r["dangling"] for r in rows])),
                plausible_but_wrong=float((faithful & ~ok).mean()),
                relaxed_f1=float(np.mean(rf)),
                correct_and_faithful=float((faithful & ok).mean()),
                n=len(rows))


def composite(known, hole, pred_bin):
    return np.where(hole.astype(bool), pred_bin.astype(bool), known.astype(bool)).astype(np.uint8)
