"""Segmentation / detection metrics with breakdown by camouflage difficulty."""
import numpy as np
from scipy.ndimage import label, binary_dilation

DELTA_BINS = [("same δ<0.25", 0, 0.25), ("same 0.25–0.5", 0.25, 0.5),
              ("same 0.5–0.75", 0.5, 0.75), ("same 0.75–1", 0.75, 1.01)]


def creature_records(pred, Y, metas):
    """One record per creature: instance IoU vs the union of predicted components touching it."""
    recs, fps = [], []
    for i in range(len(pred)):
        comp, nc = label(pred[i])
        fp = 0
        for c in range(1, nc + 1):
            cm = comp == c
            if cm.sum() >= 10 and (Y[i][cm] > 0).mean() < 0.1:
                fp += 1
        fps.append(fp)
        for k, m in metas[i]["creatures"].items():
            C = Y[i] == k
            touching = np.unique(comp[C])
            touching = touching[touching > 0]
            P = np.isin(comp, touching)
            inter, union = (P & C).sum(), (P | C).sum()
            isolated = not ((Y[i] > 0) & (Y[i] != k) & binary_dilation(C, iterations=2)).any()
            r = dict(scene=i, iou=inter / union, isolated=isolated, coverage=(pred[i] & C).sum() / C.sum(),
                     kind=m["kind"], delta=m["delta"], family=m["family"], area=int(C.sum()),
                     stat_raw=np.array(m["stat_dist_raw"], np.float32))
            recs.append(r)
    return recs, np.array(fps)


def scene_iou(pred, Y):
    g = Y > 0
    inter, union = (pred & g).sum((1, 2)), (pred | g).sum((1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def stat_distance(recs, scale):
    """Standardised Euclidean distance between creature-skin and background texture descriptors."""
    for r in recs:
        r["stat_dist"] = float(np.sqrt((((r["stat_raw"]) / scale) ** 2).mean()))
    return recs


def _agg(sel):
    iso = [r for r in sel if r["isolated"]]
    return dict(n=len(sel), n_isolated=len(iso),
                det_rate=float(np.mean([r["iou"] >= 0.5 for r in sel])),
                det_rate_isolated=float(np.mean([r["iou"] >= 0.5 for r in iso])) if iso else float("nan"),
                det_rate_coverage=float(np.mean([r["coverage"] >= 0.5 for r in sel])),
                inst_iou=float(np.mean([r["iou"] for r in sel])))


def summarize(pred, Y, metas, scale, quartile_edges=None):
    recs, fps = creature_records(pred, Y, metas)
    recs = stat_distance(recs, scale)
    s_iou = scene_iou(pred, Y)
    g = Y > 0
    out = dict(scene_iou=float(s_iou.mean()),
               pixel_iou=float((pred & g).sum() / (pred | g).sum()),
               det_rate=float(np.mean([r["iou"] >= 0.5 for r in recs])),
               inst_iou=float(np.mean([r["iou"] for r in recs])),
               det_rate_isolated=float(np.mean([r["iou"] >= 0.5 for r in recs if r["isolated"]])),
               frac_isolated=float(np.mean([r["isolated"] for r in recs])),
               det_rate_coverage=float(np.mean([r["coverage"] >= 0.5 for r in recs])),
               fp_per_scene=float(fps.mean()), n_creatures=len(recs), n_scenes=len(pred))
    by = {}

    def agg(name, sel):
        if sel:
            by[name] = _agg(sel)
    agg("other family", [r for r in recs if r["kind"] == "other"])
    for name, lo, hi in DELTA_BINS:
        agg(name, [r for r in recs if r["kind"] == "same" and lo <= r["delta"] < hi])
    out["by_delta"] = by
    d = np.array([r["stat_dist"] for r in recs])
    if quartile_edges is None:
        quartile_edges = np.quantile(d, [0.25, 0.5, 0.75]).tolist()
    e = [-np.inf] + list(quartile_edges) + [np.inf]
    byq = {}
    for q in range(4):
        sel = [r for r in recs if e[q] <= r["stat_dist"] < e[q + 1]]
        name = ["Q1 (closest match)", "Q2", "Q3", "Q4 (least match)"][q]
        if sel:
            byq[name] = _agg(sel)
    out["by_stat_quartile"] = byq
    out["quartile_edges"] = quartile_edges
    byf = {}
    for f in sorted({r["family"] for r in recs}):
        sel = [r for r in recs if r["family"] == f]
        byf[f] = _agg(sel)
    out["by_skin_family"] = byf
    return out, recs, s_iou


def stat_scale(metas_list):
    """Per-feature scale (std of raw skin-minus-background differences) from a reference split."""
    raw = np.array([c["stat_dist_raw"] for metas in metas_list for m in metas for c in m["creatures"].values()])
    return raw.std(0) + 1e-6
