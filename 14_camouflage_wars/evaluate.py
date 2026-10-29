"""Evaluate the trained U-Net and the classical baseline on test_id (seen texture families, new
scenes) and test_ood (three texture families never seen in training). Writes results/eval.json
and results/eval.md."""
import argparse, json, os, pickle, subprocess, time
import numpy as np
import model as M
import baseline as B
from metrics import summarize, stat_scale, creature_records, stat_distance
from train import predict


def load(name, data):
    d = np.load(os.path.join(data, f"{name}.npz"))
    return d["X"], d["Y"], pickle.load(open(os.path.join(data, f"{name}_meta.pkl"), "rb"))


def fmt_table(rows, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        s += "| " + " | ".join(r) + " |\n"
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--ckpt", default="checkpoints/unet")
    ap.add_argument("--out", default="results")
    ap.add_argument("--n", type=int, default=0, help="limit scenes per split (0 = all)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    choice = json.load(open("baseline_choice.json"))["best"]
    # Difficulty scale and quartile edges come from the TRAIN split (fixed, model-independent).
    tr_meta = pickle.load(open(os.path.join(a.data, "train_meta.pkl"), "rb"))
    scale = stat_scale([tr_meta])
    tr_d = [float(np.sqrt(((np.array(c["stat_dist_raw"]) / scale) ** 2).mean()))
            for m in tr_meta for c in m["creatures"].values()]
    edges = np.quantile(tr_d, [0.25, 0.5, 0.75]).tolist()

    net = M.load(a.ckpt)
    res = dict(baseline=choice, quartile_edges_from_train=edges, splits={})
    load_avg = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    for split in ("test_id", "test_ood"):
        X, Y, metas = load(split, a.data)
        if a.n:
            X, Y, metas = X[:a.n], Y[:a.n], metas[:a.n]
        t = time.time()
        P_net = predict(net, X) > 0.5
        t_net = (time.time() - t) / len(X) * 1000
        t = time.time()
        P_base = B.run(X, choice["method"], choice["sigma"])
        t_base = (time.time() - t) / len(X) * 1000
        res["splits"][split] = {}
        for name, P, ms in (("unet", P_net, t_net), ("baseline", P_base, t_base)):
            s, recs, s_iou = summarize(P, Y, metas, scale, edges)
            s["ms_per_scene_under_load"] = ms
            res["splits"][split][name] = s
            np.save(os.path.join(a.out, f"scene_iou_{split}_{name}.npy"), s_iou)
        print(split, "unet", round(res["splits"][split]["unet"]["scene_iou"], 4),
              "baseline", round(res["splits"][split]["baseline"]["scene_iou"], 4), flush=True)
    res["uptime_during_eval"] = load_avg
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=1)

    md = f"Baseline = `{choice['method']}` (sigma {choice['sigma']}), chosen on val. Load: {load_avg}\n\n"
    rows = []
    for split in ("test_id", "test_ood"):
        for name in ("unet", "baseline"):
            s = res["splits"][split][name]
            rows.append([split, name, f"{s['scene_iou']:.3f}", f"{s['pixel_iou']:.3f}", f"{s['det_rate']:.3f}",
                         f"{s['inst_iou']:.3f}", f"{s['det_rate_isolated']:.3f}", f"{s['det_rate_coverage']:.3f}", f"{s['fp_per_scene']:.2f}", f"{s['ms_per_scene_under_load']:.1f}"])
    md += fmt_table(rows, ["split", "method", "scene IoU", "pixel IoU", "det@IoU0.5", "inst IoU",
                           "det@0.5 isolated only", "coverage≥0.5", "FP comps/scene", "ms/scene (under load)"])
    for key, title in (("by_delta", "camouflage knob δ (generation parameter)"),
                       ("by_stat_quartile", "measured texture-statistics distance (quartiles from train)"),
                       ("by_skin_family", "creature skin family")):
        md += (f"\n**By {title}.** det(iso) = instance IoU ≥ 0.5 among creatures not touching another "
               f"creature; cov = ≥ 50% of the creature's pixels predicted (all creatures)\n\n")
        rows = []
        for split in ("test_id", "test_ood"):
            u, b = res["splits"][split]["unet"][key], res["splits"][split]["baseline"][key]
            for k in u:
                rows.append([split, k, f"{u[k]['n']} ({u[k]['n_isolated']})", f"{u[k]['det_rate_isolated']:.3f}",
                             f"{b[k]['det_rate_isolated']:.3f}", f"{u[k]['det_rate_coverage']:.3f}",
                             f"{b[k]['det_rate_coverage']:.3f}", f"{u[k]['inst_iou']:.3f}", f"{b[k]['inst_iou']:.3f}"])
        md += fmt_table(rows, ["split", "bin", "n (isolated)", "U-Net det(iso)", "baseline det(iso)",
                               "U-Net cov", "baseline cov", "U-Net inst IoU", "baseline inst IoU"])
    open(os.path.join(a.out, "eval.md"), "w").write(md)
    print(md)
