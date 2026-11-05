"""R8 Gate C, part 2: the restated R6 gates G1-G4 at 224 px for one trained run, and the gate table for the
checkpoint-vs-scratch pair (results/r8/lewm_ckpt/protocol.json).

    ./r8_job.sh lewm-eval ../.venv/bin/python r8_rep/rep_eval.py eval runs/r8_lewm_ckpt [--ckpt final]
        -> results/r8/lewm_ckpt/eval_<run>.json
    ../.venv/bin/python r8_rep/rep_eval.py gates runs/r8_lewm_ckpt runs/r8_lewm_scratch runs/r8_lewm_rand
        -> results/r8/lewm_ckpt/gates.json, gates.md

G1 held-out set_lamp frames (<= 20,000, rng 2026), eval mode: share of dims with std in [0.5, 2], effective rank,
   SIGReg (training log) at step 1000 vs the last logged step.
G2 4,000 windows (rng 2026) each of held-out set_lamp and dev-room episodes: R = mean||zhat3 - z3||^2 / mean||z2 - z3||^2.
G3 the first 2,000 of those held-out windows: true action block 2 vs 15 blocks of windows from other episodes (rng 7).
G4 the R8 probe set: linear probes (rep_probes.linear_probe) on the run's 192-d latent (projector output) and,
   reported, on [CLS ; mean patch]; the plate bar is P2-A' s0's AUC on the same frames (bake-off row p2a).
Reported: rollout ratio by horizon 1-5 (2,000 windows, rng 11), held-out go_and_return windows (R, top-1).
"""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import rep_data as D  # noqa: E402
import rep_ports as P  # noqa: E402
from rep_probes import linear_probe, load_items  # noqa: E402
from rep_train import load_for_eval  # noqa: E402

RES = ROOT / "results" / "r8" / "lewm_ckpt"
NEG = 15


def embed(model, frames, bs=64):
    out = []
    for i in range(0, len(frames), bs):
        z = model.encode(mx.array(np.ascontiguousarray(frames[i:i + bs])))
        mx.eval(z)
        out.append(np.array(z))
    return np.concatenate(out)


def window_latents(model, win, ids):
    F, A, e = win.get(ids)
    B = len(ids)
    z = embed(model, F.reshape(-1, *F.shape[2:])).reshape(B, win.T, -1)
    return z, A, e


def predict_last(model, z, A):
    zh = model.predict(mx.array(z[:, :3]), mx.array(A[:, :3]))[:, 2]
    mx.eval(zh)
    return np.array(zh)


def g2(model, win, ids, bs=64):
    se, cl = [], []
    for i in range(0, len(ids), bs):
        z, A, _ = window_latents(model, win, ids[i:i + bs])
        zh = predict_last(model, z, A)
        se.append(((zh - z[:, 3]) ** 2).sum(-1))
        cl.append(((z[:, 2] - z[:, 3]) ** 2).sum(-1))
    se, cl = np.concatenate(se), np.concatenate(cl)
    return {"R": float(se.mean() / cl.mean()), "pred_mse": float(se.mean()), "copy_last_mse": float(cl.mean()),
            "n": int(len(ids))}


def g3(model, win, ids, rng, bs=32):
    ok = n = 0
    for i in range(0, len(ids), bs):
        z, A, e = window_latents(model, win, ids[i:i + bs])
        B = len(z)
        neg_ids = rng.integers(0, len(win), (B, NEG * 2))
        _, An, en = win.get(neg_ids.reshape(-1), frames=False)
        An, en = An.reshape(B, NEG * 2, win.T, -1), en.reshape(B, NEG * 2)
        cands = np.empty((B, NEG + 1, A.shape[-1]), np.float32)
        cands[:, 0] = A[:, 2]
        for b in range(B):
            other = [j for j in range(NEG * 2) if en[b, j] != e[b]][:NEG]
            assert len(other) == NEG
            cands[b, 1:] = An[b, other, 2]
        zz = np.repeat(z, NEG + 1, axis=0)
        AA = np.repeat(A, NEG + 1, axis=0)
        AA[:, 2] = cands.reshape(-1, A.shape[-1])
        zh = predict_last(model, zz, AA).reshape(B, NEG + 1, -1)
        d = ((zh - z[:, None, 3]) ** 2).sum(-1)
        ok += int((d.argmin(1) == 0).sum())
        n += B
    return {"top1": ok / n, "chance": 1 / (NEG + 1), "n": n}


def horizon(model, E, rng, H=5, n=2000, bs=64):
    win = D.Windows(E, T=3 + H, fs=5)
    ids = rng.choice(len(win), min(n, len(win)), replace=False)
    se, cl = np.zeros(H), np.zeros(H)
    for i in range(0, len(ids), bs):
        z, A, _ = window_latents(model, win, ids[i:i + bs])
        ctx = z[:, :3].copy()
        for h in range(H):
            zh = predict_last(model, np.concatenate([ctx, ctx[:, -1:]], 1), A[:, h:h + 4])
            se[h] += ((zh - z[:, 3 + h]) ** 2).sum()
            cl[h] += ((z[:, 2] - z[:, 3 + h]) ** 2).sum()
            ctx = np.concatenate([ctx[:, 1:], zh[:, None]], 1)
    return {"ratio_by_step": [float(x) for x in se / cl], "mse_by_step": [float(x) for x in se / len(ids)],
            "copy_last_mse_by_step": [float(x) for x in cl / len(ids)], "n": int(len(ids))}


def g1(model, run, E, rng, n=20000):
    idx = D.frame_index(E)
    sel = np.sort(rng.choice(len(idx), min(n, len(idx)), replace=False))
    z = []
    for i in range(0, len(sel), 256):
        F = np.stack([E.frames(idx[j][0])[idx[j][1]] for j in sel[i:i + 256]])
        z.append(embed(model, F))
    z = np.concatenate(z)
    sd = z.std(0)
    ev = np.clip(np.linalg.eigvalsh(np.cov(z.T)), 0, None)
    p = ev / ev.sum()
    erank = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    log = [json.loads(line) for line in open(run / "log.jsonl")]
    s1000 = next((r["sigreg"] for r in log if r["step"] == 1000), None)
    return {"frac_std_in_0.5_2": float(((sd >= 0.5) & (sd <= 2)).mean()), "std_median": float(np.median(sd)),
            "std_min": float(sd.min()), "std_max": float(sd.max()), "effective_rank": erank,
            "sigreg_step1000": s1000, "sigreg_last": log[-1]["sigreg"] if log else None,
            "last_step": log[-1]["step"] if log else None, "n_frames": len(z)}


def g4(model):
    it = load_items()
    f224 = np.load(ROOT / "data" / "r8_rep" / "probeset" / "frames224.npy", mmap_mode="r")
    proj, glob = [], []
    for i in range(0, len(f224), 64):
        x = mx.array(np.ascontiguousarray(f224[i:i + 64]))
        tok = model.encoder.tokens(P.normalize_u8(x))
        pr = model.projector(tok[:, 0])
        gl = mx.concatenate([tok[:, 0], tok[:, 1:].mean(1)], -1)
        mx.eval(pr, gl)
        proj.append(np.array(pr))
        glob.append(np.array(gl))
    proj, glob = np.concatenate(proj), np.concatenate(glob)
    out = {}
    for t, name in ((0, "lamp"), (1, "plate"), (2, "bearing")):
        s = it["task"] == t
        fr, y, tr = it["frame"][s], it["y"][s], it["cfg"][s] < 270
        if t < 2:
            y = (y > 0.5).astype(np.float32)
        out[name] = {"linear_proj": linear_probe(proj[fr], y, tr, regress=(t == 2)),
                     "linear_glob": linear_probe(glob[fr], y, tr, regress=(t == 2))}
    return out


def evaluate(run, ckpt="final", n=4000, n3=2000, quick=False):
    model, cfg, step = load_for_eval(run, ckpt)
    d = json.loads((run / "data.json").read_text())
    norm = (np.array(d["act_mu"], np.float32), np.array(d["act_sd"], np.float32))
    H = D.Episodes(d["heldout"], norm)
    dev = D.Episodes(D.episode_list("dev"), norm)
    gr = D.episode_list("heldout", task="go_and_return")
    rng = np.random.default_rng(2026)
    out = {"run": str(run.relative_to(ROOT)), "ckpt": ckpt, "step": step, "init": cfg["init"],
           "heldout_episodes": len(d["heldout"]), "dev_episodes": len(dev.eps), "train_episodes": len(d["train"])}
    if cfg["init"] != "random_encoder" and not quick:
        out["G1"] = g1(model, run, H, rng)
    hw = D.Windows(H)
    dw = D.Windows(dev) if len(dev.eps) else None
    ids_h = rng.choice(len(hw), min(n, len(hw)), replace=False)
    out["G2_heldout"] = g2(model, hw, ids_h)
    if dw is not None:
        ids_d = rng.choice(len(dw), min(n, len(dw)), replace=False)
        out["G2_dev"] = g2(model, dw, ids_d)
        out["G3_dev"] = g3(model, dw, ids_d[:n3], np.random.default_rng(7))
    out["G3_heldout"] = g3(model, hw, ids_h[:n3], np.random.default_rng(7))
    if not quick:
        out["horizon_heldout"] = horizon(model, H, np.random.default_rng(11))
        if gr:
            G = D.Episodes(gr, norm)
            gw = D.Windows(G)
            gid = np.random.default_rng(5).choice(len(gw), min(n, len(gw)), replace=False)
            out["go_and_return_heldout"] = {"G2": g2(model, gw, gid), "G3": g3(model, gw, gid[:n3], np.random.default_rng(7))}
        if cfg["init"] != "random_encoder":
            out["G4"] = g4(model)
    return out


def gates(ck, sc, ra):
    """Gate table for the pair (+ control) from their eval files."""
    E = {k: json.loads((RES / f"eval_{Path(r).name}.json").read_text()) for k, r in (("ckpt", ck), ("scratch", sc))}
    ctrl = json.loads((RES / f"eval_{Path(ra).name}.json").read_text()) if ra and (RES / f"eval_{Path(ra).name}.json").exists() else None
    bake = json.loads((ROOT / "results" / "r8" / "probes" / "table.json").read_text())
    p2a_auc = bake["p2a"]["plate"]["linear"]["test_auc"]
    c_top1 = ctrl["G3_heldout"]["top1"] if ctrl else None
    out = {"p2a_plate_auc_bar": p2a_auc, "control_top1": c_top1}
    for arm, e in E.items():
        g = {}
        g["G1"] = {"frac_std": e["G1"]["frac_std_in_0.5_2"], "rank": e["G1"]["effective_rank"],
                   "sigreg_1000": e["G1"]["sigreg_step1000"], "sigreg_last": e["G1"]["sigreg_last"]}
        g["G1"]["pass"] = bool(g["G1"]["frac_std"] >= 0.9 and g["G1"]["rank"] >= 100 and
                               g["G1"]["sigreg_1000"] is not None and g["G1"]["sigreg_last"] < g["G1"]["sigreg_1000"])
        rd = e.get("G2_dev", {}).get("R")
        g["G2"] = {"R_heldout": e["G2_heldout"]["R"], "R_dev": rd,
                   "pass": bool(e["G2_heldout"]["R"] <= 0.70 and rd is not None and rd <= 0.85)}
        t1 = e["G3_heldout"]["top1"]
        g["G3"] = {"top1": t1, "control_top1": c_top1, "pass_absolute": bool(t1 >= 0.5),
                   "pass": bool(t1 >= 0.5 and c_top1 is not None and t1 >= 3 * c_top1)}
        la, pa = e["G4"]["lamp"]["linear_proj"]["test_acc"], e["G4"]["plate"]["linear_proj"]["test_auc"]
        g["G4"] = {"lamp_acc": la, "plate_auc": pa, "plate_bar": p2a_auc, "pass": bool(la >= 0.9 and pa >= p2a_auc)}
        g["all_pass"] = all(g[k]["pass"] for k in ("G1", "G2", "G3", "G4"))
        out[arm] = g
    c, s = out["ckpt"], out["scratch"]
    better = {"G2_R_lower_by_0.03": c["G2"]["R_heldout"] <= s["G2"]["R_heldout"] - 0.03,
              "G3_top1_higher_by_0.05": c["G3"]["top1"] >= s["G3"]["top1"] + 0.05,
              "G4_lamp_higher_by_0.03": c["G4"]["lamp_acc"] >= s["G4"]["lamp_acc"] + 0.03,
              "G1_rank_higher_by_10": c["G1"]["rank"] >= s["G1"]["rank"] + 10}
    out["ckpt_clearly_beats_scratch"] = {"criteria": better, "count": int(sum(better.values())),
                                         "verdict": bool(sum(better.values()) >= 3)}
    nm = (c["G1"]["frac_std"] >= 0.81 and c["G1"]["rank"] >= 90 and c["G2"]["R_heldout"] <= 0.77 and
          (c["G2"]["R_dev"] or 9) <= 0.935 and c["G3"]["top1"] >= 0.45 and c["G4"]["lamp_acc"] >= 0.81)
    out["ckpt_narrow_miss_or_pass"] = bool(c["all_pass"] or nm)
    out["propose_second_seed"] = bool(c["all_pass"] or (nm and out["ckpt_clearly_beats_scratch"]["verdict"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["eval", "gates"])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--quick", action="store_true", help="smoke test: G2/G3 on 64 windows only")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    if a.mode == "eval":
        run = ROOT / a.runs[0]
        out = evaluate(run, a.ckpt, n=64 if a.quick else 4000, n3=32 if a.quick else 2000, quick=a.quick)
        name = f"eval_{run.name}" + ("" if a.ckpt == "final" else f"_{a.ckpt}") + ("_quick" if a.quick else "")
        (RES / f"{name}.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
    else:
        out = gates(*a.runs[:3]) if len(a.runs) >= 3 else gates(a.runs[0], a.runs[1], None)
        (RES / "gates.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
