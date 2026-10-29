"""Synthesize every held-out clip with Griffin-Lim (iteration sweep) and the neural vocoder from the
same ground-truth log-mels; score against ground truth; write results/metrics.json and output WAVs."""
import argparse, json, os, subprocess, time
import numpy as np
import soundfile as sf
from metrics import mrstft, mel_l1, pitch_metrics, transient_metrics
from vocoders import gl_fn, neural_fn, CKPT
from mel import SR

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "cache", "test.npz"))
    ap.add_argument("--split", default=os.path.join(ROOT, "cache", "split.json"))
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--gl_iters", default="8,16,32,64,128")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips (0 = all)")
    a = ap.parse_args()
    z = np.load(a.data)
    ids = list(z["ids"])
    if a.limit:
        ids = ids[:a.limit]
    info = {d["id"]: d for d in json.load(open(a.split))["test"]}
    info.update({c: {"id": c, "transient": False} for c in ids if c not in info})
    methods = {f"gl{n}": gl_fn(int(n)) for n in a.gl_iters.split(",")}
    if os.path.exists(a.ckpt):
        methods["neural"] = neural_fn(a.ckpt)
    os.makedirs(os.path.join(a.out, "wavs"), exist_ok=True)
    rows = []
    for name, fn in methods.items():
        fn(z["mel_0"][:64])  # warm-up
        for i, cid in enumerate(ids):
            x, m = z[f"wav_{i}"], z[f"mel_{i}"]
            t = time.perf_counter(); y = fn(m); dt = time.perf_counter() - t
            y = y[:len(x)]
            sf.write(os.path.join(a.out, "wavs", f"{cid}_{name}.wav"), y, SR, subtype="FLOAT")
            r = {"method": name, "id": cid, "transient_clip": info[cid]["transient"], "seconds": len(x) / SR,
                 "synth_s": dt, "mrstft": mrstft(y, x), "mel_l1": mel_l1(y, x), **pitch_metrics(y, x)}
            r.update(transient_metrics(y, x))
            rows.append(r)
        print(name, "done", flush=True)
    load = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    summary = {}
    for name in methods:
        R = [r for r in rows if r["method"] == name]
        T = [r for r in R if r["transient_clip"] and r["n_onsets"] > 0]
        mean = lambda k, S: float(np.nanmean([r[k] for r in S]))
        summary[name] = {k: mean(k, R) for k in ["mrstft", "mel_l1", "f0_rmse_cents", "gpe_pct", "vde_pct", "periodicity_rmse"]}
        summary[name].update({f"transient_{k}": mean(k, T) for k in ["pre_onset_excess_db", "post_onset_excess_db", "hf_level_db", "rise_ms", "rise_ms_gt"]})
        summary[name]["transient_n_onsets"] = int(sum(r["n_onsets"] for r in T))
        summary[name]["rtf_under_load"] = float(sum(r["synth_s"] for r in R) / sum(r["seconds"] for r in R))
    out = {"n_clips": len(ids), "uptime": load, "summary": summary, "per_clip": rows}
    json.dump(out, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    hdr = ["method", "mrstft", "mel_l1", "f0_rmse_cents", "gpe_pct", "vde_pct", "periodicity_rmse",
           "transient_pre_onset_excess_db", "transient_post_onset_excess_db", "transient_hf_level_db", "transient_rise_ms", "rtf_under_load"]
    print(" | ".join(hdr))
    for name, s in summary.items():
        print(" | ".join([name] + [f"{s[k]:.3f}" for k in hdr[1:]]))
    print("rise_ms_gt", summary[list(summary)[0]]["transient_rise_ms_gt"], "onsets", summary[list(summary)[0]]["transient_n_onsets"])
    print(load)


if __name__ == "__main__":
    main()
