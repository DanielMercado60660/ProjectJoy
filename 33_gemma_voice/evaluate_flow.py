"""Generate val-100 / test-100 with a flow-matching run (train_flow.py) and score it: Whisper judge + UTMOS.

  ../bin/gpu_run ../.venv/bin/python evaluate_flow.py --run runs/f1 --ckpt last --split val --tag val_t0.8

Writes runs/<run>/eval_<tag>/{lat.npz, wavs/*.wav, clips.jsonl, judge.json, utmos.json} and
results/eval_<run>_<tag>.json. Failures: hit the frame cap (no stop), or shorter than 0.4x the real clip (early stop).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_tokenizer  # noqa: E402
from tts_flow import build_model, generate, encode_text  # noqa: E402
from score import eval_sets, MOSHI_PY  # noqa: E402
from evaluate import frames_per_char  # noqa: E402

FRAME_RATE = 12.5


def load_run(run, ckpt):
    meta = json.load(open(os.path.join(run, ckpt, "meta.json")))
    a = meta["args"]
    model = build_model(a["seed"], a["n_spk"], a.get("progress", 0.0), a.get("prefix_lm", 0), a.get("head_w", 768),
                        a.get("head_depth", 6))
    model.load_weights(list(mx.load(os.path.join(run, ckpt, "model.safetensors")).items()), strict=False)
    model.backbone.compute_dtype = mx.bfloat16 if a.get("compute") == "bfloat16" else mx.float32
    mx.eval(model.parameters())
    return model, meta, np.array(a["lat_mean"], np.float32), np.array(a["lat_std"], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="last")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8, help="std of the flow's starting noise")
    ap.add_argument("--n_steps", type=int, default=16, help="Euler steps per frame")
    ap.add_argument("--stop_th", type=float, default=0.5)
    ap.add_argument("--cfg", type=float, default=1.0, help="classifier-free guidance scale (1 = off)")
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--judge", type=int, default=1)
    ap.add_argument("--mos", type=int, default=1, help="UTMOS via 27's mos.py (../.venv-sac, CPU)")
    a = ap.parse_args()
    run = os.path.join(HERE, a.run) if not os.path.isabs(a.run) else a.run
    out = os.path.join(run, f"eval_{a.tag}")
    os.makedirs(out, exist_ok=True)

    model, meta, mean, std = load_run(run, a.ckpt)
    tok = load_tokenizer()
    z = np.load(os.path.join(HERE, "data", "codes32.npz"))
    by_id = {str(c): i for i, c in enumerate(z["ids"])}
    fpc = frames_per_char()
    clips = eval_sets()[a.split][: a.n]
    names, lens, lats, rec = [], [], [], []
    for k, (cid, ref) in enumerate(clips):
        i = by_id[cid]
        text = str(z["norm"][i])
        ids = encode_text(tok, text, "chars")
        cap = int(a.cap * fpc * len(text)) + 10
        real = int(z["lens"][i])
        t0 = time.perf_counter()
        lat, info = generate(model, ids, cap, a.temp, a.n_steps, seed=a.seed * 100000 + k + 1, spk=0,
                             est_frames=int(round(fpc * len(text))), stop_th=a.stop_th, cfg=a.cfg)
        gen_s = time.perf_counter() - t0
        lat = lat * std + mean
        names.append(cid); lens.append(lat.shape[0]); lats.append(lat)
        F = info["frames"]
        rec.append({"id": cid, "text": ref, "real_frames": real, "frames": F, "len_ratio": F / real,
                    "hit_max": info["hit_max"], "early": F < 0.4 * real, "first_audio_s": info["first_audio_s"],
                    "gen_s": gen_s, "rtf": gen_s / max(F / FRAME_RATE, 1e-6)})
        if k % 20 == 0:
            print(f"{k}/{len(clips)} {cid} frames {F} / real {real} gen {gen_s:.2f}s", flush=True)
    lpath = os.path.join(out, "lat.npz")
    np.savez(lpath, names=np.array(names), lens=np.array(lens),
             lat=np.concatenate([l for l in lats if len(l)] or [np.zeros((0, 512), np.float32)]).astype(np.float32))
    wdir = os.path.join(out, "wavs")
    dec = subprocess.run([MOSHI_PY, os.path.join(HERE, "mimi_io.py"), "decode-latents", "--lat", lpath, "--out", wdir],
                         check=True, capture_output=True, text=True)
    dec_info = json.loads(dec.stdout.strip().splitlines()[-1])
    with open(os.path.join(out, "clips.jsonl"), "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    audio_s = sum(lens) / FRAME_RATE
    summ = {"run": a.run, "ckpt": a.ckpt, "step": meta["step"], "split": a.split, "n": len(rec), "temp": a.temp,
            "n_steps": a.n_steps, "stop_th": a.stop_th, "cfg": a.cfg, "seed": a.seed,
            "hit_max": sum(r["hit_max"] for r in rec), "early_stop": sum(r["early"] for r in rec),
            "len_ratio_median": round(float(np.median([r["len_ratio"] for r in rec])), 3),
            "gen_rtf": round(sum(r["gen_s"] for r in rec) / max(audio_s, 1e-6), 3),
            "decode_rtf": round(dec_info["decode_s"] / max(audio_s, 1e-6), 4),
            "first_audio_s_median": round(float(np.median([r["first_audio_s"] or 0 for r in rec])), 3)}
    lst = os.path.join(out, "judge_list.json")
    json.dump([{"id": r["id"], "wav": os.path.join(wdir, f"{r['id']}.wav"), "ref": r["text"]} for r in rec], open(lst, "w"))
    if a.judge:
        jout = os.path.join(out, "judge.json")
        subprocess.run([os.path.join(HERE, "..", ".venv-judge", "bin", "python"), os.path.join(HERE, "judge.py"),
                        "--list", lst, "--out", jout], check=True, capture_output=True)
        j = json.load(open(jout))
        summ.update(whisper_cer=j["cer"], whisper_wer=j["wer"])
    if a.mos:
        mout = os.path.join(out, "utmos.json")
        subprocess.run([os.path.join(HERE, "..", ".venv-sac", "bin", "python"),
                        os.path.join(HERE, "..", "27_pocket_voice_foundry", "mos.py"), "--list", lst, "--out", mout,
                        "--threads", "4"], check=True, capture_output=True)
        summ.update(utmos=json.load(open(mout))["mos"])
    print(json.dumps(summ), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(summ, open(os.path.join(HERE, "results", f"eval_{os.path.basename(a.run.rstrip('/'))}_{a.tag}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
