"""Generate val-100 / test-100 with the continuous Gemma TTS and score it like project 27 did.

  ../bin/gpu_run ../.venv/bin/python evaluate_mel.py --run runs/mel1 --split val --tag val100
Two scores, matching 27's RESULTS rows: the predicted mel fed straight to 27's recognizer ("mel, no vocoder"), and the
mel rendered by 28's vocoder then re-analysed ("neural vocoder"). Wavs (16 kHz) are kept for listening.
Failures: hit the step cap (no stop), or shorter than 0.4x the real clip.
"""
import argparse, json, math, os, sys, time
import numpy as np
import mlx.core as mx
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_tokenizer  # noqa: E402
from tts import encode_text  # noqa: E402
from tts_mel import build_model, generate, mel_stats  # noqa: E402
from train import load_into  # noqa: E402
from score import eval_sets, corpus_cer  # noqa: E402
from mel import log_mel  # noqa: E402
from vocoder_model import load_vocoder  # noqa: E402

CACHE27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "cache")


def load_run(run, ckpt):
    meta = json.load(open(os.path.join(run, ckpt, "meta.json")))
    a = meta["args"]
    model = build_model(a["init"], a["seed"], a["K"], a["p_drop"])
    load_into(model, None, os.path.join(run, ckpt))
    model.backbone.compute_dtype = mx.bfloat16 if a.get("compute") == "bfloat16" else mx.float32
    model.eval()
    mx.eval(model.parameters())
    return model, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--infer_dropout", type=int, default=1)
    ap.add_argument("--stop", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    run = os.path.join(HERE, a.run) if not os.path.isabs(a.run) else a.run
    out = os.path.join(run, f"eval_{a.tag}")
    os.makedirs(os.path.join(out, "wavs"), exist_ok=True)
    model, meta = load_run(run, a.ckpt)
    model.infer_dropout = bool(a.infer_dropout)
    K, mode = meta["args"]["K"], meta["args"].get("text", "chars")
    mean, std = mel_stats()
    tok = load_tokenizer()
    voc, _ = load_vocoder(os.path.join(HERE, "checkpoints", "vocoder28", "vocoder.safetensors"))
    codes = np.load(os.path.join(HERE, "data", "codes32.npz"))
    norm = {str(i): str(t) for i, t in zip(codes["ids"], codes["norm"])}
    z = np.load(os.path.join(CACHE27, "train.npz"))
    fpc = float(np.median(z["lens"] / np.array([len(t) for t in z["texts"]])))
    real = {}
    for sp in ("val", "test"):
        zz = np.load(os.path.join(CACHE27, f"{sp}.npz"))
        real.update({str(c): int(l) for c, l in zip(zz["ids"], zz["lens"])})
    clips = eval_sets()[a.split][: a.n]
    rec, mels, wav_mels = [], [], []
    for k, (cid, ref) in enumerate(clips):
        text = norm[cid]
        ids = encode_text(tok, text, mode)
        max_steps = int(math.ceil((a.cap * fpc * len(text) + 20) / K))
        t0 = time.perf_counter()
        m_norm, info = generate(model, ids, max_steps, seed=a.seed * 100000 + k, stop_threshold=a.stop)
        gen_s = time.perf_counter() - t0
        m = m_norm * std + mean
        t1 = time.perf_counter()
        y = voc(mx.array(m[None]))
        mx.eval(y)
        y = np.array(y[0], np.float32)
        voc_s = time.perf_counter() - t1
        sf.write(os.path.join(out, "wavs", f"{cid}.wav"), y, 16000)
        mels.append(m)
        wav_mels.append(log_mel(y))
        audio_s = len(m) / 62.5
        rec.append({"id": cid, "text": ref, "frames": len(m), "real_frames": real.get(cid), "len_ratio": len(m) / real[cid],
                    "hit_max": info["hit_max"], "early": len(m) < 0.4 * real[cid], "gen_s": gen_s, "voc_s": voc_s,
                    "rtf": (gen_s + voc_s) / max(audio_s, 1e-6), "first_chunk_s": info["first_chunk_s"]})
        if k % 20 == 0:
            print(f"{k}/{len(clips)} {cid} frames {len(m)} / real {real[cid]} gen {gen_s:.2f}s", flush=True)
    texts = [r["text"] for r in rec]
    s_mel = corpus_cer(mels, texts)
    s_voc = corpus_cer(wav_mels, texts)
    for r, hm, cm, hv, cv in zip(rec, s_mel["hyps"], s_mel["clip_cer"], s_voc["hyps"], s_voc["clip_cer"]):
        r.update(hyp_mel=hm, cer_mel=cm, hyp=hv, cer=cv)
    with open(os.path.join(out, "clips.jsonl"), "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    summ = {"run": a.run, "ckpt": a.ckpt, "step": meta["step"], "split": a.split, "n": len(rec),
            "infer_dropout": a.infer_dropout, "stop": a.stop, "seed": a.seed,
            "cer_mel": round(100 * s_mel["cer"], 2), "cer_vocoder": round(100 * s_voc["cer"], 2),
            "hit_max": sum(r["hit_max"] for r in rec), "early_stop": sum(r["early"] for r in rec),
            "len_ratio_median": round(float(np.median([r["len_ratio"] for r in rec])), 3),
            "rtf": round(sum(r["gen_s"] + r["voc_s"] for r in rec) / sum(r["frames"] / 62.5 for r in rec), 3),
            "first_chunk_s_median": round(float(np.median([r["first_chunk_s"] for r in rec])), 3)}
    print(json.dumps(summ), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(summ, open(os.path.join(HERE, "results", f"evalmel_{os.path.basename(a.run.rstrip('/'))}_{a.tag}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
