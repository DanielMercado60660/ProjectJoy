"""Generate speech for val-100 or test-100 and score it with 27's recognizer proxy.

  ../bin/gpu_run ../.venv/bin/python evaluate.py --run runs/main --split val --tag val_t0.8
  ../bin/gpu_run ../.venv/bin/python evaluate.py --run runs/main --split test --tag test   # once per shipped ckpt

Writes runs/<run>/eval_<tag>/{codes.npz, wavs/*.wav, clips.jsonl} and results/eval_<run>_<tag>.json.
Failures: hit the frame cap (no end token), or shorter than 0.4x the real clip (early stop).
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gemma import load_tokenizer  # noqa: E402
from tts import build_model, generate, encode_text  # noqa: E402
from train import load_into  # noqa: E402
from score import eval_sets, corpus_cer, wav24_to_mel, MOSHI_PY  # noqa: E402

FRAME_RATE = 12.5


def load_run(run, ckpt):
    meta = json.load(open(os.path.join(run, ckpt, "meta.json")))
    args = meta["args"]
    model = build_model(args["nq"], args["init"], args["seed"],
                        mx.float32 if args["dtype"] == "float32" else mx.bfloat16, args.get("n_spk", 1),
                        args.get("progress", 0.0), args.get("prefix_lm", 0))
    load_into(model, None, os.path.join(run, ckpt))
    model.backbone.compute_dtype = mx.bfloat16 if args.get("compute") == "bfloat16" else mx.float32
    mx.eval(model.parameters())
    return model, meta


def frames_per_char():
    """Median Mimi frames per character of normalized text on the train split (sets the frame cap)."""
    z = np.load(os.path.join(HERE, "data", "codes32.npz"))
    tr = z["splits"] == "train"
    r = z["lens"][tr] / np.array([len(t) for t in z["norm"][tr]])
    return float(np.median(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--cap", type=float, default=2.0, help="frame cap = cap x median frames/char x chars + 10")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--judge", type=int, default=1, help="also score the wavs with the Whisper judge (../.venv-judge)")
    ap.add_argument("--best_of", type=int, default=1, help="N takes per sentence; keep the one 27's CTC recognizer "
                    "transcribes best against the target text (a different ASR than the Whisper judge)")
    a = ap.parse_args()
    run = os.path.join(HERE, a.run) if not os.path.isabs(a.run) else a.run
    out = os.path.join(run, f"eval_{a.tag}")
    os.makedirs(os.path.join(out, "wavs"), exist_ok=True)

    model, meta = load_run(run, a.ckpt)
    nq = meta["args"]["nq"]
    tok = load_tokenizer()
    z = np.load(os.path.join(HERE, "data", "codes32.npz"))
    by_id = {str(c): i for i, c in enumerate(z["ids"])}
    fpc = frames_per_char()
    clips = eval_sets()[a.split][: a.n]

    import shutil
    N = max(1, a.best_of)
    cand_names, cand_lens, cand_codes, cand_info, rec = [], [], [], [], []
    for k, (cid, ref) in enumerate(clips):
        i = by_id[cid]
        text = str(z["norm"][i])
        ids = encode_text(tok, text, meta["args"].get("text", "subword"))
        cap = int(a.cap * fpc * len(text)) + 10
        real = int(z["lens"][i])
        gen_s, infos = 0.0, []
        for n in range(N):
            t0 = time.perf_counter()
            est = int(round(fpc * len(text))) + nq - 1                     # estimated total steps (progress models)
            codes, info = generate(model, ids, cap, a.temp, a.top_k, seed=a.seed * 100000 + k + 1000003 * n, est_steps=est)
            gen_s += time.perf_counter() - t0
            cand_names.append(f"{cid}__{n}"); cand_lens.append(codes.shape[1]); cand_codes.append(codes); infos.append(info)
        cand_info.append(infos)
        rec.append({"id": cid, "text": ref, "real_frames": real, "gen_s": gen_s})
        if k % 20 == 0:
            print(f"{k}/{len(clips)} {cid} frames {[x['frames'] for x in infos]} / real {real} gen {gen_s:.2f}s", flush=True)
    cpath = os.path.join(out, "codes.npz")
    np.savez(cpath, names=np.array(cand_names), lens=np.array(cand_lens),
             codes=np.concatenate([c if c.shape[1] else np.zeros((nq, 0), np.int64) for c in cand_codes], axis=1))
    cdir = os.path.join(out, "wavs_cand")
    dec = subprocess.run([MOSHI_PY, os.path.join(HERE, "mimi_io.py"), "decode", "--codes", cpath, "--nq", str(nq),
                          "--out", cdir], check=True, capture_output=True, text=True)
    dec_info = json.loads(dec.stdout.strip().splitlines()[-1])
    mels = [wav24_to_mel(os.path.join(cdir, f"{c}.wav")) if l > 0 else np.zeros((1, 80), np.float32) - 11.5
            for c, l in zip(cand_names, cand_lens)]
    sc = corpus_cer(mels, [r["text"] for r in rec for _ in range(N)])
    lens = []
    for k, r in enumerate(rec):
        cers = sc["clip_cer"][k * N:(k + 1) * N]
        n = int(np.argmin(cers))                  # ties -> first take
        info = cand_info[k][n]
        shutil.copy(os.path.join(cdir, f"{r['id']}__{n}.wav"), os.path.join(out, "wavs", f"{r['id']}.wav"))
        r.update(pick=n, cand_cer=[round(c, 4) for c in cers], hyp=sc["hyps"][k * N + n], cer=cers[n],
                 frames=info["frames"], len_ratio=info["frames"] / r["real_frames"], hit_max=info["hit_max"],
                 early=info["frames"] < 0.4 * r["real_frames"], first_audio_s=info.get("first_audio_s"),
                 rtf=r["gen_s"] / max(info["frames"] / FRAME_RATE, 1e-6))
        lens.append(info["frames"])
    sc = {"cer": sum(r["cer"] * len(r["text"]) for r in rec) / sum(len(r["text"]) for r in rec)}  # approx corpus CER of picks
    with open(os.path.join(out, "clips.jsonl"), "w") as f:
        for r in rec:
            f.write(json.dumps(r) + "\n")
    audio_s = sum(lens) / FRAME_RATE
    summ = {"run": a.run, "ckpt": a.ckpt, "step": meta["step"], "split": a.split, "n": len(rec),
            "temp": a.temp, "top_k": a.top_k, "cap": a.cap, "seed": a.seed, "best_of": N,
            "cer": round(100 * sc["cer"], 2),
            "hit_max": sum(r["hit_max"] for r in rec), "early_stop": sum(r["early"] for r in rec),
            "len_ratio_median": round(float(np.median([r["len_ratio"] for r in rec])), 3),
            "gen_rtf": round(sum(r["gen_s"] for r in rec) / audio_s, 3),     # includes all N takes
            "decode_rtf": round(dec_info["decode_s"] / audio_s, 4),
            "first_audio_s_median": (round(float(np.median([r["first_audio_s"] for r in rec])), 3)
                                     if rec[0]["first_audio_s"] is not None else None)}
    if a.judge:
        lst = os.path.join(out, "judge_list.json")
        json.dump([{"id": r["id"], "wav": os.path.join(out, "wavs", f"{r['id']}.wav"), "ref": r["text"]} for r in rec],
                  open(lst, "w"))
        jout = os.path.join(out, "judge.json")
        subprocess.run([os.path.join(HERE, "..", ".venv-judge", "bin", "python"), os.path.join(HERE, "judge.py"),
                        "--list", lst, "--out", jout], check=True, capture_output=True)
        j = json.load(open(jout))
        summ.update(whisper_cer=j["cer"], whisper_wer=j["wer"])
    print(json.dumps(summ), flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(summ, open(os.path.join(HERE, "results", f"eval_{os.path.basename(a.run.rstrip('/'))}_{a.tag}.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
