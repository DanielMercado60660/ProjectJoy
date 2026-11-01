"""Where is the intelligibility lost? Teacher-forced resynthesis on val clips.

For each clip, run the model on the REAL delayed grid (as in training) and take each step's argmax for every
codebook (each prediction sees the real history). Variants:
  tf_all      : all 16 codebooks from the model's one-step predictions
  tf_cb0_real : real codebook 0, model codebooks 1-15
  tf_cb0_only : model codebook 0, real codebooks 1-15
  tf_sample   : all 16 codebooks SAMPLED (temp 0.8, top-k 50) from the one-step predictions (argmax is the
                wrong decode for high-entropy acoustic heads)
Decode with Mimi and score CER with the recognizer proxy. Compare with real codes (the nq16 ceiling) and
free-running generation (evaluate.py).

  ../bin/gpu_run ../.venv/bin/python diag_tf.py --run runs/r2 --ckpt best     # -> results/diag_tf_<run>_<ckpt>.json
"""
import argparse, json, os, subprocess, sys
import numpy as np
import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import load_run  # noqa: E402
from gemma import load_tokenizer  # noqa: E402
from tts import encode_text, delay_grid, undelay, make_batch, PAD  # noqa: E402
from score import eval_sets, corpus_cer, wav24_to_mel, MOSHI_PY  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args()
    run = os.path.join(HERE, a.run)
    model, meta = load_run(run, a.ckpt)
    nq, mode = meta["args"]["nq"], meta["args"].get("text", "subword")
    tok = load_tokenizer()
    z = np.load(os.path.join(HERE, "data", "codes32.npz"))
    by_id = {str(c): i for i, c in enumerate(z["ids"])}
    clips = eval_sets()["val"][: a.n]
    variants = {k: [] for k in ("real", "tf_all", "tf_cb0_real", "tf_cb0_only", "tf_sample")}
    key = mx.random.key(0)
    acc = []
    for cid, _ in clips:
        i = by_id[cid]
        real = z["codes"][:nq, z["offs"][i]:z["offs"][i + 1]].astype(np.int64)
        F = real.shape[1]
        t = encode_text(tok, str(z["norm"][i]), mode)
        g = delay_grid(real)
        text, tlen, grid_in, tgt = make_batch([(t, g)], nq)
        h = model.backbone.hidden(model.inputs(text, tlen, grid_in, None))
        lg = model.logits(h).astype(mx.float32)[0, len(t):len(t) + len(g)]          # (S, NQ, V)
        pred_grid = np.array(mx.argmax(lg[..., :2048], -1))                           # real codes only
        pred = undelay(pred_grid, F)
        sl = lg[..., :2048]
        kth = mx.sort(sl, axis=-1)[..., -50][..., None]
        key, sub = mx.random.split(key)
        samp = undelay(np.array(mx.random.categorical(mx.where(sl < kth, -1e9, sl) / 0.8, key=sub)), F)
        acc.append((pred == real).mean(1))
        variants["real"].append(real)
        variants["tf_all"].append(pred)
        variants["tf_cb0_real"].append(np.concatenate([real[:1], pred[1:]]))
        variants["tf_cb0_only"].append(np.concatenate([pred[:1], real[1:]]))
        variants["tf_sample"].append(samp)
    out = {"run": a.run, "ckpt": a.ckpt, "step": meta["step"], "n": len(clips),
           "tf_top1_acc_per_cb": [round(float(x), 3) for x in np.mean(acc, 0)]}
    texts = [r for _, r in clips]
    for name, codes in variants.items():
        d = os.path.join(run, f"diag_tf_{a.ckpt}", name)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "codes.npz")
        np.savez(p, names=np.array([c for c, _ in clips]), lens=np.array([c.shape[1] for c in codes]),
                 codes=np.concatenate(codes, axis=1))
        subprocess.run([MOSHI_PY, os.path.join(HERE, "mimi_io.py"), "decode", "--codes", p, "--nq", str(nq),
                        "--out", os.path.join(d, "wavs")], check=True, capture_output=True)
        sc = corpus_cer([wav24_to_mel(os.path.join(d, "wavs", f"{c}.wav")) for c, _ in clips], texts)
        out[f"cer_{name}"] = round(100 * sc["cer"], 2)
        print(name, out[f"cer_{name}"], flush=True)
    print("tf top-1 acc per cb:", out["tf_top1_acc_per_cb"][:6], "...")
    json.dump(out, open(os.path.join(HERE, "results", f"diag_tf_{os.path.basename(a.run)}_{a.ckpt}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
