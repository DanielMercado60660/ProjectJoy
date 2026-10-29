"""Held-out evaluation (test split, chapters LJ045-LJ050; never trained or tuned on).

Rows: mel source in {gt (the real recording's mel), tts (learned), baseline} x output in
{mel (no vocoder: the acoustic stage alone), neural (28's vocoder, then re-analysed to log-mel), gl (Griffin-Lim 64)}.
gt x neural / gt x gl are the upper bounds: the vocoder's own damage with a perfect acoustic stage.
Metrics: CER/WER of the CTC recognizer (proxy for intelligibility), DTW log-mel L1 and DTW mel-cepstral
distance vs the real recording, total-duration ratio, per-character duration error vs MAS durations of the real
recording (from the trained model's prior, so model-derived pseudo ground truth).
Plus the hand-written hard-sentence set (hard_sentences.txt; no recordings exist, so recognizer-only).

  ../bin/gpu_run ../.venv/bin/python eval.py --n 100      # -> results/eval_summary.json, results/eval_clips.jsonl, results/wavs/
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
from scipy.fft import dct
from mel import log_mel
from mas import dtw_cost, dtw_cost_euclid
from ctc import load_recognizer, transcribe, cer, wer_words, ref_text, edit_distance
from synth import TTS, BaselineTTS, NeuralVocoder, gl, write_wav, CKPT_TTS
from text import encode, normalize

ROOT = os.path.dirname(os.path.abspath(__file__))
MCD_K = 10.0 / np.log(10.0) * np.sqrt(2.0)


def cepstrum(m, n=13):
    """Cepstrum of the natural-log mel spectrum: c_k = (2/N) sum_n logmel_n cos(pi k (n+0.5)/N), k = 1..n (c0 dropped)."""
    return dct(m.astype(np.float64), type=2, axis=-1)[:, 1:n + 1] / m.shape[-1]


def dist(a, ref):
    l1 = dtw_cost(a.astype(np.float64), ref.astype(np.float64))[0]
    mcd = MCD_K * dtw_cost_euclid(cepstrum(a), cepstrum(ref))
    return float(l1), float(mcd)


def semiglobal(word, hyp):
    """min edit distance of `word` against any substring of `hyp` (free start/end in hyp)."""
    prev = [0] * (len(hyp) + 1)
    for i in range(1, len(word) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (word[i - 1] != hyp[j - 1]))
        prev = cur
    return min(prev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--tts", default=CKPT_TTS)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--ctc", default=os.path.join(ROOT, "checkpoints", "ctc", "ctc.safetensors"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--gl_iters", type=int, default=64)
    ap.add_argument("--no_gl", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "wavs"), exist_ok=True)
    z = np.load(os.path.join(a.cache, "test.npz"))
    rng = np.random.default_rng(0)
    sel = np.sort(rng.choice(len(z["ids"]), size=min(a.n, len(z["ids"])), replace=False))
    M, offs = z["mels"], z["offs"]
    clips = [dict(id=str(z["ids"][i]), text=str(z["texts"][i]), mel=M[offs[i]:offs[i + 1]].astype(np.float32)) for i in sel]
    tts, bl, voc = TTS(a.tts), (BaselineTTS(a.baseline) if a.baseline else BaselineTTS()), NeuralVocoder()
    rec, rcfg = load_recognizer(a.ctc)
    from model import align
    outputs = ["mel", "neural"] + ([] if a.no_gl else ["gl"])
    rows = {(s, o): [] for s in ["gt", "tts", "baseline"] for o in outputs}
    per_clip = []
    t0 = time.time()
    for k, c in enumerate(clips):
        tok = encode(c["text"])
        # reference durations: MAS of the real recording under the trained model's prior
        x = (c["mel"] - tts.mean) / tts.std
        idx, dref = align(tts.model, mx.array(np.array(tok, np.int32)[None]), mx.array([len(tok)]),
                          mx.array(x[None]), mx.array([len(x)]))
        dref = dref[0]
        mels = {"gt": c["mel"]}
        mels["tts"], d_tts = tts.mel_from_tokens(tok)
        mels["baseline"], d_bl = bl.mel_from_tokens(tok)
        pc = {"id": c["id"], "text": c["text"], "frames_real": len(c["mel"])}
        for s, d in (("tts", d_tts), ("baseline", d_bl)):
            pc[f"{s}_len_ratio"] = len(mels[s]) / len(c["mel"])
            pc[f"{s}_dur_mae_frames"] = float(np.mean(np.abs(np.asarray(d, float) - dref)))
            pc[f"{s}_dur_corr"] = float(np.corrcoef(np.asarray(d, float), dref)[0, 1])
        for s, m in mels.items():
            for o in outputs:
                if o == "mel":
                    am = m
                else:
                    y = voc(m) if o == "neural" else gl(m, a.gl_iters)
                    write_wav(os.path.join(a.out, "wavs", f"{c['id']}_{s}_{o}.wav"), y)
                    am = log_mel(y)
                rows[(s, o)].append(am)
                if not (s == "gt" and o == "mel"):
                    pc[f"{s}_{o}_l1"], pc[f"{s}_{o}_mcd"] = dist(am, c["mel"])
        per_clip.append(pc)
        if k % 10 == 0:
            print(f"{k + 1}/{len(clips)} {time.time() - t0:.0f}s", flush=True)
    refs = [ref_text(c["text"]) for c in clips]
    for (s, o), ms in rows.items():
        hyps = transcribe(rec, rcfg, ms)
        for pc, h, r in zip(per_clip, hyps, refs):
            pc[f"{s}_{o}_hyp"] = h; pc[f"{s}_{o}_cer"] = cer(h, r); pc[f"{s}_{o}_wer"] = wer_words(h, r)
    with open(os.path.join(a.out, "eval_clips.jsonl"), "w") as f:
        for pc in per_clip:
            f.write(json.dumps(pc) + "\n")

    def boot(v, B=2000):
        v = np.asarray(v, float); r = np.random.default_rng(1)
        mu = [v[r.integers(0, len(v), len(v))].mean() for _ in range(B)]
        return [float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))]

    summ = {"n_clips": len(clips), "ids": [c["id"] for c in clips], "rows": {}}
    for (s, o) in rows:
        key = f"{s}/{o}"
        r = {}
        # corpus-level CER: total edits / total ref chars
        r["cer"] = float(sum(edit_distance(pc[f"{s}_{o}_hyp"], rf) for pc, rf in zip(per_clip, refs)) / sum(len(rf) for rf in refs))
        r["cer_ci95_clip_mean"] = boot([pc[f"{s}_{o}_cer"] for pc in per_clip])
        r["wer"] = float(np.mean([pc[f"{s}_{o}_wer"] for pc in per_clip]))
        if not (s == "gt" and o == "mel"):
            for m in ("l1", "mcd"):
                v = [pc[f"{s}_{o}_{m}"] for pc in per_clip]
                r[f"dtw_logmel_{m}" if m == "l1" else "dtw_mcd_db"] = float(np.mean(v))
                r[f"{m}_ci95"] = boot(v)
        summ["rows"][key] = r
    # ---- recognizer-bias probe: does the CTC proxy reward temporal smoothing of *real* mels? ----
    from scipy.ndimage import gaussian_filter1d
    probe = {}
    for sg in (1.0, 2.0, 3.0):
        hyps = transcribe(rec, rcfg, [gaussian_filter1d(c["mel"], sg, axis=0, mode="nearest") for c in clips])
        probe[f"gt_time_smoothed_sigma{sg:g}_frames/mel"] = float(sum(edit_distance(h, rf) for h, rf in zip(hyps, refs)) / sum(len(rf) for rf in refs))
    summ["recognizer_probe_cer"] = probe
    # ---- oversmoothing: per-band std over time of the predicted mel / of the real mel (global variance ratio) ----
    summ["gv_ratio"] = {s: float(np.mean([np.mean(m.std(0) / (c["mel"].std(0) + 1e-6)) for m, c in zip(rows[(s, "mel")], clips)]))
                        for s in ("tts", "baseline")}
    # ---- paired bootstrap of clip CER differences vs the real mel ----
    def pboot(d, B=2000):
        d = np.asarray(d, float); r = np.random.default_rng(2)
        mu = [d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]
        return {"mean": float(d.mean()), "ci95": [float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))]}
    summ["paired_cer_diff_vs_gt_mel"] = {f"{s}/{o}": pboot([pc[f"{s}_{o}_cer"] - pc["gt_mel_cer"] for pc in per_clip])
                                          for (s, o) in rows if not (s == "gt" and o == "mel")}
    summ["paired_cer_diff_tts_minus_baseline"] = {o: pboot([pc[f"tts_{o}_cer"] - pc[f"baseline_{o}_cer"] for pc in per_clip]) for o in outputs}
    for s in ("tts", "baseline"):
        lr = np.log([pc[f"{s}_len_ratio"] for pc in per_clip])
        summ[f"{s}_duration"] = {"len_ratio_geomean": float(np.exp(lr.mean())), "abs_log_len_ratio_mean": float(np.abs(lr).mean()),
                                 "abs_len_err_pct_mean": float(np.mean(np.abs(np.exp(lr) - 1)) * 100),
                                 "char_dur_mae_frames": float(np.mean([pc[f"{s}_dur_mae_frames"] for pc in per_clip])),
                                 "char_dur_mae_ms": float(np.mean([pc[f"{s}_dur_mae_frames"] for pc in per_clip]) * 16.0),
                                 "char_dur_corr_mean": float(np.nanmean([pc[f"{s}_dur_corr"] for pc in per_clip]))}

    # ---- hard sentences (no recordings: recognizer-only, per hard word) ----
    hard = []
    for line in open(os.path.join(ROOT, "hard_sentences.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sent, words = line.split("||")
        hard.append((sent.strip(), [w.strip() for w in words.split(",")]))
    import re
    train_words = set(re.findall(r"[a-z']+", " ".join(np.load(os.path.join(a.cache, "train.npz"))["texts"]).lower().replace("-", " ")))
    hres = {"sentences": [], "systems": {}}
    hm = {"tts": [], "baseline": []}
    for sent, words in hard:
        nt = normalize(sent)
        tok = encode(nt)
        hm["tts"].append(tts.mel_from_tokens(tok)[0]); hm["baseline"].append(bl.mel_from_tokens(tok)[0])
        hres["sentences"].append({"raw": sent, "normalized": nt, "hard_words": words,
                                  "oov_in_train_text": [w for w in nt.replace("-", " ").replace(",", "").replace(".", "").replace("?", "").split()
                                                        if w.strip("'\"") not in train_words]})
    for s in ("tts", "baseline"):
        for o in ("mel", "neural"):
            ms = hm[s] if o == "mel" else [log_mel(voc(m)) for m in hm[s]]
            hyps = transcribe(rec, rcfg, ms)
            fails, tot, sent_c = 0, 0, []
            per = []
            for (sent, words), h, info in zip(hard, hyps, hres["sentences"]):
                ref = ref_text(info["normalized"]); sent_c.append(cer(h, ref))
                wr = []
                for w in words:
                    wn = ref_text(normalize(w))
                    dd = semiglobal(wn, h) / max(1, len(wn))
                    ok = dd <= 1 / 3
                    fails += (not ok); tot += 1
                    wr.append({"word": w, "norm": wn, "char_err": round(dd, 3), "ok": bool(ok)})
                per.append({"hyp": h, "cer": round(sent_c[-1], 3), "words": wr})
            hres["systems"][f"{s}/{o}"] = {"hard_word_failures": fails, "hard_words": tot, "sentence_cer_mean": float(np.mean(sent_c)), "per_sentence": per}
    summ["hard_sentences"] = {k: {kk: vv for kk, vv in v.items() if kk != "per_sentence"} for k, v in hres["systems"].items()}
    json.dump(hres, open(os.path.join(a.out, "hard_sentences_eval.json"), "w"), indent=1)
    summ["wall_s"] = time.time() - t0
    summ["tts_ckpt"] = json.load(open(a.tts.replace(".safetensors", ".json")))["step"]
    json.dump(summ, open(os.path.join(a.out, "eval_summary.json"), "w"), indent=1)
    for k, v in summ["rows"].items():
        print(f"{k:16s} CER {v['cer']*100:5.1f}%  WER {v['wer']*100:5.1f}%  " +
              (f"L1 {v['dtw_logmel_l1']:.3f}  MCD {v['dtw_mcd_db']:.2f} dB" if "dtw_mcd_db" in v else ""))
    for s in ("tts", "baseline"):
        print(s, summ[f"{s}_duration"])
    print("probe", summ["recognizer_probe_cer"], "gv", summ["gv_ratio"])
    print("paired vs gt", {k: round(v["mean"], 4) for k, v in summ["paired_cer_diff_vs_gt_mel"].items()})
    for k, v in summ["hard_sentences"].items():
        print("hard", k, v)


if __name__ == "__main__":
    main()
