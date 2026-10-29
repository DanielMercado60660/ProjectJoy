"""Reviewer diagnostic: separate "acoustics" from "timing" in the TTS-vs-baseline gap.

Same 100 test clips as eval.py (ids read from results/eval_summary.json), same recognizer, same 28 vocoder.
Rows (outputs: mel with no vocoder, and 28's neural vocoder; GL skipped):
  baseline / own durations       = the shelf baseline (reproduces eval.py's numbers)
  baseline / TTS durations       = baseline state-mean tables, but per-character frame counts from the learned
                                   TTS duration predictor (with its val-fitted pace and rounding). A hybrid, not a
                                   cheap baseline: it borrows the learned model's timing. Uses no test information.
  (each borrowed timing in three modes, see baseline_with_durs: per character, per character with >= 1 frame per
   state, and per word keeping the baseline's own within-word proportions)
  baseline / oracle durations    = per-character frame counts from MAS of the *real* test mel under the trained
                                   model (the same pseudo-truth eval.py's duration table uses). Oracle diagnostic.
  tts / own durations            = shipped TTS (reproduces eval.py)
  tts / oracle durations         = TTS decoder fed the oracle MAS durations. Oracle diagnostic.
Splitting a character's D frames over the baseline's K states: largest-remainder allocation proportional to the
chosen template's mean state durations, so a 1-frame character gets only its longer state (a state may get 0 frames;
the baseline's own synthesis instead forces >= 1 frame per state).
  ../.venv/bin/python timing_split.py        # -> results/timing_split.json (+ per-clip CERs)
"""
import json, os
import numpy as np
import mlx.core as mx
from mel import log_mel
from ctc import load_recognizer, transcribe, cer, edit_distance, ref_text
from synth import TTS, BaselineTTS, NeuralVocoder
from text import encode, SYMBOLS
from model import align

ROOT = os.path.dirname(os.path.abspath(__file__))


def split_frames(D, w):
    """D frames over len(w) states, proportional to w (largest remainder)."""
    w = np.asarray(w, float); w = w / max(w.sum(), 1e-9)
    raw = D * w; base = np.floor(raw).astype(int)
    for k in np.argsort(-(raw - base))[: D - base.sum()]:
        base[k] += 1
    return base


def lookup(bl, tok, n):
    N = len(tok)
    c = lambda j: int(tok[j]) if 0 <= j < N else -1
    ctx7, ctx5, ctx3 = tuple(c(n + j) for j in range(-3, 4)), tuple(c(n + j) for j in range(-2, 3)), (c(n - 1), c(n), c(n + 1))
    return bl.tri.get(ctx7) or bl.tri.get(ctx5) or bl.tri.get(ctx3) or bl.mono.get(int(tok[n])) or bl.mono[SYMBOLS.index(" ")]


def word_groups(tok):
    """[start, end) token spans: each word plus the space/punctuation run after it."""
    sp = SYMBOLS.index(" ")
    g, s = [], 0
    for n in range(1, len(tok) + 1):
        if n == len(tok) or (tok[n] != sp and tok[n - 1] == sp):
            g.append((s, n)); s = n
    return g


def baseline_with_durs(bl, tok, D, mode="char"):
    """Baseline tables (same context back-off as Baseline.synth) with borrowed timing D (per character).
    mode "char": each character gets exactly D[n] frames, split over its K states (a state may get 0).
    mode "char_min": as "char" but every state gets >= 1 frame (as in the baseline's own synthesis; lengthens).
    mode "word": each word (+ following spaces/punctuation) gets sum(D) frames, spread over its states in proportion
                 to the baseline's own state durations (>= 1 per state when the word has enough frames), so the
                 baseline keeps its own within-word segmentation and borrows only word-level timing."""
    N = len(tok)
    ent = [lookup(bl, tok, n) for n in range(N)]
    alloc = []
    if mode == "word":
        for s, e in word_groups(tok):
            w = np.concatenate([ent[n][1] for n in range(s, e)])
            Dw = int(np.sum(D[s:e])); K = len(w)
            f = (np.ones(K, int) + split_frames(Dw - K, w)) if Dw >= K else split_frames(Dw, w)
            alloc += [f[(n - s) * bl.K:(n - s + 1) * bl.K] for n in range(s, e)]
    else:
        for n in range(N):
            f = split_frames(int(D[n]), ent[n][1])
            alloc.append(np.maximum(f, 1) if mode == "char_min" else f)
    frames = []
    for n in range(N):
        mu = ent[n][0]
        for k, f in enumerate(alloc[n]):
            if f > 0:
                frames.append(np.repeat(mu[k][None], f, 0))
    m = np.concatenate(frames, 0)
    if bl.sigma > 0:
        from scipy.ndimage import gaussian_filter1d
        m = gaussian_filter1d(m, bl.sigma, axis=0, mode="nearest")
    return (m * bl.std + bl.mean).astype(np.float32)


def tts_with_durs(tts, tok, D):
    t = mx.array(np.asarray(tok, np.int32)[None])
    h, _, _ = tts.model.encode(t, mx.array([t.shape[1]]))
    idx = np.repeat(np.arange(len(D)), np.asarray(D, np.int64))[None]
    mel = tts.model.decode(h, mx.array(idx, dtype=mx.int32), mx.array([idx.shape[1]]))
    mx.eval(mel)
    return np.array(mel[0]) * tts.std + tts.mean


def clips_for(split, ids=None, n=100):
    z = np.load(os.path.join(ROOT, "cache", f"{split}.npz"))
    pos = {str(c): i for i, c in enumerate(z["ids"])}
    sel = [pos[i] for i in ids] if ids is not None else list(range(n))
    return [dict(id=str(z["ids"][i]), text=str(z["texts"][i]), mel=z["mels"][z["offs"][i]:z["offs"][i + 1]].astype(np.float32)) for i in sel]


def run(clips, tts, bl, voc, rec, rcfg, outputs):
    mels = {k: [] for k in ("baseline/own", "baseline/tts_dur", "baseline/tts_dur_min", "baseline/tts_dur_word",
                            "baseline/oracle_dur", "baseline/oracle_dur_min", "baseline/oracle_dur_word", "tts/own", "tts/oracle_dur")}
    for c in clips:
        tok = encode(c["text"])
        x = (c["mel"] - tts.mean) / tts.std
        _, dref = align(tts.model, mx.array(np.array(tok, np.int32)[None]), mx.array([len(tok)]), mx.array(x[None]), mx.array([len(x)]))
        dref = dref[0].astype(int)
        m_tts, d_tts = tts.mel_from_tokens(tok)
        mels["tts/own"].append(m_tts)
        mels["tts/oracle_dur"].append(tts_with_durs(tts, tok, dref))
        mels["baseline/own"].append(bl.bl.synth(np.asarray(tok))[0])
        for name, D in (("tts_dur", d_tts), ("oracle_dur", dref)):
            for suf, mode in (("", "char"), ("_min", "char_min"), ("_word", "word")):
                mels[f"baseline/{name}{suf}"].append(baseline_with_durs(bl.bl, np.asarray(tok), np.asarray(D), mode))
    refs = [ref_text(c["text"]) for c in clips]
    res = {}
    for k, ms in mels.items():
        for o in outputs:
            am = ms if o == "mel" else [log_mel(voc(m)) for m in ms]
            hyps = transcribe(rec, rcfg, am)
            res[f"{k}/{o}"] = {"cer": float(sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / sum(len(r) for r in refs)),
                               "clip_cer": [float(cer(h, r)) for h, r in zip(hyps, refs)],
                               "len_ratio_geomean": float(np.exp(np.mean([np.log(len(m) / len(c["mel"])) for m, c in zip(ms, clips)])))}
    return res


def pboot(d, B=2000):
    d = np.asarray(d, float); r = np.random.default_rng(2)
    mu = [d[r.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(mu, 2.5)), float(np.percentile(mu, 97.5))]}


def main():
    summ = json.load(open(os.path.join(ROOT, "results", "eval_summary.json")))
    per = [json.loads(l) for l in open(os.path.join(ROOT, "results", "eval_clips.jsonl"))]
    tts, bl, voc = TTS(), BaselineTTS(), NeuralVocoder()
    rec, rcfg = load_recognizer(os.path.join(ROOT, "checkpoints", "ctc", "ctc.safetensors"))
    test = clips_for("test", summ["ids"])
    assert [c["id"] for c in test] == [p["id"] for p in per]
    out = {"note": __doc__.strip().splitlines()[0], "n_clips": len(test), "test": run(test, tts, bl, voc, rec, rcfg, ["mel", "neural"])}
    T = out["test"]
    # reproduction check against eval.py's per-clip CERs (same clips, recognizer, checkpoints)
    out["repro_max_abs_clip_cer_diff"] = {k: float(np.max(np.abs(np.array(T[f"{k}/own/{o}"]["clip_cer"]) - np.array([p[f"{k}_{o}_cer"] for p in per]))))
                                          for k in ("tts", "baseline") for o in ("mel",)}
    gt = np.array([p["gt_mel_cer"] for p in per])
    T_KEYS = sorted({k.rsplit("/", 1)[0] for k in T})
    out["paired_vs_tts_own"] = {}
    out["paired_vs_gt_mel"] = {}
    for o in ("mel", "neural"):
        ref = np.array(T[f"tts/own/{o}"]["clip_cer"])
        for k in [m for m in T_KEYS if m != "tts/own"]:
            out["paired_vs_tts_own"][f"{k}/{o}"] = pboot(np.array(T[f"{k}/{o}"]["clip_cer"]) - ref)
        for k in T_KEYS:
            out["paired_vs_gt_mel"][f"{k}/{o}"] = pboot(np.array(T[f"{k}/{o}"]["clip_cer"]) - gt)
    out["paired_vs_baseline_own"] = {f"{k}/{o}": pboot(np.array(T[f"{k}/{o}"]["clip_cer"]) - np.array(T[f"baseline/own/{o}"]["clip_cer"]))
                                     for k in T_KEYS if k.startswith("baseline/") and k != "baseline/own" for o in ("mel", "neural")}
    # val sanity for the hybrid (first 100 val clips, mel only): nothing is selected here
    out["val_first100_mel"] = {k: v["cer"] for k, v in run(clips_for("val", n=100), tts, bl, voc, rec, rcfg, ["mel"]).items()}
    json.dump(out, open(os.path.join(ROOT, "results", "timing_split.json"), "w"), indent=1)
    for k, v in T.items():
        print(f"{k:28s} CER {100 * v['cer']:5.1f}%  len {v['len_ratio_geomean']:.3f}")
    print("repro", out["repro_max_abs_clip_cer_diff"])
    print("val", {k: round(100 * v, 1) for k, v in out["val_first100_mel"].items()})


if __name__ == "__main__":
    main()
