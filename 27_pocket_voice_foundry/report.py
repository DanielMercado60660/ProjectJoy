"""Print the RESULTS.md tables straight from the result JSONs (so no number is hand-copied).

  ../.venv/bin/python report.py            # -> stdout and results/tables.md
"""
import json, os

ROOT = os.path.dirname(os.path.abspath(__file__))
import sys
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
NAMES = {"gt": "real mel (upper bound)", "tts": "learned TTS", "baseline": "baseline (template concat)"}
OUTS = {"mel": "mel, no vocoder", "neural": "28's neural vocoder", "gl": "Griffin-Lim 64"}


def main():
    out = []
    s = json.load(open(os.path.join(R, "eval_summary.json")))
    out.append(f"### Held-out test, {s['n_clips']} clips (TTS checkpoint step {s['tts_ckpt']})\n")
    out.append("| mel source | output | CER % (CTC proxy) [95% CI, clip mean] | WER % | DTW log-mel L1 | MCD-like dB |")
    out.append("|---|---|---|---|---|---|")
    for key, r in s["rows"].items():
        src, o = key.split("/")
        ci = r["cer_ci95_clip_mean"]
        l1 = f"{r['dtw_logmel_l1']:.3f}" if "dtw_logmel_l1" in r else "0 (reference)"
        mcd = f"{r['dtw_mcd_db']:.2f}" if "dtw_mcd_db" in r else "0 (reference)"
        out.append(f"| {NAMES[src]} | {OUTS[o]} | {100 * r['cer']:.1f} [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}] | "
                   f"{100 * r['wer']:.1f} | {l1} | {mcd} |")
    out.append("\n### Durations on the same test clips (reference = MAS of the real mel under the trained model: model-derived pseudo-truth)\n")
    out.append("| system | total length ratio (geo-mean) | mean abs length error % | per-char duration MAE ms | per-char duration corr |")
    out.append("|---|---|---|---|---|")
    for k in ("tts", "baseline"):
        d = s[f"{k}_duration"]
        out.append(f"| {NAMES[k]} | {d['len_ratio_geomean']:.3f} | {d['abs_len_err_pct_mean']:.1f} | {d['char_dur_mae_ms']:.1f} | {d['char_dur_corr_mean']:.2f} |")
    out.append("\n### Hard-sentence set (hard_sentences.txt, recognizer-only, no recordings)\n")
    out.append("| system | hard words failed (char err > 1/3) | sentence CER mean % |")
    out.append("|---|---|---|")
    for k, v in s["hard_sentences"].items():
        src, o = k.split("/")
        out.append(f"| {NAMES[src]} / {OUTS[o]} | {v['hard_word_failures']} / {v['hard_words']} | {100 * v['sentence_cer_mean']:.1f} |")
    r1 = os.path.join(R, "run1", "eval_summary.json")
    if os.path.exists(r1):
        import numpy as np
        s1 = json.load(open(r1))
        c1 = [json.loads(l) for l in open(os.path.join(R, "run1", "eval_clips.jsonl"))]
        c2 = [json.loads(l) for l in open(os.path.join(R, "eval_clips.jsonl"))]
        assert [c["id"] for c in c1] == [c["id"] for c in c2]
        vc = json.load(open(os.path.join(R, "val_compare.json")))["candidates"] if os.path.exists(os.path.join(R, "val_compare.json")) else {}
        out.append(f"\n### Run 1 (step {s1['tts_ckpt']}) vs shipped (step {s['tts_ckpt']}), same 100 test clips\n")
        out.append("| output | run-1 CER % | shipped CER % | paired diff, points [95% CI] | run-1 DTW L1 | shipped DTW L1 | run-1 MCD-like | shipped MCD-like |")
        out.append("|---|---|---|---|---|---|---|---|")
        rng = np.random.default_rng(3)
        for o in ("mel", "neural", "gl"):
            k = f"tts/{o}"
            d = np.array([b[f"tts_{o}_cer"] - a[f"tts_{o}_cer"] for a, b in zip(c1, c2)])
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            a, b = s1["rows"][k], s["rows"][k]
            out.append(f"| {OUTS[o]} | {100 * a['cer']:.1f} | {100 * b['cer']:.1f} | {100 * d.mean():+.2f} [{100 * np.percentile(bs, 2.5):+.2f}, {100 * np.percentile(bs, 97.5):+.2f}] | "
                       f"{a['dtw_logmel_l1']:.3f} | {b['dtw_logmel_l1']:.3f} | {a['dtw_mcd_db']:.2f} | {b['dtw_mcd_db']:.2f} |")
        out.append(f"\nGlobal-variance ratio (TTS mel / real): run 1 {s1['gv_ratio']['tts']:.3f}, shipped {s['gv_ratio']['tts']:.3f}. "
                   f"Duration: run-1 length ratio {s1['tts_duration']['len_ratio_geomean']:.3f}, shipped {s['tts_duration']['len_ratio_geomean']:.3f}.")
        for pth, v in vc.items():
            out.append(f"- val (100 clips) `{pth}` step {v['step']}: CER {100 * v['val_cer']:.2f} %, DTW L1 {v['val_dtw_logmel_l1']:.4f}, length ratio {v['val_len_ratio']:.3f}")
    hp = os.path.join(R, "hard_sentences_eval.json")
    if os.path.exists(hp):
        import re
        import numpy as np
        h = json.load(open(hp))
        txt = " ".join(np.load(os.path.join(ROOT, "cache", "train.npz"))["texts"]).lower().replace("-", " ")
        cnt = lambda w: sum(len(re.findall(r"\b" + re.escape(p) + r"\b", txt)) for p in [w.replace("-", " ")])
        out.append("\n### Hard sentences one by one (28's neural vocoder; failed hard words with their count in the training text)\n")
        out.append("| # | sentence | TTS CER % | TTS failed words | baseline CER % | baseline failed words |")
        out.append("|---|---|---|---|---|---|")
        for i, info in enumerate(h["sentences"]):
            row = []
            for k in ("tts/neural", "baseline/neural"):
                ps = h["systems"][k]["per_sentence"][i]
                fl = [f"{w['norm']} ({cnt(w['norm'])})" for w in ps["words"] if not w["ok"]]
                row += [f"{100 * ps['cer']:.0f}", ", ".join(fl) or "none"]
            out.append(f"| {i + 1} | {info['raw']} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")
    bp = os.path.join(R, "bench.json")
    if os.path.exists(bp):
        b = json.load(open(bp))
        out.append(f"\n### Speed, {b['label']}\n")
        out.append(f"`uptime` before: `{b['uptime_before']}`  \n`uptime` after: `{b['uptime_after']}`\n")
        out.append("| pipeline | acoustic ms (median) | first audio ms (median) | full utterance ms (median) | RTF median | RTF max |")
        out.append("|---|---|---|---|---|---|")
        for k, r in b["results"].items():
            out.append(f"| {k} | {1e3 * r['acoustic_s_median']:.1f} | {1e3 * r['first_audio_s_median']:.1f} | {1e3 * r['full_s_median']:.1f} | "
                       f"{r['rtf_median']:.4f} | {r['rtf_max']:.4f} |")
    tp = os.path.join(R, "baseline_tuning.json")
    if os.path.exists(tp):
        t = json.load(open(tp))
        out.append(f"\n### Baseline tuning on val ({t['n_val']} val clips, {sum(r['n_configs'] for r in t['rounds'])} configs over {len(t['rounds'])} rounds; {t['selection']})\n")
        for nm in ("selected", "selected_by_dtw_l1"):
            r = t[nm]
            out.append(f"- {nm}: K={r['K']}, context={r['context']}, dur_scale={r['dur_scale']}, sigma={r['sigma']}: "
                       f"val CER {100 * r.get('val_cer', float('nan')):.1f} %, val DTW L1 {r['val_dtw_logmel_l1']:.3f}, "
                       f"val length ratio {r['val_dur_ratio']:.3f}")
    sp = os.path.join(R, "timing_split.json")
    if os.path.exists(sp):  # reviewer diagnostic (timing_split.py)
        t = json.load(open(sp))
        T, g, b = t["test"], t["paired_vs_gt_mel"], t["paired_vs_baseline_own"]
        lab = {"baseline/own": "baseline, own durations (shelf baseline)", "baseline/tts_dur": "baseline + TTS-predicted durations, per char",
               "baseline/tts_dur_min": "baseline + TTS durations, per char, >=1 frame/state", "baseline/tts_dur_word": "baseline + TTS durations, per word",
               "baseline/oracle_dur": "baseline + oracle (real-mel MAS) durations, per char", "baseline/oracle_dur_min": "baseline + oracle durations, per char, >=1 frame/state",
               "baseline/oracle_dur_word": "baseline + oracle durations, per word", "tts/own": "TTS, own durations (shipped)", "tts/oracle_dur": "TTS decoder + oracle durations"}
        pf = lambda v: f"{100 * v['mean']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
        out.append(f"\n### Timing vs acoustics (reviewer diagnostic, timing_split.py; same {t['n_clips']} test clips; paired differences on mel, no vocoder)\n")
        out.append("| system | CER % | CER % with 28's vocoder | minus real mel, points [95% CI] | minus shelf baseline, points [95% CI] | length ratio | val CER % (first 100 val clips) |")
        out.append("|---|---|---|---|---|---|---|")
        for k in lab:
            out.append(f"| {lab[k]} | {100 * T[k + '/mel']['cer']:.1f} | {100 * T[k + '/neural']['cer']:.1f} | {pf(g[k + '/mel'])} | "
                       f"{pf(b[k + '/mel']) if k + '/mel' in b else 'n/a'} | {T[k + '/mel']['len_ratio_geomean']:.3f} | {100 * t['val_first100_mel'][k + '/mel']:.1f} |")
    txt = "\n".join(out) + "\n"
    open(os.path.join(R, "tables.md"), "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
