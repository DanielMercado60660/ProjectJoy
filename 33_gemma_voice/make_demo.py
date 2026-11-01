"""Self-contained listening page: media/demo.html (audio embedded as MP3 data URIs, works offline).

For the first N held-out test sentences: the real recording, the Mimi 4-codebook round trip (the ceiling), Gemma TTS,
the same model trained from random init, and project 27's FastSpeech TTS (its saved neural-vocoder output).
Each row shows what the recognizer proxy heard. Plus the step-500 "good failure".

  ../.venv/bin/python make_demo.py --n 6
"""
import argparse, base64, html, json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1", "wavs")
P27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "results")


def mp3_uri(path):
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", path, "-ac", "1", "-b:a", "48k", f.name], check=True)
        return "data:audio/mpeg;base64," + base64.b64encode(open(f.name, "rb").read()).decode()


def clips_by_id(path):
    return {r["id"]: r for r in map(json.loads, open(path))} if os.path.exists(path) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--main", default="runs/r3")
    ap.add_argument("--scratch", default="runs/scratch")
    ap.add_argument("--tag", default="test")
    a = ap.parse_args()
    test = [json.loads(l) for l in open(os.path.join(P27, "eval_clips.jsonl"))][: a.n]
    g = clips_by_id(os.path.join(HERE, a.main, f"eval_{a.tag}", "clips.jsonl"))
    s = clips_by_id(os.path.join(HERE, a.scratch, f"eval_{a.tag}", "clips.jsonl"))
    summ = {k: json.load(open(os.path.join(HERE, "results", f"eval_{os.path.basename(r)}_{a.tag}.json")))
            for k, r in (("gemma", a.main), ("scratch", a.scratch))
            if os.path.exists(os.path.join(HERE, "results", f"eval_{os.path.basename(r)}_{a.tag}.json"))}
    rows = []
    for c in test:
        cid = c["id"]
        srcs = [("Real recording", os.path.join(LJ, cid + ".wav"), None),
                ("Mimi round trip, 4 codebooks (ceiling)", os.path.join(HERE, "runs", "ceiling_low", "test", "nq4", cid + ".wav"), None),
                ("Gemma TTS (this project)", os.path.join(HERE, a.main, f"eval_{a.tag}", "wavs", cid + ".wav"), g.get(cid)),
                ("Same model, random init", os.path.join(HERE, a.scratch, f"eval_{a.tag}", "wavs", cid + ".wav"), s.get(cid)),
                ("Project 27 FastSpeech + vocoder", os.path.join(P27, "wavs", f"{cid}_tts_neural.wav"), None)]
        cells = []
        for label, p, rec in srcs:
            if not os.path.exists(p):
                cells.append(f"<div class='cell'><div class='lab'>{label}</div><div class='na'>not available</div></div>")
                continue
            heard = f"<div class='heard'>heard: “{html.escape(rec['hyp'])}” · CER {100 * rec['cer']:.0f}%</div>" if rec else ""
            cells.append(f"<div class='cell'><div class='lab'>{label}</div><audio controls preload='none' src='{mp3_uri(p)}'></audio>{heard}</div>")
        rows.append(f"<section><h3>“{html.escape(c['text'])}”</h3><div class='grid'>{''.join(cells)}</div></section>")
    fail = os.path.join(HERE, "media", "failure_step500", "LJ041-0005_step500_babble.wav")
    fail_html = (f"<section><h3>Good failure: “he concluded that school had nothing to offer him.”</h3>"
                 f"<p>First run, step 500 (subword text, 16 codebooks). Right voice, right length, wrong words: "
                 f"the recognizer heard “the ers sego obas ftb”.</p><audio controls preload='none' src='{mp3_uri(fail)}'></audio></section>"
                 if os.path.exists(fail) else "")
    nums = " · ".join(f"{k}: test CER {v['cer']}%" for k, v in summ.items())
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma Voice</title><style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a66;--card:#fff;--line:#e6e3dc;--acc:#2f6f5e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--mut:#9d9b95;--card:#1f1f1d;--line:#33322f;--acc:#7cc3ae}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,system-ui,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:24px 16px 64px}} h1{{margin:0 0 4px}} .sub{{color:var(--mut);margin:0 0 24px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 16px}}
h3{{margin:0 0 10px;font-weight:600;font-size:1rem}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}}
.lab{{font-size:.85rem;color:var(--acc);margin-bottom:4px}} audio{{width:100%}} .heard{{font-size:.8rem;color:var(--mut);margin-top:4px}}
.na{{font-size:.85rem;color:var(--mut)}}
</style></head><body><main><h1>Gemma Voice</h1>
<p class="sub">Gemma 3 270M fine-tuned to speak Mimi codec tokens, from 19 h of LJSpeech on a laptop. Held-out test sentences
(chapters LJ045–050). "Heard" = what project 27's character recognizer transcribed. {html.escape(nums)}</p>
{''.join(rows)}{fail_html}</main></body></html>"""
    os.makedirs(os.path.join(HERE, "media"), exist_ok=True)
    open(os.path.join(HERE, "media", "demo.html"), "w").write(page)
    print("wrote media/demo.html", round(len(page) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
