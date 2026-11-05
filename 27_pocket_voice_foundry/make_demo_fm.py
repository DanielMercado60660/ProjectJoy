"""Follow-up listening demo: media/followup.html (offline, blind, per-viewer shuffle) + a few WAVs + a mel strip.

Held-out groups use the test WAVs every system already produced for scoring (results/fm/lists/test_<name>.json, all
regenerable from the checkpoints with eval_fm.py), so what you hear is exactly what was scored. The 6 sentences are a
fixed seeded draw from the 100 test clips with 2.5-6 s of audio. Hard-sentence groups (no recording exists) are
synthesized here from the checkpoints.

  ../.venv/bin/python make_demo_fm.py --systems results/fm/demo_systems.json
demo_systems.json: {"heldout": [[key, label, list_name], ...], "hard": [[key, label, kind, ckpt, vocoder], ...]}
  kind is "fm" (ckpt = FM export) or "v1" (27's shipped model).
"""
import argparse, base64, json, os
from math import gcd
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from make_demo import level, mp3_b64, HARD
from model_fm import cap_memory
from synth import write_wav, NeuralVocoder
from text import normalize

ROOT = os.path.dirname(os.path.abspath(__file__))


def load16(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        g = gcd(16000, sr)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=os.path.join(ROOT, "results", "fm", "demo_systems.json"))
    ap.add_argument("--media", default=os.path.join(ROOT, "media"))
    ap.add_argument("--n_held", type=int, default=6)
    a = ap.parse_args()
    cfg = json.load(open(a.systems))
    lists = {k: {x["id"]: x for x in json.load(open(os.path.join(ROOT, "results", "fm", "lists", f"test_{n}.json")))}
             for k, _, n in cfg["heldout"]}
    first = lists[cfg["heldout"][0][0]]
    ids = list(first)
    lens = {i: sf.info(first[i]["wav"]).duration for i in ids}
    ok = [i for i in ids if 2.5 <= lens[i] <= 6.0]
    sel = sorted(str(x) for x in np.random.default_rng(3).choice(ok, size=min(a.n_held, len(ok)), replace=False))
    names = {k: lab for k, lab, _ in cfg["heldout"]}
    groups = []
    os.makedirs(a.media, exist_ok=True)
    for gi, cid in enumerate(sel):
        opts = []
        for k, lab, _ in cfg["heldout"]:
            y = level(load16(lists[k][cid]["wav"]).astype(np.float64))
            if gi == 0:
                write_wav(os.path.join(a.media, f"followup_heldout1_{k}.wav"), y)
            opts.append({"lab": base64.b64encode(lab.encode()).decode(), "audio": mp3_b64(y)})
        groups.append({"kind": "held-out test sentence", "text": first[cid]["ref"], "opts": opts})
    if cfg.get("hard"):
        from eval_fm import FM
        from synth import TTS
        hard = [l.split("||")[0].strip() for l in open(os.path.join(ROOT, "hard_sentences.txt")) if l.strip() and not l.startswith("#")]
        engines = {}
        for k, lab, kind, ckpt, voc in cfg["hard"]:
            engines[k] = (lab, FM(ckpt) if kind == "fm" else TTS(), NeuralVocoder(os.path.join(ROOT, voc)), kind)
        for hk in HARD:
            t = hard[hk]
            opts = []
            for k, (lab, eng, voc, kind) in engines.items():
                m = eng(t, seed=0) if kind == "fm" else eng(t)
                y = level(voc(m).astype(np.float64))
                write_wav(os.path.join(a.media, f"followup_hard{hk}_{k}.wav"), y)
                opts.append({"lab": base64.b64encode(lab.encode()).decode(), "audio": mp3_b64(y)})
            groups.append({"kind": "hand-written hard sentence (no recording exists)", "text": t + "  [normalized: " + normalize(t) + "]",
                           "opts": opts})
    html = open(os.path.join(ROOT, "followup_template.html")).read().replace("__DATA__", json.dumps(groups))
    open(os.path.join(a.media, "followup.html"), "w").write(html)
    print("wrote", len(groups), "groups,", os.path.getsize(os.path.join(a.media, "followup.html")) // 1024, "KB; held-out ids", sel)


if __name__ == "__main__":
    main()
