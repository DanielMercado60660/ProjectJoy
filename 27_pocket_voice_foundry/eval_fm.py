"""Generate and proxy-score a flow-matching export on 27's evaluation clips, and write the judge list.

  ../bin/gpu_run ../.venv/bin/python eval_fm.py --ckpt checkpoints/fm/tts_fm --name fm_ft --split test
  ../bin/gpu_run ../.venv/bin/python eval_fm.py --system v1 --vocoder checkpoints/vocoder_gan/vocoder.safetensors --name p27v1_gan --split test
Splits: test = 27's 100 test clips (seed-0 subset of LJ045-050; ids, order and references from results/fm/lists/test_real.json,
a copy of ../33_gemma_voice's phase0 judge list, whose ids equal 27's results/eval_summary.json ids); val = 33's 100-clip
seed-0 sample of LJ041-044 (results/fm/lists/val_real.json), used only for choosing checkpoints and sampler settings.
Each clip: text (27's normalized LJ text) -> durations (val-fitted pace) -> ODE sample (seed = clip position, so every
system/setting sees the same noise per clip) -> 28's vocoder -> 16 kHz wav in runs/fm/gen/<split>_<name>/.
Writes results/fm/lists/<split>_<name>.json (for judge.py and mos.py) and results/fm/proxy/<split>_<name>.json:
27's CTC proxy CER (predicted mel, and the vocoded wav re-analysed), DTW log-mel L1 / MCD-like vs the real mel, global
variance ratio, length ratio, and generation time.
"""
import argparse, json, os, time
import numpy as np
import mlx.core as mx
from model_fm import load_fm, infer, cap_memory
from mel import log_mel
from text import encode
from synth import NeuralVocoder, write_wav
from ctc import load_recognizer, transcribe, edit_distance, ref_text
from eval import dist

ROOT = os.path.dirname(os.path.abspath(__file__))
LISTS = os.path.join(ROOT, "results", "fm", "lists")  # copies of 33's phase0 lists (same ids, order, refs)


def hard_clips():
    """27's hand-written hard sentences (no recordings). ref = 27's normalized text (numbers spelled out)."""
    from text import normalize
    out = []
    for k, line in enumerate(l for l in open(os.path.join(ROOT, "hard_sentences.txt")) if l.strip() and not l.startswith("#")):
        sent, words = line.split("||")
        out.append(dict(id=f"hard{k}", ref=normalize(sent.strip()), text=normalize(sent.strip()), mel=None,
                        words=[w.strip() for w in words.split(",")]))
    return out


def clips_for(split):
    if split == "hard":
        return hard_clips()
    lst = json.load(open(os.path.join(LISTS, f"{split}_real.json")))
    z = np.load(os.path.join(ROOT, "cache2", "lj", "index.npz"))
    pos = {str(k): i for i, k in enumerate(z["ids"])}
    mels = np.memmap(os.path.join(ROOT, "cache2", "lj", "mels.f16"), np.float16, "r").reshape(-1, 80)
    out = []
    for it in lst:
        i = pos[it["id"]]
        out.append(dict(id=it["id"], ref=it["ref"], text=str(z["texts"][i]),
                        mel=np.array(mels[z["offs"][i]:z["offs"][i + 1]], np.float32)))
    return out


class FM:
    def __init__(self, path, n_steps=None, temp=None, solver=None, spk=0):
        """Sampler settings default to the checkpoint's val-chosen "sampler" entry (shipped checkpoints), else Euler 10 / 0.667."""
        self.model, self.cfg = load_fm(path + ".safetensors" if not path.endswith(".safetensors") else path)
        smp = self.cfg.get("sampler", {})
        n_steps = n_steps or smp.get("n_steps", 10)
        temp = temp if temp is not None else smp.get("temp", 0.667)
        solver = solver or smp.get("solver", "euler")
        self.mean = np.array(self.cfg["mel_mean"], np.float32); self.std = np.array(self.cfg["mel_std"], np.float32)
        self.pace, self.rounding = self.cfg.get("pace", 1.0), self.cfg.get("rounding", "round")
        self.n_steps, self.temp, self.solver, self.spk = n_steps, temp, solver, spk

    def mel_from_tokens(self, tok, seed=0, **kw):
        m, d = infer(self.model, mx.array(np.asarray(tok, np.int32)[None]), spk=self.spk, pace=self.pace, rounding=self.rounding,
                     n_steps=self.n_steps, temp=self.temp, seed=seed, solver=self.solver, **kw)
        return m * self.std + self.mean, d

    def __call__(self, text, seed=0):
        from text import normalize
        return self.mel_from_tokens(encode(normalize(text)), seed)[0]


def main():
    cap_memory()
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="FM export (for --system fm)")
    ap.add_argument("--system", default="fm", choices=["fm", "v1", "real"], help="v1 = 27's shipped FastSpeech model; real = the recording's mel")
    ap.add_argument("--vocoder", default=os.path.join(ROOT, "checkpoints", "vocoder28", "vocoder.safetensors"))
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "hard"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_steps", type=int, default=None, help="default: checkpoint's sampler entry, else 10")
    ap.add_argument("--temp", type=float, default=None, help="default: checkpoint's sampler entry, else 0.667")
    ap.add_argument("--solver", default=None, help="default: checkpoint's sampler entry, else euler")
    ap.add_argument("--spk", type=int, default=0)
    a = ap.parse_args()
    clips = clips_for(a.split)[: a.n]
    if a.system == "fm":
        tts = FM(a.ckpt, a.n_steps, a.temp, a.solver, a.spk)
    elif a.system == "v1":
        from synth import TTS
        tts = TTS()
    voc = NeuralVocoder(a.vocoder)
    rec, rcfg = load_recognizer(os.path.join(ROOT, "checkpoints", "ctc", "ctc.safetensors"))
    tag = f"{a.split}_{a.name}"
    wdir = os.path.join(ROOT, "runs", "fm", "gen", tag)
    os.makedirs(wdir, exist_ok=True)
    pred, rean, per, lst = [], [], [], []
    gen_t = []
    for k, c in enumerate(clips):
        tok = encode(c["text"])
        t0 = time.time()
        if a.system == "real" and c["mel"] is None:
            raise SystemExit("no recordings for the hard set")
        if a.system == "real":
            m = c["mel"]
        elif a.system == "v1":
            m, d = tts.mel_from_tokens(tok)
        else:
            m, d = tts.mel_from_tokens(tok, seed=k)
        y = voc(m)
        gen_t.append((time.time() - t0, len(y) / 16000))
        wav = os.path.join(wdir, c["id"] + ".wav")
        write_wav(wav, y)
        lst.append({"id": c["id"], "wav": wav, "ref": c["ref"]})
        am = log_mel(y)
        pred.append(m); rean.append(am)
        if c["mel"] is None:
            per.append({"id": c["id"]})
            continue
        l1, mcd = dist(am, c["mel"])
        per.append({"id": c["id"], "len_ratio": len(m) / len(c["mel"]), "l1": l1, "mcd": mcd,
                    "gv": float(np.mean(m.std(0) / (c["mel"].std(0) + 1e-6)))})
    refs = [ref_text(c["text"]) for c in clips]
    res = {}
    for nm, ms in (("mel", pred), ("neural", rean)):
        hyps = transcribe(rec, rcfg, ms)
        res[f"ctc_cer_{nm}"] = 100 * sum(edit_distance(h, r) for h, r in zip(hyps, refs)) / sum(len(r) for r in refs)
        for p, h, r in zip(per, hyps, refs):
            p[f"ctc_cer_{nm}"] = edit_distance(h, r) / max(1, len(r))
    if a.split == "hard":
        json.dump(lst, open(os.path.join(ROOT, "results", "fm", "lists", tag + ".json"), "w"), indent=0)
        print(json.dumps(res), flush=True)
        return
    lr = np.log([p["len_ratio"] for p in per])
    gt = np.array(gen_t[1:])  # first call includes compilation
    res.update(n=len(clips), dtw_logmel_l1=float(np.mean([p["l1"] for p in per])), mcd_like_db=float(np.mean([p["mcd"] for p in per])),
               gv_ratio=float(np.mean([p["gv"] for p in per])), len_ratio_geomean=float(np.exp(lr.mean())),
               abs_len_err_pct=float(np.mean(np.abs(np.exp(lr) - 1)) * 100),
               gen_ms_median=float(np.median(gt[:, 0]) * 1000), rtf_median=float(np.median(gt[:, 0] / gt[:, 1])),
               system=a.system, ckpt=a.ckpt, vocoder=os.path.relpath(a.vocoder, ROOT))
    if a.system != "real":
        res.update(step=tts.cfg.get("step"), pace=tts.pace, rounding=tts.rounding)
    if a.system == "fm":
        res.update(n_steps=tts.n_steps, temp=tts.temp, solver=tts.solver)
    os.makedirs(os.path.join(ROOT, "results", "fm", "lists"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "fm", "proxy"), exist_ok=True)
    json.dump(lst, open(os.path.join(ROOT, "results", "fm", "lists", tag + ".json"), "w"), indent=0)
    json.dump({**res, "clips": per}, open(os.path.join(ROOT, "results", "fm", "proxy", tag + ".json"), "w"), indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)


if __name__ == "__main__":
    main()
