"""Paired bootstrap on the shared 100 test clips: 33's flow model vs project 27 (v1 and shipped v2) and f2.

  ../.venv/bin/python compare_27.py      # -> results/test_vs_27.json

Reads Whisper judge and UTMOS per-clip files only (27's are read-only here). CER/WER deltas are clip-mean
in percentage points (the headline numbers are corpus CER = total edits / total reference characters). Positive delta = 33 is better.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P27 = os.path.join(HERE, "..", "27_pocket_voice_foundry", "results", "fm")
SYSTEMS = {
    "33 f3": (f"{HERE}/results/judge/test_f3.json", f"{HERE}/results/utmos/test_f3.json"),
    "33 f2": (f"{HERE}/results/judge/test_f2.json", f"{HERE}/results/utmos/test_f2.json"),
    "27 v2 shipped (flow-matching mel, LJ only)": (f"{P27}/judge/test_fm_lj.json", f"{P27}/mos/test_fm_lj.json"),
    "27 v1": (f"{P27}/judge/test_p27v1.json", f"{P27}/mos/test_p27v1.json"),
    "real recording": (f"{P27}/judge/test_real.json", f"{P27}/mos/test_real.json"),
}


def load(judge, mos):
    j = {c["id"]: c for c in json.load(open(judge))["clips"]}
    m = {c["id"]: c["mos"] for c in json.load(open(mos))["clips"]}
    return {i: (100 * j[i]["cer"], 100 * j[i]["wer"], m[i]) for i in j if i in m}   # CER/WER in %


def ci(x, rng, B=5000):
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def main():
    data = {k: load(*v) for k, v in SYSTEMS.items()}
    ids = sorted(set.intersection(*[set(d) for d in data.values()]))
    me = np.array([data["33 f3"][i] for i in ids])
    out = {"n_clips": len(ids), "means": {}, "f3_vs": {}}
    for name, d in data.items():
        a = np.array([d[i] for i in ids])
        out["means"][name] = {"cer": round(float(a[:, 0].mean()), 3), "wer": round(float(a[:, 1].mean()), 3),
                              "utmos": round(float(a[:, 2].mean()), 3)}
        if name == "33 f3":
            continue
        rng = np.random.default_rng(0)
        dcer, dwer, dmos = a[:, 0] - me[:, 0], a[:, 1] - me[:, 1], me[:, 2] - a[:, 2]
        out["f3_vs"][name] = {
            "cer_gain": round(float(dcer.mean()), 3), "cer_gain_ci": ci(dcer, rng),
            "wer_gain": round(float(dwer.mean()), 3), "wer_gain_ci": ci(dwer, rng),
            "utmos_gain": round(float(dmos.mean()), 3), "utmos_gain_ci": ci(dmos, rng),
            "cer_better_equal_worse": [int((me[:, 0] < a[:, 0]).sum()), int((me[:, 0] == a[:, 0]).sum()),
                                       int((me[:, 0] > a[:, 0]).sum())],
            "utmos_better_clips": int((me[:, 2] > a[:, 2]).sum())}
    json.dump(out, open(os.path.join(HERE, "results", "test_vs_27.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
