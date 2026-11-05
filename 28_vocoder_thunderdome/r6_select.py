"""Round 6 checkpoint selection, exactly the rule pre-registered in STATUS.md before any val score:
candidates per arm = snapshots every 4k steps + the last one; pick max val100 UTMOS per arm; the shipped generator is
the max-val-UTMOS candidate over the GAN arms. Test sets are never read here.
  ../.venv/bin/python r6_select.py --arms recon,gan,gan_mrstft --gan_arms gan,gan_mrstft --export checkpoints/r6_gan
Prints every candidate's val UTMOS / GPE / mel L1, writes results_r6/selection.json, and (with --export) copies the
shipped snapshot to <export>.safetensors + .json (loadable by r6_vocoder.load) with provenance.
"""
import argparse, glob, json, os, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(ROOT, "results_r6")


def candidates(arm):
    snaps = sorted(int(os.path.basename(f)[2:-12]) for f in glob.glob(os.path.join(ROOT, "checkpoints", "r6_runs", arm, "snaps", "G_*.safetensors")))
    if not snaps:
        return []
    keep = [s for s in snaps if s % 4000 == 0]
    if snaps[-1] not in keep:
        keep.append(snaps[-1])
    rows = []
    for s in keep:
        p = os.path.join(R, "mos", "val100", f"{arm}_s{s}.json")
        m = os.path.join(R, "metrics", "val100", f"{arm}_s{s}.json")
        if not os.path.exists(p):
            rows.append({"arm": arm, "step": s, "utmos": None}); continue
        d = json.load(open(p)); e = json.load(open(m))["summary"]
        rows.append({"arm": arm, "step": s, "utmos": d["mos"], "ci95": d["ci95"], "gpe_pct": round(e["gpe_pct"], 3),
                     "mel_l1": round(e["mel_l1"], 4), "mrstft": round(e["mrstft"], 4),
                     "ckpt": os.path.join("checkpoints", "r6_runs", arm, "snaps", f"G_{s}.safetensors")})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="recon,gan,gan_mrstft")
    ap.add_argument("--gan_arms", default="gan,gan_mrstft")
    ap.add_argument("--export", default="")
    a = ap.parse_args()
    out = {"rule": "per arm: max val100 UTMOS over 4k snapshots + last; shipped = max over GAN arms", "arms": {}}
    for arm in a.arms.split(","):
        rows = candidates(arm)
        scored = [r for r in rows if r["utmos"] is not None]
        for r in rows:
            print(arm, r["step"], r.get("utmos"), r.get("ci95"), "GPE", r.get("gpe_pct"), "melL1", r.get("mel_l1"))
        best = max(scored, key=lambda r: r["utmos"]) if scored else None
        out["arms"][arm] = {"candidates": rows, "selected": best, "missing": [r["step"] for r in rows if r["utmos"] is None]}
        print(f"  -> {arm} selected:", best and best["step"], "missing:", out["arms"][arm]["missing"])
    ship = max((out["arms"][g]["selected"] for g in a.gan_arms.split(",") if out["arms"].get(g, {}).get("selected")), key=lambda r: r["utmos"])
    out["shipped"] = ship
    print("SHIPPED:", ship["arm"], ship["step"], ship["utmos"])
    json.dump(out, open(os.path.join(R, "selection.json"), "w"), indent=1)
    if a.export:
        src = os.path.join(ROOT, ship["ckpt"])
        shutil.copyfile(src, a.export + ".safetensors")
        meta = json.load(open(src.replace(".safetensors", ".json")))
        meta.update({"selected_by": out["rule"], "val100_utmos": ship["utmos"], "val100_gpe_pct": ship["gpe_pct"],
                     "source_snapshot": ship["ckpt"],
                     "mel": "16 kHz, n_fft 1024, Hann 1024, hop 256, centred, 80 Slaney bands 0-8 kHz, ln(max(mel,1e-5))"})
        json.dump(meta, open(a.export + ".json", "w"), indent=1)
        print("exported", a.export + ".safetensors")


if __name__ == "__main__":
    main()
