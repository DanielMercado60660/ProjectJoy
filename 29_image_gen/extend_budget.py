"""Create a continuation run that resumes a finished run's exact state with a larger budget (go-deeper 2x runs).

    ../.venv/bin/python extend_budget.py runs/flow runs/flow_x2 --pflop 5 --minutes 80   # U-Nets (FLOP-bound)
    ../.venv/bin/python extend_budget.py runs/gan  runs/gan_x2  --pflop 5 --minutes 50   # VAE/GAN (wall-bound)
then train with:  ../bin/gpu_run ../.venv/bin/python train_gen.py --model flow --run runs/flow_x2 --resume
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("src"); ap.add_argument("dst")
ap.add_argument("--pflop", type=float, default=5.0)
ap.add_argument("--minutes", type=float, default=50.0)
a = ap.parse_args()
os.makedirs(a.dst, exist_ok=True)
shutil.copytree(os.path.join(a.src, "latest"), os.path.join(a.dst, "latest"), dirs_exist_ok=True)
c = json.load(open(os.path.join(a.src, "config.json")))
c.update(run=a.dst, budget_pflop=a.pflop, budget_min=a.minutes, n_ckpts=0,
         x2_note=f"continuation of {a.src} (same optimizer/EMA/RNG state) to {a.pflop} PFLOP / {a.minutes} train-min")
json.dump(c, open(os.path.join(a.dst, "config.json"), "w"), indent=1)
m = json.load(open(os.path.join(a.dst, "latest", "meta.json")))
m.update(done=False, stop_reason=None, ckpts=[])  # keep the old snaps list so no catch-up snapshots fire
json.dump(m, open(os.path.join(a.dst, "latest", "meta.json"), "w"))
print("created", a.dst, "from step", m["step"])
