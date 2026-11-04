"""How scripted is an edited net? Drive it on the fresh held-out tracks (strict, grip 1.0), record the
observations it visits, and compare its actions there with the tuned ray follower's and the
strict-trained ES policy's actions on the same observations (Pearson r of steer and of pedal).
    python wsp/action_similarity.py --start 90000 --n 200
"""
import argparse, json, os, sys
os.environ.setdefault("NUMBA_NUM_THREADS", "4")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wsp_common import ROOT, WSP, WeightsController, load_weights  # noqa: E402
from racing.controllers import get_controller  # noqa: E402
from racing.evaluate import run_lap  # noqa: E402
from racing.track import make_tracks, pack  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--start", type=int, required=True); ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()
bank = pack(make_tracks("heldout", a.n, start=a.start))
refs = {"ray": get_controller("ray"), "es_strict": get_controller("es")}
pols = {"astra_edit_a1": "wsp/edit_a1/best.npz", "astra_edit_a2": "wsp/edit_a2/best.npz",
        "astra_oneshot_a1": "wsp/edit0_a1/final.npz", "es_ft_hack_x100_seed1": "wsp/baselines/es_ft_hack_x100_seed1.npz",
        "es_strict": "checkpoints/es_best.npz"}
out = {}
for n, p in pols.items():
    c = WeightsController(load_weights(os.path.join(ROOT, p)))
    O = []
    def f(o, env=None):
        O.append(o[~env.done].copy()); return c(o)
    run_lap(f, bank)
    O = np.concatenate(O)
    A = np.clip(c(O), -1, 1)
    out[n] = {}
    for rn, rc in refs.items():
        R = np.clip(rc(O), -1, 1)
        out[n][rn] = dict(steer_r=float(np.corrcoef(A[:, 0], R[:, 0])[0, 1]), pedal_r=float(np.corrcoef(A[:, 1], R[:, 1])[0, 1]))
    print(n, json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out[n].items()}))
json.dump(out, open(os.path.join(WSP, "results", "action_similarity.json"), "w"), indent=1)
