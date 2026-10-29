"""Phase 2, amendment 7a: the bounded sampler experiments behind the 2,000-step window warm-up for H/Hr.

    OMP_NUM_THREADS=4 ../bin/gpu_run ../.venv/bin/python p2_stream_debug.py MODE STEPS [flags]
Trains a fresh R3-style Policy2 (binary walk + discrete turn heads) for STEPS steps, batch 32, window 128,
on the BC demos + filtered DAgger rounds 1-2, prints mean loss / turn CE / walk BCE every 50 steps, then
evaluates on 4 fixed window batches and 4 fixed chunk batches (seed 123).
MODE: window (R3's sampler); iid_chunks (aligned 128-tick chunks, episode uniform, zero memory);
iid_ticks (aligned chunks, episode in proportion to its chunks); stream / stream_reset (32 streams in order,
memory carried / zeroed); pool / pool_reset (512 streams, 32 drawn per step).
Flags: nostart (window with p_start 0), noburn (window with no burn-in), burn (chunks: mask 32 ticks of
non-first chunks), nodone (chunks: no done positives), notail (chunks: no last partial chunk), h0none (chunks:
no explicit zero state), warm=N (first N steps use the window sampler).
Run on 2026-10-27 18:05-18:30 (amendment 7a): only window, iid_chunks and warm-started pool learned. The first
two runs (stream, window, 300 steps) used batch 16 and the BC demos only; the rest used the settings above.
"""
import sys, numpy as np, mlx.core as mx, mlx.nn as nn, mlx.optimizers as optim
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_train
from p2_learner import Policy2, policy_config
rows, eps = p2_train.load_all(["results/m5/bc_index.jsonl","episodes/dagger_filtered_index.jsonl","episodes/dagger2_filtered_index.jsonl"], 64)
lens = np.array([len(e["frames"]) for e in eps], float)
B, W, STEPS = 32, 128, int(sys.argv[2])
mode = sys.argv[1]
mx.random.seed(0)
m = Policy2(img=64, act="leaky", vx_head="binary", turn_head="discrete"); mx.eval(m.parameters())
opt = optim.Adam(learning_rate=3e-4); vg = nn.value_and_grad(m, p2_train.loss_fn)
rng = np.random.default_rng(0); st = p2_train.Streams(B, m.gru.hidden_size, 512 if mode.startswith('pool') else None)
acc = []; gn = []
WARM = int([a for a in sys.argv if a.startswith("warm=")][0][5:]) if any(a.startswith("warm=") for a in sys.argv) else 0
real_mode = mode
for i in range(STEPS):
    mode = "window" if i < WARM else real_mode
    if mode in ("iid_chunks", "iid_ticks"):
        nch = np.array([(len(e["frames"]) - 1) // 128 + 1 for e in eps], float)
        st.ep[:] = rng.integers(len(eps), size=B) if mode == "iid_chunks" else rng.choice(len(eps), size=B, p=nch / nch.sum())
        st.pos[:] = [128 * rng.integers(0, max(1, len(eps[j]["frames"]) // 128) if "notail" in sys.argv else (len(eps[j]["frames"]) - 1) // 128 + 1) for j in st.ep]
        arrays, h0 = st.next(eps, rng, W); h0 = np.zeros_like(h0)
        if "burn" in sys.argv:
            Mb = arrays[5].copy(); Mb[st.pos[st.last] - W > 0, :32] = 0; arrays = arrays[:5] + (Mb,) + arrays[6:]
        if "nodone" in sys.argv: arrays = arrays[:4] + (arrays[4] * 0,) + arrays[5:]
        (loss, met), g = vg(m, *[mx.array(x) for x in arrays], aux_weight=1.0, h0=None if "h0none" in sys.argv else mx.array(h0))
    elif mode.startswith("stream") or mode.startswith("pool"):
        arrays, h0 = st.next(eps, rng, W)
        if mode in ("stream_reset", "pool_reset"): h0 = np.zeros_like(h0)
        (loss, met), g = vg(m, *[mx.array(x) for x in arrays], aux_weight=1.0, h0=mx.array(h0))
    else:
        arrays = p2_train.batch(eps, lens, rng, B, W, 0 if "noburn" in sys.argv else 32, 0.0 if "nostart" in sys.argv else 0.2)
        (loss, met), g = vg(m, *[mx.array(x) for x in arrays], aux_weight=1.0)
    g, gnorm = optim.clip_grad_norm(g, 1.0); gn.append(float(gnorm)); opt.update(m, g); mx.eval(m.parameters(), opt.state, loss, met)
    if mode in ("stream", "pool"): st.update(np.array(met["h_last"]))
    acc.append([float(loss), float(met["turn_ce"]), float(met["walk_bce"])])
    if (i + 1) % 50 == 0:
        a = np.mean(acc[-50:], 0)
        print(mode, i + 1, "loss %.3f turn %.3f walk %.3f" % tuple(a), "| batch: mean pos", float(st.pos.mean()) if mode != "window" else "-", "n_eps_short", int(sum(len(eps[j]["frames"]) <= 128 for j in st.ep[st.last])) if mode != "window" else "-", "reg_vx_mean %.3f" % float(arrays[3][..., 0][arrays[5] > 0].mean()), "gradnorm %.1f" % float(np.median(gn[-50:])), "done_pos", int(arrays[4].sum()), flush=True)

mode = real_mode
# cross-evaluation on fixed batches of both kinds
erng = np.random.default_rng(123)
nch = np.array([(len(e["frames"]) - 1) // 128 + 1 for e in eps], float)
res = {"window": [], "chunk": []}
for _ in range(4):
    a = p2_train.batch(eps, lens, erng, B, W, 32)
    l, mt = p2_train.loss_fn(m, *[mx.array(x) for x in a]); res["window"].append([float(l), float(mt["turn_ce"]), float(mt["walk_bce"]), float(mt["mse"])])
    st2 = p2_train.Streams(B, m.gru.hidden_size)
    st2.ep[:] = erng.choice(len(eps), size=B, p=nch / nch.sum())
    st2.pos[:] = [128 * erng.integers(0, (len(eps[j]["frames"]) - 1) // 128 + 1) for j in st2.ep]
    a, h0 = st2.next(eps, erng, W)
    l, mt = p2_train.loss_fn(m, *[mx.array(x) for x in a]); res["chunk"].append([float(l), float(mt["turn_ce"]), float(mt["walk_bce"]), float(mt["mse"])])
for k, v in res.items():
    print("EVAL trained=%s on %s batches: loss %.3f turn %.3f walk %.3f mse %.3f" % ((mode, k) + tuple(np.mean(v, 0))), flush=True)
