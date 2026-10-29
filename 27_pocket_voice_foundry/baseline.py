"""Cheap baseline TTS: per-character template mels + average durations, concatenated and smoothed.

No neural network and no help from the learned model. Its alignment comes from classical Viterbi
("flat start") training of a left-to-right template model: every character is K states, each state a
unit-variance Gaussian mean in normalized-mel space. Start from a uniform segmentation, then alternate
(means <- frames assigned) and (re-align with the same monotonic search the learned model uses).
After the last pass, state means and mean durations are tabulated per character, or per
(left, char, right) context when that context was seen >= min_count times (else back off to the char).
Synthesis = repeat each state mean for its rounded mean duration (x dur_scale), then Gaussian-smooth in time.

Tuned on the validation split (LJ041-044) only: K, context/min_count, smoothing sigma, dur_scale.
Selection: lowest val CER of the CTC recognizer on the baseline's mel (the intelligibility proxy, the main Measure),
ties broken by val DTW log-mel L1. The DTW-L1-best config is saved too (baseline_dtw.npz) because L1 rewards
oversmoothing; both are in results/baseline_tuning.json. Without --ctc (smoke test), selection falls back to DTW L1.
  ../bin/gpu_run ../.venv/bin/python baseline.py --ctc checkpoints/ctc/ctc.safetensors   # -> results/baseline_tuning.json, checkpoints/baseline/baseline.npz
"""
import argparse, json, os, time
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from mas import maximum_path_batch, dtw_cost
from text import encode, SYMBOLS

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_split(cache, name, limit=0):
    z = np.load(os.path.join(cache, f"{name}.npz"))
    n = limit or len(z["ids"])
    M, offs = z["mels"], z["offs"]
    mels = [M[offs[i]:offs[i + 1]].astype(np.float32) for i in range(n)]
    return list(z["ids"][:n]), list(z["texts"][:n]), mels


def viterbi_train(toks, xs, K, iters, n_sym):
    """toks: list of id arrays, xs: list of normalized mels. Returns per-utterance state index per frame."""
    # uniform init
    align = []
    for tk, x in zip(toks, xs):
        S, T = len(tk) * K, len(x)
        align.append(np.minimum((np.arange(T) * S) // T, S - 1).astype(np.int32))
    for it in range(iters + 1):
        sums = np.zeros((n_sym * K, 80)); cnt = np.zeros(n_sym * K)
        for tk, x, al in zip(toks, xs, align):
            st = (tk[al // K] * K + al % K)
            np.add.at(sums, st, x); np.add.at(cnt, st, 1)
        means = (sums / np.maximum(cnt, 1)[:, None]).astype(np.float32)
        if it == iters:
            break
        tot = 0.0
        for k0 in range(0, len(toks), 64):  # realign in batches
            bt, bx = toks[k0:k0 + 64], xs[k0:k0 + 64]
            Smax = max(len(t) for t in bt) * K; Tmax = max(len(x) for x in bx)
            V = np.full((len(bt), Smax, Tmax), -1e30, np.float32)
            for b, (tk, x) in enumerate(zip(bt, bx)):
                st = (np.repeat(tk, K) * K + np.tile(np.arange(K), len(tk)))
                mu = means[st]
                V[b, : len(st), : len(x)] = -0.5 * ((x * x).sum(1)[None] + (mu * mu).sum(1)[:, None] - 2 * mu @ x.T)
            idx = maximum_path_batch(V, np.array([len(t) * K for t in bt], np.int64), np.array([len(x) for x in bx], np.int64))
            for b, x in enumerate(bx):
                align[k0 + b] = idx[b, : len(x)].astype(np.int32)
                tot += V[b, idx[b, : len(x)], np.arange(len(x))].sum()
        print(f"  viterbi iter {it + 1}/{iters}: mean frame loglik {tot / sum(len(x) for x in xs):.3f}", flush=True)
    return align


def tabulate(toks, xs, align, K):
    """Per-char and per-(left,char,right) sums of state means / durations (one pass; filter by count later)."""
    stats = {}  # key -> [sum of state means (K,80), sum of durations (K,), count]
    for tk, x, al in zip(toks, xs, align):
        N = len(tk)
        cnt = np.bincount(al, minlength=N * K).astype(np.float64)
        sm = np.zeros((N * K, 80)); np.add.at(sm, al, x)
        mu = (sm / np.maximum(cnt, 1)[:, None]).reshape(N, K, 80); d = cnt.reshape(N, K)
        tl = [-1, -1, -1] + [int(t) for t in tk] + [-1, -1, -1]
        for n in range(N):  # n-th char is tl[n + 3]
            for key in (tuple(tl[n:n + 7]), tuple(tl[n + 1:n + 6]), tuple(tl[n + 2:n + 5]), tl[n + 3]):
                e = stats.get(key)
                if e is None:
                    stats[key] = [mu[n].copy(), d[n].copy(), 1]
                else:
                    e[0] += mu[n]; e[1] += d[n]; e[2] += 1
    return stats


def tables(stats, min_count, quin_min=10 ** 9, sept_min=10 ** 9):
    """mono table + one context dict holding 3-, 5- and 7-char contexts seen >= min_count / quin_min / sept_min times."""
    mono = {k: (v[0] / v[2], v[1] / v[2]) for k, v in stats.items() if isinstance(k, int)}
    thr = {3: min_count, 5: quin_min, 7: sept_min}
    tri = {k: (v[0] / v[2], v[1] / v[2]) for k, v in stats.items() if isinstance(k, tuple) and v[2] >= thr[len(k)]}
    return mono, tri


class Baseline:
    def __init__(self, mono, tri, K, sigma, dur_scale, mean, std):
        self.mono, self.tri, self.K, self.sigma, self.dur_scale = mono, tri, K, sigma, dur_scale
        self.mean, self.std = mean, std

    def synth(self, tok):
        """tok: id array (with edge spaces) -> raw log-mel (T,80), per-char durations."""
        frames, durs = [], []
        N = len(tok)
        for n in range(N):
            c = lambda j: int(tok[j]) if 0 <= j < N else -1
            ctx7, ctx5, ctx3 = tuple(c(n + j) for j in range(-3, 4)), tuple(c(n + j) for j in range(-2, 3)), (c(n - 1), c(n), c(n + 1))
            mu, d = (self.tri.get(ctx7) or self.tri.get(ctx5) or self.tri.get(ctx3) or self.mono.get(int(tok[n]))
                     or self.mono[SYMBOLS.index(" ")])
            dd = np.maximum(1, np.round(d * self.dur_scale)).astype(int)
            durs.append(dd.sum())
            for k in range(self.K):
                frames.append(np.repeat(mu[k][None], dd[k], 0))
        m = np.concatenate(frames, 0)
        if self.sigma > 0:
            m = gaussian_filter1d(m, self.sigma, axis=0, mode="nearest")
        return (m * self.std + self.mean).astype(np.float32), np.array(durs)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mk = sorted(self.mono); tk = sorted(k for k in self.tri if len(k) == 3); qk = sorted(k for k in self.tri if len(k) == 5)
        sk = sorted(k for k in self.tri if len(k) == 7)
        np.savez_compressed(path, K=self.K, sigma=self.sigma, dur_scale=self.dur_scale, mean=self.mean, std=self.std,
                            mono_keys=np.array(mk), mono_mu=np.array([self.mono[k][0] for k in mk], np.float16),
                            mono_d=np.array([self.mono[k][1] for k in mk], np.float32),
                            tri_keys=np.array(tk, np.int32).reshape(-1, 3), tri_mu=np.array([self.tri[k][0] for k in tk], np.float16).reshape(-1, self.K, 80),
                            tri_d=np.array([self.tri[k][1] for k in tk], np.float32).reshape(-1, self.K),
                            quin_keys=np.array(qk, np.int32).reshape(-1, 5), quin_mu=np.array([self.tri[k][0] for k in qk], np.float16).reshape(-1, self.K, 80),
                            quin_d=np.array([self.tri[k][1] for k in qk], np.float32).reshape(-1, self.K),
                            sept_keys=np.array(sk, np.int8).reshape(-1, 7), sept_mu=np.array([self.tri[k][0] for k in sk], np.float16).reshape(-1, self.K, 80),
                            sept_d=np.array([self.tri[k][1] for k in sk], np.float32).reshape(-1, self.K))

    @staticmethod
    def load(path):
        z = np.load(path)
        mono = {int(k): (mu.astype(np.float32), d) for k, mu, d in zip(z["mono_keys"], z["mono_mu"], z["mono_d"])}
        tri = {tuple(int(v) for v in k): (mu.astype(np.float32), d) for k, mu, d in zip(z["tri_keys"], z["tri_mu"], z["tri_d"])}
        if "quin_keys" in z:
            tri.update({tuple(int(v) for v in k): (mu.astype(np.float32), d) for k, mu, d in zip(z["quin_keys"], z["quin_mu"], z["quin_d"])})
        if "sept_keys" in z:
            tri.update({tuple(int(v) for v in k): (mu.astype(np.float32), d) for k, mu, d in zip(z["sept_keys"], z["sept_mu"], z["sept_d"])})
        return Baseline(mono, tri, int(z["K"]), float(z["sigma"]), float(z["dur_scale"]), z["mean"], z["std"])


# context back-off settings swept on val: (name, min count for 3-char contexts, min count for 5-char contexts)
# (round 2 added 5-char contexts and lower counts, round 3 added 7-char contexts: the best setting kept landing on the grid edge)
CONTEXTS = [("mono", 10 ** 9, 10 ** 9, 10 ** 9), ("tri_min50", 50, 10 ** 9, 10 ** 9), ("tri_min10", 10, 10 ** 9, 10 ** 9),
            ("tri_min5", 5, 10 ** 9, 10 ** 9), ("tri_min2", 2, 10 ** 9, 10 ** 9), ("quin_min10_tri5", 5, 10, 10 ** 9),
            ("quin_min5_tri5", 5, 5, 10 ** 9), ("quin_min2_tri2", 2, 2, 10 ** 9), ("quin_min1_tri1", 1, 1, 10 ** 9),
            ("sept_min5_quin2_tri2", 2, 2, 5), ("sept_min2_quin2_tri2", 2, 2, 2), ("sept_min1_quin1_tri1", 1, 1, 1)]


def val_score(bl, toks, mels):
    """mean DTW log-mel L1 (raw log-mel) and mean |log duration ratio| on val clips."""
    dists, lr = [], []
    for tk, m in zip(toks, mels):
        s, _ = bl.synth(tk)
        dists.append(dtw_cost(s.astype(np.float64), m.astype(np.float64))[0])
        lr.append(np.log(len(s) / len(m)))
    return float(np.mean(dists)), float(np.mean(np.abs(lr))), float(np.exp(np.mean(lr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--n_train", type=int, default=0, help="0 = all training clips")
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--Ks", default="1,2,3")
    ap.add_argument("--contexts", default="", help="comma list of CONTEXTS names to sweep (default: all)")
    ap.add_argument("--sigmas", default="0,1,2,3,5", help="time-smoothing widths (frames) to sweep")
    ap.add_argument("--out", default=os.path.join(ROOT, "checkpoints", "baseline", "baseline.npz"))
    ap.add_argument("--tune_out", default=os.path.join(ROOT, "results", "baseline_tuning.json"))
    ap.add_argument("--ctc", default=None, help="recognizer for val-CER selection (omit: select by DTW L1)")
    a = ap.parse_args()
    rec_m = None
    if a.ctc:
        from ctc import load_recognizer, transcribe, edit_distance, ref_text
        rec_m, rcfg = load_recognizer(a.ctc)
        vrefs = None
    info = json.load(open(os.path.join(a.cache, "info.json")))
    mean, std = np.array(info["mel_mean"], np.float32), np.array(info["mel_std"], np.float32)
    _, ttexts, tmels = load_split(a.cache, "train", a.n_train)
    _, vtexts, vmels = load_split(a.cache, "val", a.n_val)
    ttoks = [np.array(encode(t)) for t in ttexts]; vtoks = [np.array(encode(t)) for t in vtexts]
    txs = [(m - mean) / std for m in tmels]
    grid, best, best_dtw = [], None, None
    if rec_m is not None:
        vrefs = [ref_text(t) for t in vtexts]
    t0 = time.time()
    for K in [int(k) for k in a.Ks.split(",")]:
        print(f"K={K}: viterbi training on {len(txs)} clips", flush=True)
        al = viterbi_train(ttoks, txs, K, a.iters, len(SYMBOLS))
        stats = tabulate(ttoks, txs, al, K)
        for ctx_name, mc, qm, sm in CONTEXTS:
            if a.contexts and ctx_name not in a.contexts.split(","):
                continue
            mono, tri = tables(stats, mc, qm, sm)
            # duration scale: fit total duration on val (ratio of geometric means), then sweep smoothing
            bl = Baseline(mono, tri, K, 0.0, 1.0, mean, std)
            _, _, ratio = val_score(bl, vtoks[:40], vmels[:40])
            for ds in sorted({round(f / ratio, 3) for f in (0.9, 1.0, 1.1)} | {1.0}):
                for sigma in [float(v) for v in a.sigmas.split(",")]:
                    bl = Baseline(mono, tri, K, sigma, ds, mean, std)
                    d, ldr, r = val_score(bl, vtoks, vmels)
                    rec = {"K": K, "context": ctx_name, "n_tri": sum(len(k) == 3 for k in tri), "n_quin": sum(len(k) == 5 for k in tri), "n_sept": sum(len(k) == 7 for k in tri), "dur_scale": ds, "sigma": sigma,
                           "val_dtw_logmel_l1": d, "val_abs_log_dur_ratio": ldr, "val_dur_ratio": r}
                    if rec_m is not None:
                        hyps = transcribe(rec_m, rcfg, [bl.synth(tk)[0] for tk in vtoks])
                        rec["val_cer"] = sum(edit_distance(h, rf) for h, rf in zip(hyps, vrefs)) / sum(len(rf) for rf in vrefs)
                    grid.append(rec); print(json.dumps(rec), flush=True)
                    if best_dtw is None or d < best_dtw[0]["val_dtw_logmel_l1"]:
                        best_dtw = (rec, bl)
                    key = (rec.get("val_cer", 0.0), d)
                    if best is None or key < (best[0].get("val_cer", 0.0), best[0]["val_dtw_logmel_l1"]):
                        best = (rec, bl)
        print(f"K={K} done at {time.time() - t0:.0f}s", flush=True)
    rec, bl = best
    bl.save(a.out)
    best_dtw[1].save(a.out.replace(".npz", "_dtw.npz"))
    json.dump({"selected": rec, "selected_by_dtw_l1": best_dtw[0], "grid": grid, "n_train": len(txs), "n_val": len(vtoks),
               "iters": a.iters, "wall_s": time.time() - t0,
               "selection": "min val CER (CTC recognizer on the baseline mel), tie -> val DTW log-mel L1" if rec_m is not None
               else "min val DTW log-mel L1 (no recognizer given)"}, open(a.tune_out, "w"), indent=1)
    print("selected", rec, flush=True)
    print("selected_by_dtw_l1", best_dtw[0], flush=True)


if __name__ == "__main__":
    main()
