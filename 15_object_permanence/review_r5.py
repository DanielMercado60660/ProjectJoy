"""Claude reviewer diagnostics for round 5 turn 2 (NumPy only; reads saved traces).

  ../.venv/bin/python review_r5.py                 # tails/double-claims/strips, ~10 s
  ../.venv/bin/python review_r5.py --pf-scaling    # + PF particle-count sweep on hard_val[160:224], ~1 min
  ../.venv/bin/python review_r5.py --pf-sweep      # + 4096-particle q/miss/bandwidth search, ~1.5 min

Writes results/r5/review_diagnostics.json (+ review_pf_scaling.json / review_pf4096_sweep.json)
and media/r5_t2_review_*.png. Never reads a test split for selection; fresh traces are
only re-scored.
"""
import argparse
import itertools
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from r5_metrics import nll, moments
from r5_demo import panel
import metrics as old_metrics

TRACE = 'results/r5/t2_fresh_ac.npz'


def frame_masks(z):
    pos, vis, pres = z['pos'], z['vis'], z['present']
    mask = np.broadcast_to(pres[:, None, :], vis.shape)
    return pos, pres, mask, mask & (vis >= .999), mask & (vis == 0)


def dist_of(z, name, policy):
    if name == 'mixture':
        return tuple(z[f'mixture_{policy}_{k}'] for k in ('weights', 'modes', 'modecov'))
    m = z[f'{name}_{policy}_mean']
    return np.ones(m.shape[:3]+(1,)), m[..., None, :], z[f'{name}_{policy}_cov'][..., None, :, :]


def tails(z):
    pos, pres, mask, vis, hid = frame_masks(z)
    out = {}
    for name in ('gaussian', 'mixture', 'particle', 'kalman'):
        for policy in ('never', 'random', 'uncertainty'):
            m = z[f'{name}_{policy}_mean']
            err = np.linalg.norm(m-pos, axis=-1)
            ev = err[vis]
            row = dict(visible_error_mean=float(ev.mean()), visible_error_median=float(np.median(ev)),
                       visible_error_p99=float(np.quantile(ev, .99)), visible_frac_over_3px=float((ev > 3).mean()),
                       visible_mean_excluding_over_3px=float(ev[ev <= 3].mean()),
                       hidden_error_median=float(np.median(err[hid])))
            # Two present slots on one ball (<2 px apart) while the true balls are >10 px apart.
            dc = np.zeros(mask.shape[:2], bool)
            for i, j in itertools.combinations(range(3), 2):
                both = (pres[:, i] & pres[:, j])[:, None]
                dc |= both & (np.linalg.norm(m[:, :, i]-m[:, :, j], axis=-1) < 2) & \
                    (np.linalg.norm(pos[:, :, i]-pos[:, :, j], axis=-1) > 10)
            row.update(double_claim_frames=int(dc.sum()), double_claim_sequences=int(dc.any(1).sum()))
            if name in ('gaussian', 'mixture'):  # exact densities are stored only for the GRUs
                loss = nll(dist_of(z, name, policy), pos)
                lm = loss[mask]
                per_seq = np.array([loss[i][mask[i]].mean() for i in range(len(pos))])
                order = np.sort(lm)[::-1]
                row.update(all_nll_mean=float(lm.mean()), all_nll_median=float(np.median(lm)),
                           top_0p1pct_frames_share_of_total_nll=float(order[:len(order)//1000].sum()/lm.sum()),
                           frames_nll_over_50=int((lm > 50).sum()),
                           sequences_with_frame_nll_over_50=int(((loss > 50) & mask).any((1, 2)).sum()),
                           all_nll_mean_excluding_frames_over_50=float(lm[lm <= 50].mean()),
                           per_sequence_nll_median=float(np.median(per_seq)),
                           per_sequence_nll_trimmed10=float(np.sort(per_seq)[len(per_seq)//10:-(len(per_seq)//10)].mean()),
                           worst_sequences=[int(i) for i in np.argsort(per_seq)[::-1][:5]],
                           worst_sequence_nll=[float(v) for v in np.sort(per_seq)[::-1][:5]],
                           visible_nll_median=float(np.median(loss[vis])), hidden_nll_median=float(np.median(loss[hid])))
            out[f'{name}/{policy}'] = row
    return out


def strip(z, index, label, times, policy='never'):
    d = {k: z[k] for k in ('pos', 'present', 'occ', 'colors', 'vis', 'noise_seed')}
    methods = ('gaussian', 'mixture', 'particle')
    canvas = Image.new('RGB', (256*len(times), 256*len(methods)+44), '#111111')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f'R5 turn2_fresh sequence {index}: {label}; never-peek policy (review)', fill='white')
    draw.text((8, 23), 'circle=truth, +=mean; mixture=component 1-sigma ellipses; particle=moment ellipse incl. KDE', fill='white')
    for row, method in enumerate(methods):
        for col, t in enumerate(times):
            canvas.paste(panel(d, z, method, index, t, policy), (256*col, 44+256*row))
    target = Path(f'media/r5_t2_review_{label}_seq{index}.png')
    canvas.save(target)
    return str(target)


def pf_runs(configs, bandwidths, lo=160, hi=224):
    from r5_data import read_split, frames, cues, observation_seed
    from r5_classical import particle_track
    d = read_split('hard_val', hi)
    rows = []
    for cfg in configs:
        started = time.monotonic()
        ev, eh, sw, ess, per = [], [], [], [], {b: [] for b in bandwidths}
        for i in range(lo, hi):
            dist, diag = particle_track(frames(d, i), cues(d, [i])[0], d['present'][i], d['occ'][i], None,
                                        seed=observation_seed(d, i)+900, **cfg)
            gt, pr, vis = d['pos'][i], d['present'][i], d['vis'][i]
            msk = np.broadcast_to(pr, vis.shape)
            mean, _ = moments(dist)
            err = np.linalg.norm(mean-gt, axis=-1)
            ev.append(err[msk & (vis >= .999)])
            eh.append(err[msk & (vis == 0)])
            sw.append(old_metrics.id_switches(mean[None], gt[None], pr[None])['switches_per_seq'])
            ess.append(diag['min_ess_fraction'])
            w, m, c = dist
            for b in bandwidths:  # bandwidth only changes the output KDE, not tracking
                per[b].append(float(nll((w, m, np.broadcast_to(np.eye(2)*b*b, c.shape)), gt)[msk].mean()))
        ev, eh = np.concatenate(ev), np.concatenate(eh)
        rows.append(dict(config={k: v for k, v in cfg.items() if k != 'bandwidth'}, seconds=time.monotonic()-started,
                         visible_error=float(ev.mean()), visible_error_median=float(np.median(ev)),
                         visible_frac_over_3px=float((ev > 3).mean()), hidden_error=float(eh.mean()),
                         switches_per_seq=float(np.mean(sw)), min_ess_fraction_median=float(np.median(ess)),
                         per_seq_all_nll_by_bandwidth={str(b): float(np.mean(v)) for b, v in per.items()}))
        print(json.dumps(rows[-1]), flush=True)
    return dict(split='hard_val', indices=[lo, hi], rows=rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pf-scaling', action='store_true')
    p.add_argument('--pf-sweep', action='store_true')
    args = p.parse_args()
    with np.load(TRACE) as f:
        z = dict(f)
    out = dict(source=TRACE, tails=tails(z), per_sequence_nll={})
    # Robust per-sequence NLL summaries (each sequence weighted equally) from the evaluation JSONs.
    sources = [('results/r5/t2_fresh_ac.json', ('gaussian', 'mixture', 'particle', 'kalman'), ''),
               ('results/r5/review_pf4096_fresh.json', ('particle',), '_pf4096')]
    for path, names, suffix in sources:
        if not Path(path).exists():
            continue
        rows = json.loads(Path(path).read_text())['per_sequence']
        for name in names:
            for policy in ('never', 'random', 'uncertainty'):
                nl = np.sort([r['all']['nll'] for r in rows[name][policy]])
                hn = [r['hidden']['nll'] for r in rows[name][policy] if r['hidden']['n']]
                cut = len(nl)//10
                out['per_sequence_nll'][f'{name}{suffix}/{policy}'] = dict(
                    all_median=float(np.median(nl)), all_trimmed10=float(nl[cut:-cut].mean()),
                    all_max=float(nl[-1]), hidden_median=float(np.median(hn)))
    out['strips'] = [strip(z, 401, 'gaussian_overconfident', [0, 18, 24, 27, 30, 39]),
                     strip(z, 198, 'two_queries_one_ball', [0, 5, 9, 15, 21, 39])]
    Path('results/r5/review_diagnostics.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps({k: {kk: v[kk] for kk in ('visible_error_mean', 'visible_error_median', 'double_claim_sequences')}
                      for k, v in out['tails'].items() if k.endswith('never')}, indent=1))
    base = dict(r=1., miss=.35, bandwidth=4.)
    if args.pf_scaling:
        cfgs = [dict(base, q=.5, particles=512), dict(base, q=.5, particles=4096), dict(base, q=.15, particles=4096),
                dict(base, q=.05, particles=4096), dict(base, q=.5, particles=16384)]
        Path('results/r5/review_pf_scaling.json').write_text(json.dumps(pf_runs(cfgs, (.5, 1.5, 4.)), indent=2)+'\n')
    if args.pf_sweep:
        cfgs = [dict(q=q, r=1., miss=miss, particles=4096, bandwidth=1.) for q in (.15, .3, .5) for miss in (.2, .35)]
        Path('results/r5/review_pf4096_sweep.json').write_text(json.dumps(pf_runs(cfgs, (.5, 1., 1.5, 2.5, 4.)), indent=2)+'\n')


if __name__ == '__main__':
    main()
