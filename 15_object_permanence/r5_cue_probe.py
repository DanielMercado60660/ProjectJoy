"""Runner-only val cue interventions. These are diagnostics, not headline test scores."""
import argparse
import hashlib
from pathlib import Path
import time
import numpy as np
from r5_data import read_split, frames, cues
from r5_evaluate import neural_track
from r5_metrics import moments, nll
from r5_tune import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--n', type=int, default=32)
    p.add_argument('--start', type=int, default=384)
    p.add_argument('--max-minutes', type=float, required=True)
    p.add_argument('--require-equivariant', action='store_true')
    args = p.parse_args()
    if not 0 < args.max_minutes <= 35 or args.n < 1 or args.start < 384 or args.start+args.n > 1000:
        p.error('positive bounded job; reserve val[384:1000] for cue probes')
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    from r5_load import load_pair
    start = time.monotonic()
    models, state = load_pair(args.checkpoint)
    data = read_split('hard_val', args.start+args.n)
    rows = {name: [] for name in models}
    for i in range(args.start, args.start+args.n):
        if time.monotonic()-start >= args.max_minutes*60-5:
            break
        f, cue = frames(data, i), cues(data, [i])[0]
        present, occ = data['present'][i], data['occ'][i]
        order = np.array([2, 0, 1])
        null_cue = cue.copy()
        null_cue[present] = cue[present].mean(0)
        for name, model in models.items():
            full, _ = neural_track(model, f, cue, present, occ)
            swapped, _ = neural_track(model, f, cue[order], present[order], occ)
            null, _ = neural_track(model, f, null_cue, present, occ)
            mean, _ = moments(full)
            sm, _ = moments(swapped)
            nm, _ = moments(null)
            equiv = np.linalg.norm(sm-mean[:, order], axis=-1)[:, present[order]]
            losses = nll(full, data['pos'][i])[:, present]
            null_losses = nll(null, data['pos'][i])[:, present]
            rows[name].append(dict(index=i, equivariance_mean_px=float(equiv.mean()),
                equivariance_max_px=float(equiv.max()),
                cue_response_after_frame3_px=float(np.linalg.norm(nm[4:]-mean[4:], axis=-1)[:, present].mean()),
                null_minus_full_nll=float((null_losses-losses).mean()), full_nll=float(losses.mean())))
    if not rows['gaussian']:
        raise RuntimeError('guard expired before a complete probe sequence')
    summary = {name: {key: float(np.mean([r[key] for r in rs])) for key in rs[0] if key != 'index'}
               for name, rs in rows.items()}
    maximum = max(r['equivariance_max_px'] for rs in rows.values() for r in rs)
    report = dict(checkpoint=args.checkpoint, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                  paired_step=state['step'], split='hard_val', requested_n=args.n, completed_n=len(rows['gaussian']),
                  complete=len(rows['gaussian']) == args.n, summary=summary, per_sequence=rows,
                  elapsed_seconds=time.monotonic()-start,
                  note='Mean-cue ablation is OOD diagnostic only; permutation is a valid identity relabeling.')
    write_json(args.out, report)
    if args.require_equivariant and maximum > 2e-3:
        raise AssertionError(f'query equivariance broken: max difference {maximum} px')
    print(f'CUE_PROBE_OK step={state["step"]} n={report["completed_n"]} max_equivariance_px={maximum:.6g} summary={summary}')


if __name__ == '__main__':
    main()
