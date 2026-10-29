"""Headless round-5 PNG strips from evaluation traces produced by checkpoint reload."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw
from r5_data import frames

COLORS = ['#ff55cc', '#00e5ff', '#55ff55']


def panel(d, trace, method, index, t, policy):
    key = f'{method}_{policy}'
    look = tuple(int(v) for v in trace[key+'_look'][index])
    if look[0] < 0:
        look = None
    # Noise is regenerated from the original split index; same as evaluation.
    f = frames(d, index, look)[t]
    im = Image.fromarray(np.uint8(f*255)).resize((256, 256))
    draw = ImageDraw.Draw(im)
    for slot in np.flatnonzero(d['present'][index]):
        color = COLORS[slot]
        x, y = d['pos'][index, t, slot]*4
        draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=1)
        mu = trace[key+'_mean'][index, t, slot]*4
        draw.line((mu[0]-5, mu[1], mu[0]+5, mu[1]), fill=color, width=2)
        draw.line((mu[0], mu[1]-5, mu[0], mu[1]+5), fill=color, width=2)
        if method == 'mixture':
            weights = trace[key+'_weights'][index, t, slot]
            means = trace[key+'_modes'][index, t, slot]
            covs = trace[key+'_modecov'][index, t, slot]
        else:
            weights = [1.]
            means = trace[key+'_mean'][index, t, slot][None]
            covs = trace[key+'_cov'][index, t, slot][None]
        for w, mean, cov in zip(weights, means, covs):
            if w < .05:
                continue
            eig, vec = np.linalg.eigh(cov)
            theta = np.linspace(0, 2*np.pi, 48)
            ring = (np.stack([np.cos(theta), np.sin(theta)], -1)*np.sqrt(np.maximum(eig, 0))) @ vec.T
            xy = np.clip((ring+mean)*4, -1024, 1280)
            draw.line([tuple(v) for v in xy], fill=color, width=1)
    draw.rectangle((0, 0, 255, 17), fill='black')
    draw.text((4, 3), f'{method} t={t}'+(' LOOK' if look and t == look[0] else ''), fill='white')
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', default='media/r5_demo')
    p.add_argument('--max-minutes', type=float, default=1.)
    args = p.parse_args()
    start = time.monotonic()
    result = json.loads(Path(args.evaluation).read_text())
    with np.load(Path(args.evaluation).with_suffix('.npz')) as z:
        trace = dict(z)
    d = {k: trace[k] for k in ('pos', 'present', 'occ', 'colors', 'vis', 'noise_seed')}
    methods = [m for m in ('gaussian', 'mixture', 'particle', 'kalman') if m in result['summary']]
    rows = result['per_sequence'].get('mixture', result['per_sequence']['particle'])['uncertainty']
    worst = int(np.argmin([row['peek']['later_nll_gain'] for row in rows]))
    for label, index in [('fixed_sequence0', 0), ('least_helpful_look', worst)]:
        if time.monotonic()-start >= args.max_minutes*60:
            raise RuntimeError('demo guard expired')
        # Include the selected look itself and its immediate aftermath.
        peek_method = 'mixture' if 'mixture' in methods else 'particle'
        tlook = int(trace[f'{peek_method}_uncertainty_look'][index, 0])
        times = sorted(set([0, max(0, tlook-1), tlook, tlook+1, 31, 39]))
        canvas = Image.new('RGB', (256*len(times), 256*len(methods)+44), '#111111')
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f'R5 {result["split"]} sequence {index}: {label}; uncertainty policy, one look per tracker', fill='white')
        draw.text((8, 23), 'circle=truth, +=mean; mixture=component 1-sigma ellipses (weight>=.05); others=moment ellipse', fill='white')
        for row, method in enumerate(methods):
            for col, t in enumerate(times):
                canvas.paste(panel(d, trace, method, index, t, 'uncertainty'), (256*col, 44+256*row))
        target = Path(args.out+'_'+label+'.png')
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        canvas.save(target)
        print(f'DEMO {target} sequence={index} selection={label}')


if __name__ == '__main__':
    main()
