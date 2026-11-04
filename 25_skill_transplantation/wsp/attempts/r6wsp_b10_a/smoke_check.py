"""Synthetic-only checks; this script never imports or invokes an environment."""
import json
import hashlib
from pathlib import Path
import build
import numpy as np

HERE = Path(__file__).resolve().parent
EVALUATED = {0: 'derivative_motion_s0.npz', 1: 'shared_motion_s1.npz',
             2: 'shared_motion_s2.npz', 3: 'derivative_motion_s3.npz'}
report = []
for seed in range(4):
    final_path = HERE / f'merged_s{seed}.npz'
    with np.load(final_path) as z:
        assert set(z.files) == set(build.KEYS)
        p = {key: z[key].copy() for key in z.files}
    for key, shape in zip(build.KEYS, build.SHAPES):
        assert p[key].dtype == np.float32 and p[key].shape == shape
        assert np.isfinite(p[key]).all()
    rebuilt, _ = build.compress(seed, output_fit=build.FINAL_OUTPUT_FITS[seed])
    evaluated = np.load(HERE / EVALUATED[seed])
    for key in build.KEYS:
        assert np.array_equal(p[key], rebuilt[key]), (seed, key, 'rebuild differs')
        assert np.array_equal(p[key], evaluated[key]), (seed, key, 'evaluated weights differ')
    rng = np.random.default_rng(99000 + seed)
    obs = rng.uniform(-10., 10., size=(10000, 17)).astype(np.float32)
    margins, actions = [], []
    for task in range(2):
        x = np.zeros((len(obs), 19), np.float32)
        x[:, :17], x[:, 17 + task] = obs, 1.
        h1 = np.tanh(x @ p['a1.weight'].T + p['a1.bias'])
        z2 = h1 @ p['a2.weight'].T + p['a2.bias']
        h2 = np.tanh(z2)
        inactive = slice(63, 127) if task == 0 else slice(0, 63)
        assert np.min(z2[:, inactive]) >= 12. - 1e-4
        assert np.max(np.abs(h2[:, inactive] - 1.)) <= 1e-7
        a = np.tanh(h2 @ p['a3.weight'].T + p['a3.bias'])
        assert a.shape == (10000, 6) and np.isfinite(a).all()
        assert np.max(np.abs(a)) <= 1.
        assert np.max(np.abs(a - build.actor(p, obs, task))) < 1e-4
        margins.append(float(np.min(z2[:, inactive])))
        actions.append(a)
    report.append({'seed': seed, 'sha256': hashlib.sha256(final_path.read_bytes()).hexdigest(),
                   'rebuild_bitwise_equal': True, 'evaluated_candidate_bitwise_equal': True,
                   'minimum_inactive_preactivations': margins,
                   'same_input_task_action_rms_difference': float(np.sqrt(np.mean((actions[0] - actions[1]) ** 2))),
                   'parameters_including_log_std': sum(v.size for v in p.values())})
print(json.dumps({'status': 'PASS', 'synthetic_only': True, 'seeds': report}, indent=2))
