"""Runner-only child-process SIGKILL/resume vs uninterrupted training check."""
import argparse
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    common = [sys.executable, '-u', 'r5_train.py', '--tiny', '--steps', '4', '--batch', '1',
              '--save-every', '2', '--val-count', '0', '--max-minutes', '1']
    resumed = out/'resumed'
    cmd = common+['--out', str(resumed), '--pause-at', '2']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    killed = False
    for line in process.stdout:
        print(line, end='', flush=True)
        if 'PAUSED_FOR_KILL_TEST step=2' in line:
            process.kill()
            killed = True
            break
    process.wait(timeout=10)
    assert killed and process.returncode == -signal.SIGKILL, 'must exercise an actual interrupted process'
    with (resumed/'last.pkl').open('rb') as f:
        saved = pickle.load(f)
    assert saved['step'] == 2
    subprocess.run(common+['--out', str(resumed), '--resume'], check=True)
    subprocess.run(common+['--out', str(out/'continuous')], check=True)
    with (resumed/'last.pkl').open('rb') as f:
        a = pickle.load(f)
    with (out/'continuous'/'last.pkl').open('rb') as f:
        b = pickle.load(f)
    assert a['step'] == b['step'] == 4 and a['numpy_rng'] == b['numpy_rng']
    largest = 0.
    for group in ('models', 'optimizers'):
        for model in a[group]:
            assert a[group][model].keys() == b[group][model].keys()
            for key in a[group][model]:
                av, bv = a[group][model][key], b[group][model][key]
                largest = max(largest, float(np.max(np.abs(av-bv))))
                assert np.allclose(av, bv, atol=2e-5, rtol=2e-5), (group, model, key, largest)
    for name in ('gaussian', 'mixture'):
        losses = [row['loss'][name] for row in a['history']]
        assert np.isfinite(losses).all() and len(losses) == 4
        assert int(a['optimizers'][name]['step']) == 4
        assert any(np.any(a['models'][name][k] != saved['models'][name][k]) for k in a['models'][name])
    print(f'RESUME_OK actual_SIGKILL step2 -> step4; weights/Adam/RNG match continuous; max_abs={largest:.3g}')


if __name__ == '__main__':
    main()
