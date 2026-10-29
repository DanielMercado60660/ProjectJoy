"""Claude-only hard-kill/resume check on an owned tiny training subprocess."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np


def compare(a_path, b_path):
    with np.load(a_path, allow_pickle=False) as a, np.load(b_path, allow_pickle=False) as b:
        for k in a.files:
            if k.endswith('active_seconds'):
                continue
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--max-minutes', type=float, default=1.)
    a = ap.parse_args()
    deadline = time.monotonic() + a.max_minutes * 60
    base = [sys.executable, 'train_r5.py', '--envs', '8', '--horizon', '8', '--unroll', '4',
            '--minibatch-seqs', '8', '--epochs', '1', '--n-train', '2', '--n-val', '1',
            '--episode-steps', '80', '--eval-every', '1', '--save-every', '1', '--max-minutes', '.6']
    path = Path(f'checkpoints/{a.tag}_state.npz')
    if path.exists():
        raise ValueError('kill-test tag must be fresh')
    logpath = Path(f'logs/r5/{a.tag}_kill.txt')
    with logpath.open('w') as log:
        child = subprocess.Popen(base + ['--tag', a.tag], stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.monotonic() < deadline - 10:
                if child.poll() is not None:
                    raise RuntimeError(f'child exited before kill; inspect {logpath}')
                if path.exists():
                    with np.load(path) as z:
                        ready = int(z['update']) >= 1
                    if ready:
                        # Only the explicitly owned process is killed, never a shared GPU process.
                        os.kill(child.pid, signal.SIGKILL)
                        child.wait(timeout=3)
                        break
                time.sleep(.02)
            else:
                raise TimeoutError('kill-test did not checkpoint within allowance')
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=3)
    with np.load(path) as z:
        saved_step = int(z['update'])
    target = saved_step + 1
    for command in [base + ['--tag', a.tag, '--resume', str(path), '--max-updates', str(target)],
                    base + ['--tag', a.tag + '_whole', '--max-updates', str(target)]]:
        subprocess.run(command, check=True, timeout=max(1, deadline - time.monotonic()))
    compare(path, f'checkpoints/{a.tag}_whole_state.npz')
    print(f'HARD KILL / RESUME PASS: checkpoint {saved_step}, continued to {target}, exact state match', flush=True)


if __name__ == '__main__':
    main()
