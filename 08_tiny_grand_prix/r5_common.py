"""Round-5 IO, deadlines, and immutable split definitions. No MLX imports."""
import json
import os
import time
from pathlib import Path

import numpy as np

GROUPS = ('normal_patch', 'hard_patch', 'normal_switch', 'hard_switch', 'normal_static', 'hard_static')
SCENARIO_SEEDS = {'train': 51001, 'val': 52001, 'test': 2026102408, 'legacy': 53001}


def atomic_npz(path, **values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f, **values)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


class Budget:
    """Guard includes setup and finalization; no work starts without estimated room."""
    def __init__(self, minutes):
        if minutes <= 0:
            raise ValueError('--max-minutes must be positive')
        self.start = time.monotonic()
        self.seconds = 60 * minutes

    @property
    def elapsed(self):
        return time.monotonic() - self.start

    @property
    def remaining(self):
        return self.seconds - self.elapsed

    def fits(self, seconds=0, reserve=2):
        return self.remaining > seconds + reserve


def emit(record, path=None):
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'a') as f:
            f.write(line + '\n')
