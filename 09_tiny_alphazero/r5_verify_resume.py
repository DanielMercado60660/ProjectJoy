"""Claude-only exact resume check, including a zero-input finite-gradient case."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten
from r5_mlx import Trainer


def main(directory):
    root = Path(directory)
    assert not (root/"active.json").exists(), "Use a fresh smoke directory"
    x = np.zeros((16, 117), np.float32)
    x[8:, 0] = 1
    y = np.r_[np.zeros(8), np.ones(8)].astype(np.float32)
    cfg = dict(test="resume-v1")
    t = Trainer(root, cfg, np.zeros(117), np.ones(117))
    t.best = 1.
    t.train(0, x, y, 3, 16, time.monotonic()+20); t.save()
    loss1 = t.train(0, x, y, 4, 16, time.monotonic()+20)
    expected = {k: np.array(v) for k, v in tree_flatten(t.models[0].parameters())}
    random_expected = np.array(mx.random.uniform(shape=(8,)))
    resumed = Trainer(root, cfg, np.zeros(117), np.ones(117))
    loss2 = resumed.train(0, x, y, 4, 16, time.monotonic()+20)
    assert resumed.steps[0] == 7 and np.isfinite(loss2)
    assert abs(loss1-loss2) < 1e-6
    for k, v in tree_flatten(resumed.models[0].parameters()):
        np.testing.assert_allclose(np.array(v), expected[k], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(np.array(mx.random.uniform(shape=(8,))), random_expected)
    result = dict(complete=True, optimizer_weights_rng_match=True, step=7, loss=loss2)
    (root/"verified.json").write_text(json.dumps(result))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
