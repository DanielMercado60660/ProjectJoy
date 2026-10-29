import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpt import GPT, config_for_depth  # noqa: E402
from engine import generate, generate_naive  # noqa: E402
import sandbox  # noqa: E402


def _model():
    m = GPT(config_for_depth(2, 512, 64))
    m.init_weights(0)
    for b in m.h:  # zero-init output projections make the model trivial; perturb them
        b.attn.c_proj.weight = mx.random.normal(b.attn.c_proj.weight.shape) * 0.02
        b.mlp.c_proj.weight = mx.random.normal(b.mlp.c_proj.weight.shape) * 0.02
    m.lm_head.weight = mx.random.normal(m.lm_head.weight.shape) * 0.05
    m.compute_dtype = mx.float32
    return m


def test_causal():
    m = _model()
    x = np.random.default_rng(0).integers(0, 500, (1, 32))
    a = m(mx.array(x))
    x2 = x.copy()
    x2[0, 20] = (x2[0, 20] + 1) % 500
    b = m(mx.array(x2))
    assert float(mx.abs(a[0, :20] - b[0, :20]).max()) < 1e-5  # past unaffected by the future
    assert float(mx.abs(a[0, 20:] - b[0, 20:]).max()) > 1e-4


def test_kv_cache_matches_naive():
    m = _model()
    ids = [int(i) for i in np.random.default_rng(1).integers(0, 500, 12)]
    assert generate_naive(m, ids, 8) == list(generate(m, ids, 8))


def test_sandbox_expert_solves_all():
    for s in range(200):
        for split in ["train", "test", "test_seen"]:
            env = sandbox.make_env(s, split)
            sandbox.expert_trace(env)
            assert env.success()


def test_heldout_pairs_absent_from_train():
    for s in range(500):
        env = sandbox.make_env(s, "train")
        assert not any(g in sandbox.HELDOUT_PAIRS for g in env.goals)
        env = sandbox.make_env(sandbox.TEST_SEED0 + s, "test")
        assert any(g in sandbox.HELDOUT_PAIRS for g in env.goals)


def test_muon_step_runs_and_reduces_loss():
    from optim import build_optimizer
    from gpt import loss_fn
    import mlx.nn as nn
    m = GPT(config_for_depth(2, 512, 32))
    m.init_weights(0)
    opt, _ = build_optimizer(m, weight_decay=0.0)
    x = mx.array(np.random.default_rng(0).integers(0, 500, (4, 32)))
    lg = nn.value_and_grad(m, loss_fn)
    l0 = None
    for _ in range(20):
        loss, g = lg(m, x, x)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
        l0 = l0 or loss.item()
    assert loss.item() < l0 - 1.0


def test_token_bytes_match_raw_utf8_bytes():
    # review: bits/byte needs per-token byte counts that sum to the raw text's UTF-8 length,
    # including tokens that are partial multi-byte characters
    import json
    from tokenizer import ChatTokenizer, TOKENIZER_DIR
    tok = ChatTokenizer.load()
    tb = json.load(open(os.path.join(TOKENIZER_DIR, "token_bytes.json")))
    assert tb == tok.token_bytes()
    text = "naïve café — 日本語のテキスト 🙂 Ελληνικά ∑∫ plain ascii too"
    ids = tok.encode(text, prepend_bos=False)
    assert sum(tb[i] for i in ids) == len(text.encode("utf-8"))
