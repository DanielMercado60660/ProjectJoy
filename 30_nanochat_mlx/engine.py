"""
Inference with a KV cache (port of the core of upstream nanochat/engine.py, batch size 1).

Also: `python engine.py --ckpt checkpoints/pilot/last` measures first-token latency (prefill)
and decode tokens/sec at a few prompt lengths.
"""
import argparse
import json
import os
import time

import mlx.core as mx
import numpy as np


def _sample(logits, temperature, top_k, rng):
    if temperature <= 0:
        return int(mx.argmax(logits).item())
    logits = np.array(logits.astype(mx.float32)) / temperature
    if top_k:
        kth = np.sort(logits)[-top_k]
        logits[logits < kth] = -np.inf
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return int(rng.choice(len(p), p=p))


def generate(model, ids, max_tokens=64, temperature=0.0, top_k=None, seed=42, stop_ids=(), allowed_fn=None):
    """Yield new token ids. Prefill the prompt once, then decode one token at a time from the cache.
    allowed_fn(generated_so_far) -> list of allowed next ids (or None), for constrained decoding."""
    rng = np.random.default_rng(seed)
    cache = model.make_cache()
    logits = model(mx.array([ids]), cache=cache)[0, -1]
    out = []
    for _ in range(max_tokens):
        if allowed_fn is not None:
            allowed = allowed_fn(out)
            if allowed is not None:
                m = np.full(logits.shape[-1], -np.inf, dtype=np.float32)
                m[allowed] = 0.0
                logits = logits + mx.array(m)
        nxt = _sample(logits, temperature, top_k, rng)
        out.append(nxt)
        yield nxt
        if nxt in stop_ids:
            return
        logits = model(mx.array([[nxt]]), cache=cache)[0, -1]


def generate_naive(model, ids, max_tokens=16):
    """No cache: re-run the full sequence each step (used to verify the cache and as a speed baseline)."""
    ids = list(ids)
    out = []
    for _ in range(max_tokens):
        logits = model(mx.array([ids]))[0, -1]
        nxt = int(mx.argmax(logits).item())
        ids.append(nxt)
        out.append(nxt)
    return out


def bench(model, prompt_lens=(16, 128, 512), gen_tokens=128, reps=3):
    res = []
    V = model.config.vocab_size
    rng = np.random.default_rng(0)
    # warmup
    list(generate(model, [int(t) for t in rng.integers(0, V - 16, 8)], max_tokens=8))
    for L in prompt_lens:
        ttft, dec = [], []
        for r in range(reps):
            ids = [int(t) for t in rng.integers(0, V - 16, L)]
            t0 = time.perf_counter()
            g = generate(model, ids, max_tokens=gen_tokens)
            next(g)
            t1 = time.perf_counter()
            n = 1 + sum(1 for _ in g)
            t2 = time.perf_counter()
            ttft.append(t1 - t0)
            dec.append((n - 1) / (t2 - t1))
        t0 = time.perf_counter()
        generate_naive(model, [int(t) for t in rng.integers(0, V - 16, L)], max_tokens=32)
        naive = 32 / (time.perf_counter() - t0)
        res.append({"prompt_len": L, "first_token_ms": 1000 * float(np.median(ttft)),
                    "decode_tok_per_s": float(np.median(dec)), "naive_no_cache_tok_per_s": naive})
    return res


if __name__ == "__main__":
    import checkpoint as ckpt
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pilot/last")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, meta = ckpt.load_model(a.ckpt)
    r = bench(model)
    print(json.dumps(r, indent=1))
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
