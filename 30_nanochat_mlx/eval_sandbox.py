"""
Evaluate policies on Grid Courier: action validity, task success, steps, decision latency.

Splits (fixed seeds, disjoint from training seeds):
  test_seen : new layouts, tasks built only from training delivery pairs
  test      : held-out task combinations (>= 1 delivery pair never seen in training)

Policies:
  expert         scripted optimal policy (upper bound)
  random         uniform over currently-legal actions (lower bound)
  mlp            non-language baseline (baseline.py)
  lm             SFT language model, free greedy generation (KV cache)
  lm_constrained SFT model, output restricted to the 28-string action grammar
                 (argmax of the model's log-prob over the 28 action strings)

    ../bin/gpu_run ../.venv/bin/python eval_sandbox.py --ckpt checkpoints/sft/last --episodes 200
"""
import argparse
import json
import os
import random
import time

import mlx.core as mx
import numpy as np

from sandbox import ACTIONS, MAX_STEPS, TEST3_SEED0, TEST_SEED0, TESTSEEN_SEED0, expert_action, make_env, parse_action

SPLITS = {"test_seen": TESTSEEN_SEED0, "test": TEST_SEED0, "test3": TEST3_SEED0}


class LMPolicy:
    def __init__(self, model, tok, constrained=False):
        self.model, self.tok, self.constrained = model, tok, constrained
        self.end = tok.encode_special("<|assistant_end|>")
        if constrained:
            self.cands = [tok.encode(a) + [self.end] for a in ACTIONS]
            L = max(len(c) for c in self.cands)
            self.cand_arr = np.zeros((len(ACTIONS), L), dtype=np.int32)
            self.cand_mask = np.zeros((len(ACTIONS), L), dtype=np.float32)
            for k, c in enumerate(self.cands):
                self.cand_arr[k, :len(c)] = c
                self.cand_mask[k, :len(c)] = 1

    def __call__(self, env):
        from engine import generate
        prompt = self.tok.render_for_completion([{"role": "user", "content": env.observation()}])
        if not self.constrained:
            out = list(generate(self.model, prompt, max_tokens=12, temperature=0.0, stop_ids=(self.end,)))
            if out and out[-1] == self.end:
                out = out[:-1]
            return self.tok.decode(out)
        # score every action string: prefill prompt once, then batch the candidates on a copied cache
        P = len(prompt)
        n, L = self.cand_arr.shape
        seq = np.concatenate([np.tile(np.array(prompt, dtype=np.int32), (n, 1)), self.cand_arr], axis=1)
        logits = self.model(mx.array(seq[:, :-1]))  # (n, P+L-1, V)
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tgt = mx.array(seq[:, P:])  # candidate tokens
        lp = mx.take_along_axis(logp[:, P - 1:], tgt[..., None], axis=-1)[..., 0]
        score = (np.array(lp) * self.cand_mask).sum(-1)
        return ACTIONS[int(np.argmax(score))]


class MLPPolicy:
    def __init__(self, path):
        from baseline import load
        self.m = load(path)

    def __call__(self, env):
        return ACTIONS[int(mx.argmax(self.m(mx.array(env.features()[None]))[0]).item())]


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, env):
        return self.rng.choice(env.legal_actions())


def run_episode(policy, env, transcript=False):
    valid, n, lat = 0, 0, []
    lines = [env.observation()] if transcript else None
    while not env.over():
        t0 = time.perf_counter()
        raw = policy(env)
        lat.append(time.perf_counter() - t0)
        act = parse_action(raw) if isinstance(raw, str) else None
        if act is None:  # malformed output: counts as an invalid action and uses up a step
            env.steps += 1
            ok = False
        else:
            ok = env.step(act)
        valid += int(ok)
        n += 1
        if transcript:
            lines.append(f"> {raw!r}" + ("" if ok else "   [INVALID]"))
            if ok:
                lines.append(env.observation())
    return {"success": env.success(), "steps": n, "valid": valid, "latency": lat, "transcript": lines}


def optimal_steps(seed, split):
    env = make_env(seed, split)
    k = 0
    while not env.success():
        env.step(expert_action(env))
        k += 1
    return k


def evaluate(policy, episodes, split):
    s0 = SPLITS[split]
    res = [run_episode(policy, make_env(s0 + i, split)) for i in range(episodes)]
    opt = [optimal_steps(s0 + i, split) for i in range(episodes)]
    lat = np.concatenate([r["latency"] for r in res])
    succ = [r for r, o in zip(res, opt) if r["success"]]
    return {
        "episodes": episodes,
        "success_rate": float(np.mean([r["success"] for r in res])),
        "action_validity": float(sum(r["valid"] for r in res) / max(1, sum(r["steps"] for r in res))),
        "mean_steps": float(np.mean([r["steps"] for r in res])),
        "steps_over_optimal_when_success": float(np.mean([r["steps"] / o for r, o in zip(res, opt) if r["success"]])) if succ else None,
        "decision_latency_ms_median": float(1000 * np.median(lat)),
        "decision_latency_ms_p95": float(1000 * np.percentile(lat, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/sft/last")
    ap.add_argument("--mlp", default="checkpoints/baseline_mlp")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--policies", default="expert,random,mlp,lm,lm_constrained")
    ap.add_argument("--out", default="media/sandbox_eval.json")
    ap.add_argument("--tag", default="")
    ap.add_argument("--splits", default="test_seen,test")
    a = ap.parse_args()
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    model = tok = None
    for name in a.policies.split(","):
        if name == "expert":
            pol = expert_action
        elif name == "random":
            pol = RandomPolicy(0)
        elif name == "mlp":
            pol = MLPPolicy(a.mlp)
        else:
            if model is None:
                import checkpoint as ckpt
                from tokenizer import ChatTokenizer
                model, _ = ckpt.load_model(a.ckpt)
                tok = ChatTokenizer.load()
            pol = LMPolicy(model, tok, constrained=(name == "lm_constrained"))
        key = name + (f"[{a.tag}]" if a.tag else "")
        results.setdefault(key, {})
        for split in a.splits.split(","):
            t0 = time.time()
            r = evaluate(pol, a.episodes, split)
            results[key][split] = r
            print(f"{key:28s} {split:10s} success {r['success_rate']:.3f} validity {r['action_validity']:.3f} "
                  f"steps {r['mean_steps']:.1f} lat {r['decision_latency_ms_median']:.1f}ms ({time.time()-t0:.0f}s)", flush=True)
        json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
