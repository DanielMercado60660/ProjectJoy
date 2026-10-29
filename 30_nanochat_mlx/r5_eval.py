"""Observation-only policies, fresh paired evaluation, reward rollout collection, HTML.

Only syntactic command constraints are applied. No policy receives legal-action masks.
Evaluation rotates method order and includes only complete paired scenario blocks.
"""
import argparse
import html
import json
import time
from pathlib import Path
import numpy as np
from r5_data import (ROLLOUT_START, TEST_START, SECONDARY_START, arrays_from_records,
                     digest, json_write)
from r5_world import (ACTIONS, MODES, SPLITS, Env, Planner, factor_scores, features,
                      prompt, rollout, summarize)
from r5_protocol import ActionGrammar, prompt_ids, verify_episode


class LMPolicy:
    def __init__(self, model, tok, seed=0, temperature=0., use_cache=True):
        self.model, self.tok = model, tok
        self.rng, self.temperature = np.random.default_rng(seed), temperature
        self.grammar = ActionGrammar(tok)
        self.use_cache = use_cache

    def __call__(self, o):
        import mlx.core as mx
        p = prompt_ids(self.tok, o)
        cache = self.model.make_cache() if self.use_cache else None
        def scores(prefix, allowed):
            ids = (p if not prefix else [prefix[-1]]) if self.use_cache else p + list(prefix)
            logits = self.model(mx.array([ids], dtype=mx.int32), cache=cache)[0, -1]
            return np.array(logits[mx.array(allowed, dtype=mx.int32)])
        return self.grammar.decode(scores, self.rng, self.temperature)


class MLPPolicy:
    def __init__(self, model):
        self.model = model

    def __call__(self, o):
        import mlx.core as mx
        heads = self.model(mx.array(features(o)[None]))
        lp = [np.array(h[0] - mx.logsumexp(h[0])) for h in heads]
        return ACTIONS[int(np.argmax(factor_scores(lp)))]


def load_policy(path, seed=0, temperature=0.):
    import mlx.core as mx
    mx.set_memory_limit(14 * 1024**3)
    mx.set_cache_limit(1024**3)
    from r5_train import load_policy_model
    m, meta = load_policy_model(path)
    if meta["kind"] == "mlp":
        return MLPPolicy(m)
    from tokenizer import ChatTokenizer
    return LMPolicy(m, ChatTokenizer.load(), seed, temperature)


def successful_records(rows):
    # Complete terminal success is the only reward; no expert relabelling or test data.
    return [r for ep in rows if ep["success"] for r in ep["trace"]]


def collect(a):
    started = time.monotonic()
    deadline = started + a.max_minutes * 60 - min(3., a.max_minutes * 6)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError(out)
    policy = load_policy(a.checkpoint, a.seed, a.temperature)
    # A fixed cap controls disk/memory; reservoir retains at most 256 successful episodes.
    rng = np.random.default_rng(a.seed)
    successes, receipts, attempted, n_success = [], [], 0, 0
    while time.monotonic() < deadline and (not a.episodes or attempted < a.episodes):
        seed = ROLLOUT_START + attempted
        row = rollout(policy, Env(seed, "train", MODES[attempted % 4]), True, deadline)
        if row is None:
            break
        attempted += 1
        receipts.append({k: v for k, v in row.items() if k != "trace"})
        if row["success"]:
            n_success += 1
            if len(successes) < 256:
                successes.append(row)
            else:
                j = int(rng.integers(n_success))
                if j < 256:
                    successes[j] = row
        if attempted % 16 == 0:
            print(json.dumps(dict(event="collect", attempted=attempted, successes=n_success,
                                  elapsed_s=time.monotonic() - started)), flush=True)
    records = successful_records(successes)
    replay_path = None
    for row in successes:
        verify_episode(row, require_success=True)
    json_write(out / "accepted.json", successes)
    if records:
        from tokenizer import ChatTokenizer
        arrays = arrays_from_records(ChatTokenizer.load(), records, a.seq_len)
        np.savez(out / "replay.npz", **arrays)
        replay_path = str(out / "replay.npz")
    result = dict(args=vars(a), attempted=attempted, successes=n_success,
                  accepted_episodes=len(successes), accepted_steps=len(records), replay=replay_path,
                  policy_sha256=digest(Path(a.checkpoint) / "model.safetensors"),
                  accepted_trace_sha256=digest(out / "accepted.json"),
                  replay_sha256=digest(replay_path) if replay_path else None,
                  accepted_illegal_actions=sum(r["feedback"] == "illegal" for r in records),
                  accepted_failed_actions=sum(not r["valid"] for r in records),
                  receipts=receipts, elapsed_s=time.monotonic() - started,
                  rng_state=rng.bit_generator.state, policy_rng=policy.rng.bit_generator.state)
    json_write(out / "manifest.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("receipts", "args")}), flush=True)


def scenarios(episodes, secondary=False, seed_start=None):
    # Interleave all cells before second replicate, avoiding a prefix of only easy tasks.
    start = (SECONDARY_START if secondary else TEST_START) if seed_start is None else seed_start
    rng = np.random.default_rng(98317)
    for rep in range(episodes):
        cells = [(s, m) for s in SPLITS for m in MODES]
        for j in rng.permutation(len(cells)):
            split, mode = cells[int(j)]
            yield dict(seed=start + rep * 100 + int(j), split=split, mode=mode)


def render_demo(path, rows, summary):
    # Keep the first success and first failure for each method; never curate failures away.
    picked = {}
    for row in rows:
        picked.setdefault((row["method"], row["success"]), row)
    body = ["<!doctype html><meta charset='utf-8'><title>R5 Courier: revise the plan</title>",
            "<style>body{max-width:1100px;margin:35px auto;font:16px system-ui;background:#f4f3ee;color:#18242a}"
            "pre{white-space:pre-wrap;background:white;padding:18px;border-radius:8px}summary{cursor:pointer;padding:12px}</style>",
            "<h1>Courier: revise the plan</h1><p>Same observation notebook for every policy. "
            "Commands are grammar constrained, not legality constrained. First successes and failures retained. "
            "Provisional if the paired evaluation is incomplete.</p>",
            "<pre>" + html.escape(json.dumps(summary, indent=2)) + "</pre>"]
    for (method, success), row in picked.items():
        title = f"{method}: {'success' if success else 'failure'}; {row['split']}/{row['mode']}; seed {row['seed']}"
        lines = []
        for k, r in enumerate(row["trace"]):
            lines.append(f"Step {k+1}\n{prompt(r['observation'])}\n> {r['action']}"
                         f"\n{'valid' if r['valid'] else 'FAILED'}: {r['feedback']}\n")
        body.append("<details><summary>" + html.escape(title) + "</summary><pre>" +
                    html.escape("\n".join(lines)) + "</pre></details>")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(body))


def evaluate(a):
    started = time.monotonic()
    deadline = started + a.max_minutes * 60 - min(3., a.max_minutes * 6)
    target = Path(a.out)
    if target.exists():
        raise FileExistsError(target)
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    selected = manifest["planner"]["selected"]
    policies = {"planner": Planner(selected["room_order"], selected["survey_first"])}
    identities = {}
    for entry in a.methods:
        name, path = entry.split("=", 1)
        if name in policies:
            raise ValueError(f"duplicate method {name}")
        policies[name] = load_policy(path)
        meta = json.loads((Path(path) / "meta.json").read_text())
        identities[name] = dict(path=path, sha256=digest(Path(path) / "model.safetensors"),
                                step=meta["step"], seed=meta["seed"], spec=meta["spec"])
    rows, partial, blocks = [], [], 0
    for j, spec in enumerate(scenarios(a.episodes, a.secondary, a.seed_start)):
        block = []
        names = list(policies)
        names = names[j % len(names):] + names[:j % len(names)]
        for name in names:
            row = rollout(policies[name], Env(**spec), True, deadline)
            if row is None:
                break
            block.append(dict(method=name, **row))
        if len(block) != len(policies):
            partial = block
            break
        rows.extend(block)
        blocks += 1
        if blocks % 10 == 0:
            print(f"paired scenarios {blocks}/{a.episodes * 20}, {time.monotonic()-started:.1f}s", flush=True)
        if a.smoke_one and blocks == 1:
            break
    summary = {}
    for name in policies:
        summary[name] = {f"{s}/{m}": summarize([r for r in rows if r["method"] == name and r["split"] == s and r["mode"] == m])
                         for s in SPLITS for m in MODES}
        summary[name]["all"] = summarize([r for r in rows if r["method"] == name])
    result = dict(args=vars(a), complete=blocks == (1 if a.smoke_one else a.episodes * 20),
                  source_sha256={f: digest(f) for f in ("r5_world.py", "r5_protocol.py", "r5_eval.py")},
                  paired_scenarios=blocks, expected_scenarios=1 if a.smoke_one else a.episodes * 20,
                  elapsed_s=time.monotonic() - started, identities=identities,
                  summary=summary, episodes=rows, excluded_partial_block=partial)
    json_write(target, result)
    render_demo(a.demo, rows, dict(complete=result["complete"], paired_scenarios=blocks,
                                  metrics={k: v["all"] for k, v in summary.items()}))
    print(json.dumps(dict(event="evaluation", complete=result["complete"],
                          paired_scenarios=blocks, output=str(target))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--max-minutes", type=float, default=1.25)
    c.add_argument("--seed", type=int, default=51)
    c.add_argument("--temperature", type=float, default=.7)
    c.add_argument("--seq-len", type=int, default=384)
    c.add_argument("--episodes", type=int, default=0, help="0 fills time allowance; positive only for smoke")
    e = sub.add_parser("evaluate")
    e.add_argument("--methods", nargs="+", required=True, help="name=selected-checkpoint-path")
    e.add_argument("--data", default="results/r5/data")
    e.add_argument("--out", required=True)
    e.add_argument("--demo", default="media/r5_demo.html")
    e.add_argument("--max-minutes", type=float, default=4)
    e.add_argument("--episodes", type=int, default=4, help="replicates per split/disruption cell")
    e.add_argument("--secondary", action="store_true", help="old seed range, new environment; NOT old-task comparability")
    e.add_argument("--seed-start", type=int, default=None, help="explicit fresh or secondary scenario range")
    e.add_argument("--smoke-one", action="store_true")
    a = ap.parse_args()
    assert 0 < a.max_minutes <= 45
    if a.command == "collect" and a.temperature <= 0:
        raise ValueError("reward collection needs a positive sampling temperature")
    collect(a) if a.command == "collect" else evaluate(a)


if __name__ == "__main__":
    main()
