"""CPU-only audit of immutable turn-1 receipts, serialized supervision and rewards."""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from r5_data import TRAIN_START, VAL_START, build_records, digest, json_write
from r5_protocol import action_ids, prompt_ids, unpack_example, verify_episode
from r5_world import ACTIONS, Env, Planner, features
from tokenizer import ChatTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=1.5)
    ap.add_argument("--train-examples", type=int, default=512, help="evenly spaced serialization checks; all val rows checked")
    a = ap.parse_args()
    assert 0 < a.max_minutes <= 2
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    started = time.monotonic()
    deadline = started + 60 * a.max_minutes
    tok, runs, datasets = ChatTokenizer.load(), {}, {}
    for tag in ("battery_s51", "ac_s52", "ac_s53"):
        root = Path("results/r5") / tag
        manifest = json.loads((root / "data/manifest.json").read_text())
        chosen = manifest["planner"]["selected"]
        for split, start in (("train", TRAIN_START), ("val", VAL_START)):
            sha = digest(root / "data" / f"{split}.npz")
            assert sha == manifest["counts"][split]["sha256"]
            if sha in datasets:
                continue
            records = build_records(Planner(chosen["room_order"], chosen["survey_first"]),
                                    start, manifest["counts"][split]["episodes"], deadline)
            with np.load(root / "data" / f"{split}.npz") as data:
                assert len(records) == len(data["targets"])
                indices = (np.unique(np.linspace(0, len(records) - 1, min(a.train_examples, len(records)), dtype=int))
                           if split == "train" else np.arange(len(records)))
                for i in indices:
                    r = records[int(i)]
                    if time.monotonic() >= deadline:
                        raise TimeoutError("audit time guard")
                    p, target = unpack_example(data["tokens"][i], data["masks"][i])
                    assert p == prompt_ids(tok, r["observation"]), (tag, split, i, "prompt")
                    assert target == action_ids(tok, r["action"]), (tag, split, i, "action/mask")
                    assert ACTIONS[data["targets"][i]] == r["action"]
                    np.testing.assert_array_equal(data["features"][i], features(r["observation"]))
            datasets[sha] = dict(source=str(root / "data" / f"{split}.npz"), examples=len(records),
                serialized_examples_checked=len(indices),
                delivery_counts=dict(Counter(len(r["observation"]["orders"]) for r in records)),
                first32_delivery_counts=dict(Counter(len(r["observation"]["orders"]) for r in records[:32])),
                illegal_feedback_examples=sum(r["observation"]["feedback"] == "illegal" for r in records))
        evaluations = {}
        for filename in ("eval.json", "secondary.json"):
            ev = json.loads((root / filename).read_text())
            for row in ev["episodes"]:
                if time.monotonic() >= deadline:
                    raise TimeoutError("audit time guard")
                verify_episode(row)
            evaluations[filename] = dict(complete=ev["complete"], paired_scenarios=ev["paired_scenarios"],
                methods={k: v["all"] for k, v in ev["summary"].items()}, verified_episode_receipts=len(ev["episodes"]))
        reward = json.loads((root / "reward/manifest.json").read_text())
        pointer, accepted, failed, illegal = 0, 0, 0, 0
        # These runs have <256 successes, so every success was retained in order.
        assert reward["successes"] == reward["accepted_episodes"] < 256
        with np.load(root / "reward/replay.npz") as data:
            for receipt in reward["receipts"]:
                if not receipt["success"]:
                    continue
                env = Env(receipt["seed"], receipt["split"], receipt["mode"])
                trace = []
                for _ in range(receipt["steps"]):
                    o = env.observe()
                    act = ACTIONS[data["targets"][pointer]]
                    p, target = unpack_example(data["tokens"][pointer], data["masks"][pointer])
                    assert p == prompt_ids(tok, o) and target == action_ids(tok, act)
                    np.testing.assert_array_equal(data["features"][pointer], features(o))
                    valid = env.step(act)
                    trace.append(dict(observation=o, action=act, valid=valid, feedback=env.feedback))
                    failed += not valid
                    illegal += env.feedback == "illegal"
                    pointer += 1
                verify_episode({**receipt, "trace": trace}, require_success=True)
                accepted += 1
        assert accepted == reward["accepted_episodes"] and pointer == reward["accepted_steps"]
        training = {}
        for method in ("pretrained", "scratch"):
            lines = [json.loads(line) for line in (Path("logs/r5") / tag / f"{method}.log").read_text().splitlines()
                     if line.startswith("{")]
            training[method] = dict(initial_val_loss=lines[0]["val_loss"], final=lines[-1],
                last_rate=next(r for r in reversed(lines) if r["event"] == "train"))
        runs[tag] = dict(evaluations=evaluations, reward=dict(attempted=reward["attempted"],
            verified_successes=accepted, verified_transitions=pointer, failed_actions=failed, illegal_actions=illegal),
            training=training, finished=json.loads((root / "finished.json").read_text()))
    result = dict(datasets=datasets, runs=runs, elapsed_s=time.monotonic() - started,
                  mlx_imported=False, audit_source_sha256=digest(__file__))
    json_write(a.out, result)
    print(json.dumps(dict(datasets=datasets, rewards={k: v["reward"] for k, v in runs.items()},
                          elapsed_s=result["elapsed_s"]), indent=2))


if __name__ == "__main__":
    main()
