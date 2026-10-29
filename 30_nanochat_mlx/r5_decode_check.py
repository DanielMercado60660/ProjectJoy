"""Runner-only MLX gate: stored SFT bytes, padded teacher forcing and cached decode.

Prediction correctness is reported, not required of an untrained/weak model. The
gate checks identical prefixes and numerical parity; stitched teacher-forced argmax
tokens need not form a legal sentence when an earlier predicted token was wrong.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import mlx.core as mx
from r5_data import VAL_START, build_records, digest, json_write
from r5_eval import LMPolicy
from r5_protocol import prompt_ids, action_ids, unpack_example
from r5_train import load_policy_model
from r5_world import Planner
from tokenizer import ChatTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--examples", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=.5)
    a = ap.parse_args()
    assert 0 < a.max_minutes <= 1
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    started = time.monotonic()
    deadline = started + a.max_minutes * 60
    manifest = json.loads((Path(a.data) / "manifest.json").read_text())
    choice = manifest["planner"]["selected"]
    records = build_records(Planner(choice["room_order"], choice["survey_first"]), VAL_START,
                            manifest["counts"]["val"]["episodes"], deadline)
    tok = ChatTokenizer.load()
    model, meta = load_policy_model(a.checkpoint)
    assert meta["kind"] == "lm"
    checks, errors = [], []
    with np.load(Path(a.data) / "val.npz") as data:
        indices = np.unique(np.linspace(0, len(records) - 1, a.examples, dtype=int))
        for dtype, atol, rtol in ((mx.float32, .02, .002), (mx.bfloat16, .25, .02)):
            model.compute_dtype = dtype
            cached_policy, full_policy = LMPolicy(model, tok), LMPolicy(model, tok, use_cache=False)
            for idx in indices:
                if time.monotonic() >= deadline:
                    errors.append("wall guard before all cases completed")
                    break
                r = records[int(idx)]
                p, target = unpack_example(data["tokens"][idx], data["masks"][idx])
                assert p == prompt_ids(tok, r["observation"])
                assert target == action_ids(tok, r["action"])
                positions = np.flatnonzero(data["masks"][idx])
                # Padded batch path used by SFT, compared at the shifted action positions.
                full = np.array(model(mx.array(data["tokens"][idx:idx+1, :-1].astype(np.int32)))[
                    0, mx.array(positions, dtype=mx.int32)])
                cache = model.make_cache()
                logit = model(mx.array([p], dtype=mx.int32), cache=cache)[0, -1]
                cached = []
                for k in range(len(target)):
                    cached.append(np.array(logit))
                    if k + 1 < len(target):
                        logit = model(mx.array([[target[k]]], dtype=mx.int32), cache=cache)[0, -1]
                cached = np.stack(cached)
                same_logits = bool(np.allclose(full, cached, atol=atol, rtol=rtol))
                top2 = np.partition(full, -2, axis=-1)[:, -2:]
                margin = float(np.min(top2[:, 1] - top2[:, 0]))
                teacher_correct = bool(np.array_equal(full.argmax(-1), target))
                cache_action, full_action = cached_policy(r["observation"]), full_policy(r["observation"])
                high_confidence_failure = teacher_correct and margin > 2 * atol and (
                    cache_action != r["action"] or full_action != r["action"])
                checks.append(dict(row=int(idx), dtype=str(dtype), target=r["action"],
                     teacher_forced_target_exact=teacher_correct, teacher_min_margin=margin,
                     cached_action=cache_action, full_action=full_action,
                     logits_close=same_logits, max_abs_logit_diff=float(np.abs(full-cached).max()),
                     high_confidence_roundtrip_failure=high_confidence_failure))
                if not same_logits or high_confidence_failure:
                    errors.append(f"row {idx}, {dtype}: prefix/logit/round-trip mismatch")
    result = dict(checkpoint=a.checkpoint, checkpoint_sha256=digest(Path(a.checkpoint) / "model.safetensors"),
                  expected=2 * len(indices), checks=checks, errors=errors,
                  elapsed_s=time.monotonic() - started,
                  note="Teacher-forced target accuracy is a model metric, not a format assertion.")
    json_write(a.out, result)
    if errors or len(checks) != result["expected"]:
        raise RuntimeError(f"decode gate failed; preserved receipt: {a.out}")
    print(f"R5 DECODE CHECK OK: {len(checks)} cases; {a.out}")


if __name__ == "__main__":
    main()
