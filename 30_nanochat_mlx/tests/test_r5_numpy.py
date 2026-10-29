"""The R5 local test gate: stdlib, NumPy, pytest only; never import MLX."""
import copy
import sys
import subprocess
import time
import numpy as np
import pytest
from r5_world import (ACTIONS, ACTION_ID, FACTORS, HELDOUT, MODES, SPLITS, MAX_STEPS,
                      Env, Planner, action_text, factor_scores, features, prompt,
                      rollout, summarize, tune_planner)
from r5_data import encode_example, sample_indices, rate_plan
from r5_eval import scenarios, successful_records
from r5_protocol import ActionGrammar, prompt_ids, action_ids, unpack_example, verify_episode
from r5_validation import validation_batches, validation_summary


def test_no_mlx_imported():
    # Also works when the old full test suite has imported MLX during collection.
    subprocess.run([sys.executable, "-c", "import sys, r5_world, r5_data, r5_eval; "
                    "assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)"], check=True)


@pytest.mark.parametrize("split", ("train",) + SPLITS)
def test_planner_solves_fresh_scenarios_and_recovers(split):
    for seed in range(100, 132):
        for mode in MODES:
            row = rollout(Planner(), Env(seed, split, mode), trace=True)
            assert row["success"], (seed, split, mode, row)
            assert row["steps"] < MAX_STEPS
            assert bool(row["disruption"]) == (mode != "none")
            if row["disruption"]:
                assert row["disruption"]["recovered_step"] > row["disruption"]["step"]
            for r in row["trace"]:
                assert r["action"] not in prompt(r["observation"])


def test_splits_and_redirects_do_not_leak():
    for seed in range(100):
        for split in ("train", "seen", "order", "pairs", "long3", "long4"):
            e = Env(seed, split, "redirect")
            if split == "pairs":
                assert any(tuple(g) in HELDOUT for g in e.orders)
            if split == "train":
                assert [i for i, _ in e.orders] == sorted(i for i, _ in e.orders)
            if split == "order":
                assert e.orders[0][0] > e.orders[1][0]
            if split.startswith("long"):
                assert len(e.orders) == int(split[-1])
                assert [i for i, _ in e.orders] != sorted(i for i, _ in e.orders)
            row = rollout(Planner(), e, trace=True)
            if split in ("train", "seen", "order"):
                for r in row["trace"]:
                    assert not any(tuple(g) in HELDOUT for g in r["observation"]["orders"])


def test_observations_do_not_leak_or_alias_hidden_world():
    a, b = Env(1), Env(1)
    b.item_at = [(i + 1) % 3 for i in b.item_at]
    b.bin_at.reverse()
    b.locked = [True] * 3
    assert a.observe() == b.observe()
    assert np.array_equal(features(a.observe()), features(b.observe()))
    o = a.observe()
    o["orders"][0][1] = 99
    o["items"][0] = 99
    assert a.observe() != o
    assert a.observe()["items"] == [-1] * 4


def test_inspection_only_reveals_local_contents_and_not_locks():
    e = Env(10, mode="locked")
    e.step("INSPECT")
    o = e.observe()
    for i, loc in enumerate(e.item_at):
        assert o["items"][i] == (loc if loc == e.room else -1)
    for c, loc in enumerate(e.bin_at):
        assert o["bins"][c] == ([loc, e.bin_label[c], -1] if loc == e.room else [-1] * 3)


def test_failures_cost_steps_and_never_count_as_success():
    e = Env(9)
    for _ in range(MAX_STEPS):
        assert not e.step("not a command")
    assert e.over() and not e.success() and e.steps == MAX_STEPS
    with pytest.raises(RuntimeError):
        e.step("INSPECT")


def test_factored_head_can_emit_every_action_including_every_pair():
    assert len(ACTIONS) == len(set(ACTIONS))
    for target, (v, i, a) in enumerate(FACTORS):
        heads = [np.full(6, -50.), np.full(4, -50.), np.full(3, -50.)]
        for head, k in zip(heads, (v, i, a)):
            head[max(k, 0)] = 0
        assert int(np.argmax(factor_scores(heads))) == target
        assert action_text(FACTORS[ACTION_ID[ACTIONS[target]]]) == ACTIONS[target]
    for split in SPLITS:
        assert features(Env(11, split).observe()).shape == features(Env(11).observe()).shape


def test_observable_changes_are_encoded():
    o = Env(11, "long4").observe()
    for key, value in (("cursor", 1), ("steps", 1), ("feedback", "illegal"), ("holding", 0)):
        p = copy.deepcopy(o)
        p[key] = value
        assert not np.array_equal(features(o), features(p))
        assert prompt(o) != prompt(p)


def test_conditional_metrics_expose_avoidance_and_censoring():
    good = rollout(Planner(), Env(14, mode="locked"))
    bad = copy.deepcopy(good)
    bad.update(success=False, steps=MAX_STEPS, invalid=MAX_STEPS)
    bad["disruption"]["recovered_step"] = None
    avoids = copy.deepcopy(bad)
    avoids["disruption"] = None
    m = summarize([good, bad, avoids])
    assert m["success_rate"] == pytest.approx(1 / 3)
    assert m["exposure_coverage"] == pytest.approx(2 / 3)
    assert m["success_given_disruption"] == .5
    assert m["recovery_rate"] == .5
    assert m["recovery_unresolved"] == 1
    assert m["recovery_steps_capped_mean"] > m["recovery_steps_mean"]
    assert summarize([avoids])["success_given_disruption"] is None


def test_tuning_uses_validation_only_and_is_deterministic():
    p, m = tune_planner(range(4_000_000, 4_000_002))
    assert len(m["trials"]) == 12
    assert m["selected"]["success_rate"] == 1
    assert p.room_order == m["selected"]["room_order"]


def test_incomplete_rollout_is_not_a_measured_failure():
    assert rollout(Planner(), Env(1), deadline=time.monotonic() - 1) is None


class FakeTokenizer:
    def render_for_completion(self, messages):
        return [11, 12, 13]
    def encode(self, text):
        return [21, 22]
    def encode_special(self, text):
        return 23


def test_sft_shift_supervises_action_and_end_only():
    ids, mask = encode_example(FakeTokenizer(), Env(1).observe(), "INSPECT", 8)
    assert ids.tolist() == [11, 12, 13, 21, 22, 23, 0, 0, 0]
    assert ids[1:][mask.astype(bool)].tolist() == [21, 22, 23]
    with pytest.raises(ValueError, match="never truncate"):
        encode_example(FakeTokenizer(), Env(1).observe(), "INSPECT", 3)


def test_sampling_resumes_and_rate_projection_scales():
    full = [sample_indices(12, s, 1000, 16) for s in range(6)]
    resumed = [sample_indices(12, s, 1000, 16) for s in range(3, 6)]
    np.testing.assert_array_equal(full[3:], resumed)
    assert not np.array_equal(full[0], full[1])
    assert rate_plan(200, 10, 420, 6144)["projected_remaining_steps"] == 8400


def test_reward_filter_never_admits_failed_episodes():
    assert successful_records([dict(success=False, trace=["bad"]),
                               dict(success=True, trace=["good"])]) == ["good"]
    assert successful_records([dict(success=False, trace=["bad"])]) == []


def test_eval_cells_interleaved_and_fresh_seeds_disjoint():
    cells = list(scenarios(2))
    assert len(cells) == 40 and len({s["seed"] for s in cells}) == 40
    assert len({(s["split"], s["mode"]) for s in cells[:20]}) == 20
    assert min(s["seed"] for s in cells) > 80_000_000
    assert not {s["seed"] for s in scenarios(2, True)} & {s["seed"] for s in cells}


class ByteTokenizer:
    """Byte-preserving stand-in: real BPE is also checked against retained NPZs by r5_audit."""
    def encode(self, text):
        return [100 + b for b in text.encode()]
    def encode_special(self, text):
        return 3
    def render_for_completion(self, messages):
        return [1, 2] + self.encode(messages[0]["content"]) + [4, 5]


def test_all_commands_roundtrip_shared_sft_and_decode_protocol():
    tok, o = ByteTokenizer(), Env(11).observe()
    grammar = ActionGrammar(tok)
    for action in ACTIONS:
        ids, mask = encode_example(tok, o, action, 2048)
        p, target = unpack_example(ids, mask)
        assert p == prompt_ids(tok, o)  # Includes every prompt byte and both separators.
        assert target == action_ids(tok, action)
        calls = []
        def teacher_logits(prefix, allowed):
            assert list(prefix) == target[:len(prefix)]
            calls.append(prefix)
            scores = np.full(len(allowed), -10.)
            scores[allowed.index(target[len(prefix)])] = 10.
            return scores
        assert grammar.decode(teacher_logits) == action
        assert len(calls) == len(target)  # Includes EOS; no silent generation truncation.


def test_protocol_rejects_prompt_loss_and_illegal_supervised_commands():
    ids, mask = encode_example(ByteTokenizer(), Env(2).observe(), "GO 1", 2048)
    wrong = mask.copy()
    wrong[0] = 1
    with pytest.raises(ValueError, match="contiguous"):
        unpack_example(ids, wrong)
    with pytest.raises(ValueError, match="grammar"):
        action_ids(ByteTokenizer(), "GO 99")


def test_validation_covers_tail_and_weights_tokens_not_batches():
    indices = np.concatenate(list(validation_batches(597, 16)))
    np.testing.assert_array_equal(indices, np.arange(597))
    # Regression: the old first-32 monitor missed this entire second-delivery group.
    result = validation_summary([.5] * 32 + [50.], [5] * 32 + [10],
                                [True] * 32 + [False], [1] * 32 + [2])
    assert result["examples"] == 33
    assert result["nll"] == pytest.approx(66 / 170)
    assert result["teacher_forced_action_exact"] == pytest.approx(32 / 33)
    assert result["by_deliveries"]["2"]["nll"] == 5
    with pytest.raises(ValueError):
        list(validation_batches(0, 16))


def test_reward_success_is_verified_by_replay_not_a_flag():
    row = rollout(Planner(), Env(47, mode="missing"), trace=True)
    assert verify_episode(row, require_success=True)
    tampered = copy.deepcopy(row)
    tampered["trace"][-1]["action"] = "INSPECT"
    with pytest.raises(ValueError):
        verify_episode(tampered, require_success=True)
    forged = rollout(lambda o: "INSPECT", Env(47), trace=True)
    forged["success"] = True
    with pytest.raises(ValueError):
        verify_episode(forged, require_success=True)


def test_turn2_headline_and_secondary_seed_ranges_are_disjoint():
    fresh = list(scenarios(4, seed_start=91_024_000))
    old = list(scenarios(4, seed_start=81_241_000))
    assert not {r["seed"] for r in fresh} & {r["seed"] for r in old}
    assert old == list(scenarios(4))
    assert len({(r["split"], r["mode"]) for r in fresh}) == 20
