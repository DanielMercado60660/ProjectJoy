"""Round 7 gate readers (results/r7/protocol.json), used by the queue scripts and the write-up.

    ../.venv/bin/python r7_gate.py 1b results/r7/select_<run>.json [...]    # exit 0 iff a seed's pick reaches >= 6/18
    ../.venv/bin/python r7_gate.py stage results/r7/select_<run>.json [...] # print the closed-loop stage gate per pick
"""
import json
import sys
from pathlib import Path

GATE = {"reached_plate": 6, "within_0_5m": 9, "no_press": 15, "done_without_press_max": 12}


def stage_gate(select_file):
    d = json.loads(Path(select_file).read_text())
    t = d["table"][d["chosen"]]
    r = t["r7"]
    comp = {"reached_ge_6": r["reached_plate"] >= GATE["reached_plate"],
            "within_0_5m_ge_9": r["within_0_5m"] >= GATE["within_0_5m"],
            "done_after_press_lamp_right_ge_half_pressed": r["pressed"] > 0 and r["done_after_press_lamp_right"] * 2 >= r["pressed"],
            "done_without_press_le_12": r["said_done_without_press"] <= GATE["done_without_press_max"],
            "no_press_ge_15": t["no_press"][0] >= GATE["no_press"]}
    return {"run": d["run"], "pick": d["chosen"], "press_needed": t["press_needed"], "no_press": t["no_press"],
            "reached": r["reached_plate"], "within_0_5m": r["within_0_5m"], "pressed": r["pressed"],
            "done_after_press_lamp_right": r["done_after_press_lamp_right"], "done_without_press": r["said_done_without_press"],
            "intentional_reach": r["intentional_reach"], "verified_done": r["verified_done"], "components": comp,
            "pass": all(comp.values())}


def stage1c_decision(onset_bh, bearing_bh, onset_ow, b1_passed):
    """The pre-registered 1C decision rule -> the extra p2_train flags for Stage 2A (section 4, 1C)."""
    ob = json.loads(Path(onset_bh).read_text())
    bb = json.loads(Path(bearing_bh).read_text())
    ow = json.loads(Path(onset_ow).read_text())
    onset_pass_bh, bearing_pass, onset_pass_ow = bool(ob.get("pass")), bool(bb.get("pass")), bool(ow.get("pass"))
    if bearing_pass and onset_pass_bh:
        flags, why = "--belief-heads", "bearing passes and onset passes: keep the heads"
    elif bearing_pass and not onset_pass_bh:
        flags = "--belief-heads" if b1_passed else ""
        why = "bearing passes, onset fails (belief present but unused): heads kept only because 1B passed" if b1_passed else \
            "bearing passes, onset fails and 1B failed: heads dropped"
    elif onset_pass_ow and not bearing_pass:
        flags, why = "--onset-weight 20", "onset passes for the weighting run alone: the loss weighting, not the architecture"
    else:
        flags, why = "", "bearing fails: the heads cannot learn the bearing from these windows; dropped"
    if not bearing_pass and onset_pass_ow and flags == "":
        flags, why = "--onset-weight 20", "bearing fails but onset weighting passes: 2A uses the weighting"
    return {"flags": flags, "why": why, "onset_bh": {k: ob.get(k) for k in ("share_p_bot_gt_p_opp", "agree", "none", "pass")},
            "bearing_bh": {k: bb.get(k) for k in ("check", "out_of_view", "pass")},
            "onset_ow": {k: ow.get(k) for k in ("share_p_bot_gt_p_opp", "agree", "none", "pass")}, "stage1b_passed": b1_passed}


def stage3a_gate(d=Path("results/r7")):
    """3A (plan section 4): learned selector >= 50% of the bot's press-needed successes and >= 5 above both controls
    (per seed, both seeds); kill if within 2 episodes of the belief-conditioned (freq) control on both seeds.
    All rows from the same r7_eval run on the 40 selection rooms (the bot row is the ceiling)."""
    def pn(name):
        s = json.loads((d / f"eval_select_{name}.json").read_text())["summary"]
        return s["press_needed"][0], s["no_press"][0]
    bot = pn("bot")
    freq = pn("sel_freq")
    out = {"bot": bot, "freq": freq, "program": pn("sel_program"), "seeds": {}}
    passes, kills = [], []
    for s in (0, 1):
        L, Lnp = pn(f"sel_learned_s{s}")
        S, _ = pn(f"sel_shuffled_s{s}")
        ok = L * 2 >= bot[0] and L >= S + 5 and L >= freq[0] + 5
        kill = L <= freq[0] + 2
        out["seeds"][s] = {"learned": [L, Lnp], "shuffled": S, "pass": ok, "within_2_of_freq": kill}
        passes.append(ok)
        kills.append(kill)
    out["pass"] = all(passes)
    out["kill_scaffolding"] = all(kills)
    out["verdict"] = "pass: 3B funded (two seeds)" if out["pass"] else ("kill: within 2 of the belief-conditioned control on both seeds: scaffolding; 3B = one-seed documentation run" if out["kill_scaffolding"] else "neither pass nor kill: 3B = one-seed documentation run")
    (d / "gate_3a.json").write_text(json.dumps(out, indent=1))
    return out


def main():
    cmd, files = sys.argv[1], sys.argv[2:]
    if cmd == "3a":
        g = stage3a_gate()
        print(json.dumps(g))
        sys.exit(0 if g["pass"] else 1)
    if cmd == "1c":
        b1 = files[3] == "1" if len(files) > 3 else True
        d = stage1c_decision(files[0], files[1], files[2], b1)
        Path("results/r7/stage2_config.json").write_text(json.dumps(d, indent=1))
        print(d["flags"])
        return
    gates = [stage_gate(f) for f in files]
    for g in gates:
        print(json.dumps(g))
    if cmd == "1b":
        sys.exit(0 if any(g["components"]["reached_ge_6"] for g in gates) else 1)


if __name__ == "__main__":
    main()
