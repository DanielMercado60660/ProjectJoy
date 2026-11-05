"""Round 7 Stage 5 table: every arm's single look at the fresh reserved slice (results/r7/eval_fresh_<arm>.json), in
Markdown, with Wilson 95% intervals on the press-needed and no-press rates. Also the 80-room / 40-room tables for
the arms that have them.

    ../.venv/bin/python r7_report.py [fresh|confirm|select]      # prints Markdown; writes results/r7/table_<rooms>.md
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "r7"
LABEL = {"p2_hr_s0": "step-5 Hr s0 (pixels)", "p2_hr_s1": "step-5 Hr s1 (pixels)", "p2_r3_s1": "step-5 R3 s1 (pixels)",
         "r7_1bv1_s0": "1B-v1 s0: belief vector, no pixels (learned decisions + supplied perception)",
         "r7_1bv1_s1": "1B-v1 s1: belief vector, no pixels (learned decisions + supplied perception)",
         "r7_1bv0_s0_superseded": "1B-v0 s0, superseded recipe (flickering lamp belief), extra row",
         "bot": "frozen reference bot (ceiling)", "servo_lamp": "1D map-free pixel servo, sensors-only lamp latch (hand-written baseline)",
         "servo_oracle": "1D servo, ORACLE latch", "sel_learned_s0": "3A learned selector s0 (learned decisions + supplied perception/skills)",
         "sel_learned_s1": "3A learned selector s1 (learned decisions + supplied perception/skills)",
         "sel_shuffled_s0": "3A shuffled-label control s0", "sel_shuffled_s1": "3A shuffled-label control s1",
         "sel_freq": "3A belief-conditioned most-frequent control", "sel_program": "3A program rule (hand-written)",
         "sel_learned_s0_dagger": "3B learned selector s0 + decision-level DAgger (documentation run)",
         "hr_s1_gaze": "4A Hr s1 with the reflex on", "r3_s1_gaze": "4A R3 s1 with the reflex on"}


def wilson(k, n, z=1.96):
    if n == 0:
        return "—"
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return f"{100 * max(0.0, c - h):.0f}–{100 * min(1.0, c + h):.0f}%"


def table(rooms):
    files = sorted(OUT.glob(f"eval_{rooms}_*.json"))
    fams = {"fresh": ("instances-r7", "arrangements-r7", "appearances-r7", "distractor-r7"),
            "confirm": ("instances-dev", "arrangements-dev", "appearances-dev", "distractor-dev"),
            "select": ("instances-dev", "arrangements-dev", "appearances-dev", "distractor-dev")}[rooms]
    lines = ["| arm | press-needed (95% CI) | per family (inst / arr / app / dist) | no-press (95% CI) | reached | within 0.5 m | pressed | done after press (lamp right) | done w/o press | intentional / reaches | verified / dones after press | falls | median done tick (pn / np) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for f in files:
        d = json.loads(f.read_text())["summary"]
        name = d["policy"]
        if rooms == "select" and "ckpt" in name:
            continue
        r = d["r7"]["press_needed"]
        fam = d["r7"]["by_family"]
        famtxt = " / ".join(f"{fam[k]['press_needed'][0]}/{fam[k]['press_needed'][1]}" if k in fam else "—" for k in fams)
        pn, npn = d["press_needed"], d["no_press"]
        rows.append((name, f"| {LABEL.get(name, name)} | {pn[0]}/{pn[1]} ({wilson(*pn)}) | {famtxt} | {npn[0]}/{npn[1]} ({wilson(*npn)}) | {r['reached_plate']} | {r['within_0_5m']} | "
                     f"{r['pressed']} | {r['done_after_press']} ({r['done_after_press_lamp_right']}) | {r['said_done_without_press']} | {r['intentional_reach']} / {r['reached_plate']} | "
                     f"{r['verified_done']} / {r['done_after_press']} | {d['falls']} | {r['median_done_tick']} / {d['r7']['no_press']['median_done_tick']} |"))
    order = list(LABEL)
    rows.sort(key=lambda t: order.index(t[0]) if t[0] in order else 99)
    lines += [t[1] for t in rows]
    md = "\n".join(lines)
    (OUT / f"table_{rooms}.md").write_text(md + "\n")
    return md


if __name__ == "__main__":
    print(table(sys.argv[1] if len(sys.argv) > 1 else "fresh"))
