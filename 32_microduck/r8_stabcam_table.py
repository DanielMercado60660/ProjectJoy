"""Markdown tables for results/r8/stabcam/REPORT.md, straight from the result files (no numbers typed by hand).

    ../.venv/bin/python r8_stabcam_table.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r8_stabcam as S  # noqa: E402

ORDER = ["head", "reflex", "trunk", "oracle", "eis", "eis_lag", "eis_lagc", "eis_reflex", "eis_reflex_lag",
         "eis_reflex_lagc", "pl_head", "pl_eis", "pl2_head", "pl2_eis"]
LABEL = {"head": "head (as now)", "reflex": "head + reflex (gaze.py)", "trunk": "trunk camera (sim-only hardware)",
         "oracle": "ORACLE (sim truth)", "eis": "eis (sensor-only warp)", "eis_lag": "eis, 20 ms late",
         "eis_lagc": "eis, 20 ms late + extrapolated", "eis_reflex": "eis + reflex", "eis_reflex_lag": "eis + reflex, 20 ms late",
         "eis_reflex_lagc": "eis + reflex, late + extrapolated", "pl_head": "phase-locked head (1/stride)",
         "pl_eis": "phase-locked eis (1/stride)", "pl2_head": "phase-locked head (2/stride)", "pl2_eis": "phase-locked eis (2/stride)"}


def f(x, n=2):
    return "-" if x is None else f"{x:.{n}f}"


def main():
    W = json.loads((S.OUT / "wobble.json").read_text())
    NH = json.loads((S.OUT / "wobble_nohl.json").read_text()) if (S.OUT / "wobble_nohl.json").exists() else None
    wa = W["scripted_walking_average"]
    g = W["gate_check"]
    bot_off, bot_on = W["bot"].get("off", {}), W["bot"].get("on", {})
    hfc = wa["frame_change"]["head"]["mean"]
    bfc = bot_off.get("frame_change", {}).get("head", {}).get("median_of_medians")
    nh = NH["scripted_walking_average"]["frame_change"] if NH else {}
    nhh = nh.get("head", {}).get("mean")
    print("| stream | scripted yaw / pitch / roll (deg) | gate | frame change (vs head) | no headlight (vs head) | bot yaw / pitch (deg) | bot gate | bot frame change, median (vs head) |")
    print("|---|---|---|---|---|---|---|---|")
    for n in ORDER:
        row = g["scripted"].get(n)
        if row is None:
            continue
        fc = wa["frame_change"].get(n, {}).get("mean")
        fcs = f"{fc:.2f} ({100 * (fc / hfc - 1):+.0f}%)" if fc is not None else "-"
        nf = nh.get(n, {}).get("mean")
        nfs = f"{nf:.2f} ({100 * (nf / nhh - 1):+.0f}%)" if nf is not None and nhh else "-"
        bkey = "bot_on" if n.startswith("eis_reflex") or n == "reflex" else "bot_off"
        brow = g.get(bkey, {}).get(n)
        bsrc = bot_on if bkey == "bot_on" else bot_off
        bf = bsrc.get("frame_change", {}).get(n, {}).get("median_of_medians")
        bfs = f"{bf:.2f} ({100 * (bf / bfc - 1):+.0f}%)" if bf is not None and bfc else "-"
        gate = ("pass" if row["pass"] else "fail") + ("" if row["sensor_only"] else " (not sensor-only)")
        bgate = "-" if brow is None else ("pass" if brow["pass"] else "fail")
        print(f"| {LABEL[n]} | {f(row['yaw'])} / {f(row['pitch'])} / {f(row['roll'])} | {gate} | {fcs} | {nfs} | "
              f"{'-' if brow is None else f(brow['yaw']) + ' / ' + f(brow['pitch'])} | {bgate} | {bfs} |")
    print()
    print("eis usability: scripted valid", wa["valid"].get("eis"), "clamped (walking avg)", round(wa["clamped_off"], 3),
          "| bot valid", bot_off.get("valid", {}).get("eis"), "clamped", bot_off.get("clamped"))
    print("bot episodes off:", [(e["config"], e["success"], e["ticks"]) for e in bot_off.get("episodes", [])],
          "successes", bot_off.get("successes"), "falls", bot_off.get("falls"))
    print("bot episodes on:", [(e["config"], e["success"], e["ticks"]) for e in bot_on.get("episodes", [])],
          "successes", bot_on.get("successes"), "falls", bot_on.get("falls"))
    print("bot walk fraction off:", [round(x, 2) for x in bot_off.get("walk_frac", [])])
    for kind in ("pl", "pl2"):
        for src, d in (("scripted", wa.get(kind)), ("bot", bot_off.get(kind))):
            if not d:
                continue
            print(f"{kind} {src}: head wobble locked {d['head_wobble_locked']} vs same-rate clock {d['head_wobble_clock_same_rate']}"
                  f" vs 2 Hz {d['head_wobble_clock_2hz']}; frame change locked {f(d['head_change_locked'])} vs clock "
                  f"{f(d['head_change_clock_same_rate'])} / 2 Hz {f(d['head_change_clock_2hz'])} / same-gap {f(d['head_change_samegap_randphase'])};"
                  f" eis locked {d['eis_wobble_locked']} change {f(d['eis_change_locked'])} vs clock {f(d['eis_change_clock_same_rate'])}")
    turn = W["scripted"].get("vx0.0_wz0.55", {}).get("wobble", {})
    print("turn in place:", {k: v[:2] for k, v in turn.items()})
    for cond, c in W["scripted"].items():
        print(cond, {k: v[:2] for k, v in c["wobble"].items() if k in ("head", "reflex", "trunk", "oracle", "eis", "eis_lagc")})
    pr = S.OUT / "scripted_per_room.json"
    if pr.exists():
        P = json.loads(pr.read_text())["summary"]
        print("post hoc per-room:", {k: {kk: vv for kk, vv in v.items() if kk in ("runs", "head", "trunk", "oracle", "eis", "eis_lagc", "pl_head")} for k, v in P.items()})


if __name__ == "__main__":
    main()
