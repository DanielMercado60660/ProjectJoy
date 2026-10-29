"""M6: re-record a few test-set failures (and one floor success) and build media/m6_failures.html.

    python m6_clips.py record     # ~7 episodes -> episodes/m6_clips/ (gitignored), results/m6/clips.json
    python m6_clips.py page       # -> media/m6_failures.html, with notes from results/m6/clip_notes.json

Episodes are picked by fixed rules from the M6 rows (first match in config order), so nothing is chosen
by eye. They are replays for explanation only: no training, tuning or selection uses them. A policy
episode is recorded with record.py at beta 0 (the policy drives every tick, the bot runs in shadow and
its proposals show as "bot asked", blank once the bot has finished; the policy runs on to the budget, as in
M6); a bot episode is recorded plainly. All use cold_renderer, so the
bot sees the same first frame as under m6_eval (see record.record_episode). clips.json says whether
each re-recorded outcome matches the M6 row.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from m6_eval import SETS, load_set  # noqa: E402
from m6_report import load_rows  # noqa: E402

POLICY = HERE / "runs" / "m5_dagger2" / "ckpt_006000" / "weights.safetensors"
OUT = HERE / "episodes" / "m6_clips"
RULES = [
    ("policy_set_lamp_said_done", "dagger", "set_lamp", "instances-test",
     lambda r: r["press_needed"] and r["declared_done"] and not r["success"]),
    ("policy_set_lamp_floor", "dagger", "set_lamp", "instances-test",
     lambda r: (not r["press_needed"]) and r["success"]),
    ("policy_go_and_return_pad", "dagger", "go_and_return", None,
     lambda r: r["set"] != "train" and r["reached_pad_tick"] is not None),
    ("policy_find_ball", "dagger", "find_ball", "instances-test", lambda r: not r["success"]),
    ("bot_set_lamp_appearances", "bot", "set_lamp", "appearances-test",
     lambda r: r["press_needed"] and not r["success"]),
    ("bot_go_and_return_appearances", "bot", "go_and_return", "appearances-test",
     lambda r: (not r["success"]) and r["reached_pad_tick"] is not None),
    ("bot_find_ball_partition", "bot", "find_ball", "arrangements-test",
     lambda r: r["hider"] == "partition" and not r["success"]),
]


def picks():
    rows = load_rows()
    out = []
    for name, who, task, s, ok in RULES:
        rs = [r for r in rows if r["who"] == who and r["task"] == task and (s is None or r["set"] == s)]
        rs.sort(key=lambda r: (list(SETS).index(r["set"]) if r["set"] in SETS else 9, r["config_id"]))
        r = next(r for r in rs if ok(r))
        out.append({"clip": name, "who": who, "task": task, "set": r["set"], "config_id": r["config_id"],
                    "m6_success": r["success"], "m6_sim_time": r["sim_time"]})
    return out


def record():
    from record import record_episode
    res = []
    for p in picks():
        cfg = next(c for c in load_set(p["set"], 10 ** 6) if c["id"] == p["config_id"])
        kw = dict(policy_weights=POLICY, beta=0.0, tag="policy", after_bot_done="continue") if p["who"] == "dagger" else {}
        kw["cold_renderer"] = True     # the bot sees the renderer's first frame, as in m6_eval
        r = record_episode(cfg, p["set"], p["task"], 0, out_dir=OUT, **kw)
        p.update(path=r["path"], recorded_success=r["success"], recorded_ticks=r["ticks"],
                 matches_m6=bool(r["success"]) == bool(p["m6_success"]))
        print(json.dumps(p), flush=True)
        res.append(p)
    (HERE / "results" / "m6" / "clips.json").write_text(json.dumps(res, indent=1))


def page():
    from m4_replay import main
    clips = json.loads((HERE / "results" / "m6" / "clips.json").read_text())
    nf = HERE / "results" / "m6" / "clip_notes.json"
    notes = json.loads(nf.read_text()) if nf.exists() else {}
    main([HERE / c["path"] for c in clips], out=HERE / "media" / "m6_failures.html",
         title="Microduck M6 — failures on the reserved test rooms",
         sub=("Test-set episodes picked by fixed rules (m6_clips.py), re-recorded for replay. In policy episodes "
              "the learned policy drives every tick and the reference bot runs in shadow: 'command' is what the "
              "duck did, 'bot asked' and the activity are what the bot would have done. Truth is for replay only."),
         notes=[notes.get(c["clip"]) for c in clips], every=4)


if __name__ == "__main__":
    {"record": record, "page": page}[sys.argv[1]]()
