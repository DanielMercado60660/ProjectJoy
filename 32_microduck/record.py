"""M4: record reference-bot episodes in the Phase 1 episode format.

One compressed .npz per episode under episodes/<task>/ (gitignored) plus episodes/index.jsonl.
Per 10 Hz tick t (the observation the controller had when it chose command t):
  frame        (96, 96, 3) uint8   head camera, area-downsampled from the 128x128 render
  body         (34,) float32       gyro 3, gravity 3, joint pos - default 14, joint vel 14
  imu_sub      (5, 6) float32      the tick's 50 Hz gyro and gravity samples
  cmd_exec     (8,) float32        executed command [vx, vy, wz, neck_p, head_p, head_y, head_r, done]
  cmd_bot      (8,) float32        the bot's own command (= cmd_exec unless a reflex/mixing changed it)
  skill        int16 index         innermost skill running (vocabulary in the header)
  activity     int16 index         the bot's chain of activities, e.g. "set_lamp>find>scan_here"
  expected / observed              the skill's expected outcome and its report when it finished
                                   (a hindsight segment label: every tick the skill commanded carries
                                   the report it gave at its end tick; "-" if it never finished)
  memory       JSON per tick       confirmed memory slots (class, position, sightings, last seen)
  sim_time, wall_time, obs_age     obs_age = sim time between the frame and the command (0 here)
  truth        (7,) float32        REPLAY ONLY, never for learning: duck x, y, yaw, ball x, y, lamp, presses
Header (JSON): config id, split, task, instruction, controller, params, perturbation, judge result,
bot events, skill log.

Perturbations (labelled on 12.5% of the M4 episodes, ~8% actually perturbed; chosen deterministically per config): "missed_press" (the first
press doesn't register; set_lamp episodes that need a press), "shove" (1 N sideways for 1 s at a random
tick), "ball_moved" (find_ball: once the ball has been out of view 1.5 s, it is moved to another hidden
spot that some viewpoint with a clear line of sight can see).
Known limits (REVIEW_M4.md, kept so the stored episodes stay reproducible): the shove tick is drawn from
[60, budget/2) and often falls after the episode ends (then perturbation.tick is null and the episode is
identical to its clean twin); the missed press swallows the first plate depression of any kind and never
fires if the duck never reaches the plate; ball_moved does not check that the viewpoint is reachable
(it was, post hoc, in all 28 M4 moves), and when no spot is found (to = null) the ball is put back with
zero velocity, so it stops a few cm short of its clean rest point.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from interface import Interface  # noqa: E402
from refbot import controller  # noqa: E402
from room import World  # noqa: E402
from tasks import BUDGET_TICKS, Judge, _hiders, _place_ball, ball_viewpoints, ball_visibility, make_episode, start_episode  # noqa: E402

EP_DIR = HERE / "episodes"
FRAME = 96
EXPECTED = {"go_to": "reach the target", "turn_to": "face the bearing", "look": "head at the pose",
            "stop": "standing still", "press_button": "lamp toggles", "_single": "-",
            "scan_here": "look around", "watch_ball": "keep the ball in view", "-": "-"}


def down(frame):
    return np.asarray(Image.fromarray(frame).resize((FRAME, FRAME), Image.Resampling.BOX))


def cmd_vec(c):
    return np.array([c.vx, c.vy, c.wz, *c.head, float(c.done)], np.float32)


def mem_snapshot(bot):
    tr = [[t["cls"], round(t["xy"][0], 3), round(t["xy"][1], 3), t["n"], t["last"]]
          for t in bot.mem.tracks if t["n"] >= bot.P["confirm"]]
    st = bot.mem.lamp_state()
    return {"tracks": tr, "lamp": None if st is None else [st[0], st[1]],
            "ball_last": bot.mem.ball_trail[-1] if bot.mem.ball_trail else None}


def _alt_hiding_spot(w, rng):
    """A floor point hidden from the duck's camera, behind some hider, with a reachable view."""
    lay = w.cfg["layout"]
    me = w.duck.pos()[:2]
    cur = w.ball_pos()[:2]
    cands = []
    for h in _hiders(lay):
        for e in h["ends"]:
            for hs in (1.0, -1.0):
                for back in (0.1, 0.25):
                    p = h["C"] + e * h["L"] * h["a"] - e * back * h["a"] + hs * 0.3 * h["n"]
                    if np.linalg.norm(p - cur) > 0.4:
                        cands.append(p)
    rng.shuffle(cands)
    for p in cands:
        _place_ball(w, p)
        if not ball_visibility(w)[1] and ball_viewpoints(w, p, me):
            return p
    _place_ball(w, cur)
    return None


def record_episode(cfg, split, task, k, perturb=None, params=None, out_dir=EP_DIR, seed=0,
                   policy_weights=None, beta=1.0, tag=None, cold_renderer=False, after_bot_done="stop",
                   policy_kind="m5", takeover_tick=None, takeover_fresh=False, budget_ticks=None):
    """policy_weights + beta < 1: DAgger recording. The bot runs in shadow mode on the same observation
    stream and its command is always recorded as the label (cmd_bot); each tick the executed command is
    the bot's with probability beta, else the policy's (cmd_exec).

    cold_renderer: the first frame a new MuJoCo renderer draws differs from later ones by 1-2 grey
    levels in some pixels (M6). The recorder renders its own tick-0 frame before the bot's first look
    (the bot first senses at tick 1), so by default the bot never sees that first frame, while under
    run_episode (M3/M6 evaluation) it does, and episodes can drift apart. cold_renderer=True draws the
    recorder's tick-0 frame with a separate renderer, so a bot episode follows run_episode exactly.
    Off by default: the M4 episodes were recorded without it.

    after_bot_done: when the shadow bot finishes while the policy is still driving, "stop" (default)
    ends the episode there, so its "done" label is recorded once; "continue" lets the policy run on
    to the budget as in evaluation (replays only: the label after the bot finished is NaN).

    policy_kind: "m5" (learner.PolicyController, default) or "p2" (p2_learner.P2Controller, which reads
    the config beside the weights). Explicit, not detected: M5 run directories also hold a meta.json.

    takeover_tick (amendment 5): the policy drives until this tick, then the bot's commands are executed.
    takeover_fresh: at that tick a new bot instance starts from the current state instead of the shadow
    bot (the shadow bot's log is kept in header['dagger']['pre_takeover_events']).
    budget_ticks: overrides the task budget (the bot is given the same budget)."""
    ep = make_episode(cfg, task, k)
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    judge = Judge(w, ep)
    budget = BUDGET_TICKS[task] if budget_ticks is None else int(budget_ticks)
    fac, bot = controller(task, budget, params)
    gen = fac(iface, ep["instruction"])
    rng = np.random.default_rng([int(cfg["hash"], 16) % 2 ** 32, k, 99])
    mix_rng = np.random.default_rng([int(cfg["hash"], 16) % 2 ** 32, k, 7])
    pgen, pcmd, bot_done, bot_done_tick = None, None, False, None
    if policy_weights is not None:
        if policy_kind == "m5":
            from learner import PolicyController
            pgen = PolicyController(policy_weights).factory(iface, ep["instruction"])
        elif policy_kind == "p2":
            from p2_learner import P2Controller
            pgen = P2Controller(policy_weights).factory(iface, ep["instruction"])
        else:
            raise ValueError("policy_kind must be m5 or p2")
        pcmd = next(pgen)
    n_policy = 0
    pre_takeover_events = None
    pert = {"type": perturb, "tick": None}
    if perturb == "missed_press":
        w.suppress_presses = 1
        pert["tick"] = 0
    shove_at = int(rng.integers(60, max(61, budget // 2))) if perturb == "shove" else None
    shove_dir = float(rng.uniform(0, 2 * np.pi))
    hidden_run, moved = 0, False
    F, B, I, CE, CB, SK, AC, MEM, ST, WT, TR = ([] for _ in range(11))
    vocab_sk, vocab_ac = {}, {}
    obs = iface.observe()
    t0 = time.time()
    cmd = next(gen)
    while True:
        if takeover_fresh and takeover_tick is not None and iface.tick_count == takeover_tick:
            pre_takeover_events = list(bot.log)
            fac, bot = controller(task, budget - takeover_tick, params)
            gen = fac(iface, ep["instruction"])
            cmd = next(gen)
            bot_done = False
        taken = takeover_tick is not None and iface.tick_count >= takeover_tick
        exec_cmd = cmd
        if pgen is not None and not taken and (bot_done or mix_rng.random() >= beta):
            exec_cmd = pcmd
            n_policy += 1
        chain = bot.activity()
        skill = bot.skill_stack[-1] if bot.skill_stack else ("watch_ball" if chain and chain[-1] == "find_ball" else "-")
        act = ">".join(c for c in chain if c not in ("drive", "_single")) or "-"
        if cold_renderer and not F:
            import mujoco
            from interface import FRAME_SIZE
            r0 = mujoco.Renderer(w.model, FRAME_SIZE, FRAME_SIZE)
            F.append(down(obs._frame if obs._frame is not None else w.duck.head_image(r0)))
            r0.close()
        else:
            F.append(down(obs.frame()))
        b = obs.body
        B.append(np.concatenate([b["gyro"], b["gravity"], b["joint_pos"], b["joint_vel"]]).astype(np.float32))
        I.append(np.concatenate([b["gyro_substeps"], b["gravity_substeps"]], 1).astype(np.float32))
        CE.append(cmd_vec(exec_cmd))
        CB.append(np.full(8, np.nan, np.float32) if bot_done else cmd_vec(getattr(bot, "proposed", cmd)))
        SK.append(vocab_sk.setdefault(skill, len(vocab_sk)))
        AC.append(vocab_ac.setdefault(act, len(vocab_ac)))
        MEM.append(mem_snapshot(bot))
        ST.append(obs.sim_time)
        WT.append(time.time() - t0)
        p = w.duck.pos()
        bp = w.ball_pos()
        TR.append([p[0], p[1], w.duck.yaw(), bp[0], bp[1], float(w.lamp_on), float(w.press_count)])
        # perturbations act on the world between ticks
        if shove_at is not None and iface.tick_count == shove_at:
            pert["tick"] = shove_at
            w.duck.push_force = np.array([np.cos(shove_dir), np.sin(shove_dir), 0.0])
        if shove_at is not None and iface.tick_count == shove_at + 10:
            w.duck.push_force = np.zeros(3)
        if perturb == "ball_moved" and not moved and iface.tick_count > 20:
            hidden_run = hidden_run + 1 if not ball_visibility(w)[0] else 0
            if hidden_run >= 15:
                moved = True
                spot = _alt_hiding_spot(w, rng)
                pert.update(tick=iface.tick_count, to=None if spot is None else spot.round(3).tolist())
        obs = iface.tick(exec_cmd)
        judge.update(iface)
        if iface.done or iface.tick_count >= budget:
            break
        if not bot_done:
            try:
                cmd = gen.send(obs)
            except StopIteration:
                # The bot finished (its last label, "done", is already recorded). In a DAgger episode
                # the policy may still be driving; by default the episode ends here too, so the label
                # isn't repeated. (Before this fix the last command kept being recorded as the label on
                # every remaining tick: DAgger round 2 has 19,084 such ticks, REVIEW_M5.md.)
                bot_done, bot_done_tick = True, iface.tick_count
                if pgen is None or after_bot_done == "stop":
                    break
        if pgen is not None and not (takeover_tick is not None and iface.tick_count >= takeover_tick):
            pcmd = pgen.send(obs)
    res = judge.result(iface)
    res = {kk: (v.item() if hasattr(v, "item") else v) for kk, v in res.items()}
    T = len(F)
    expected = [EXPECTED.get(s, "-") for s in (sorted(vocab_sk, key=vocab_sk.get))]
    observed = ["-"] * T
    # A skill commands ticks start..end-1: at tick `end` the next skill already issues the command.
    # (Before REVIEW_M4 this range included `end`, so each skill's first tick carried the previous
    # skill's report; the 1,029 M4 episodes were recorded that way. Rebuild from header skill_log.)
    for e in bot.skill_log:
        for t in range(max(0, e["start"]), min(T, e["end"])):
            if observed[t] == "-":
                observed[t] = f"{e['skill']}: {'ok' if e['ok'] else 'not ok'} ({e['reason']})"
    header = {"format": "microduck-episode/1", "config_id": cfg["id"], "split": split, "task": task, "k": k,
              "instruction": ep["instruction"], "controller": "refbot" + ("+gaze" if (params or {}).get("gaze") else ""),
              "params": params or {}, "perturbation": pert, "result": res, "events": bot.log,
              "skill_log": bot.skill_log, "skill_vocab": sorted(vocab_sk, key=vocab_sk.get),
              "activity_vocab": sorted(vocab_ac, key=vocab_ac.get), "expected_by_skill": expected,
              "ticks": T, "frame_size": FRAME,
              "dagger": None if pgen is None else {"policy": str(policy_weights), "beta": beta,
                                                   "policy_ticks": n_policy, "bot_finished_tick": bot_done_tick,
                                                   "after_bot_done": after_bot_done,
                                                   "policy_kind": policy_kind, "takeover_tick": takeover_tick,
                                                   "takeover_fresh": takeover_fresh,
                                                   "pre_takeover_events": pre_takeover_events}}
    sub = Path(out_dir) / task
    sub.mkdir(parents=True, exist_ok=True)
    name = f"{split}_{cfg['id']}_k{k}" + (f"_{perturb}" if perturb else "") + (f"_{tag}" if tag else "")
    path = sub / f"{name}.npz"
    np.savez_compressed(path, frames=np.stack(F), body=np.stack(B), imu_sub=np.stack(I),
                        cmd_exec=np.stack(CE), cmd_bot=np.stack(CB), skill=np.array(SK, np.int16),
                        activity=np.array(AC, np.int16), sim_time=np.array(ST, np.float32),
                        wall_time=np.array(WT, np.float64), obs_age=np.zeros(T, np.float32),
                        truth=np.array(TR, np.float32), observed=np.array(observed),
                        memory=np.array(json.dumps(MEM)), header=np.array(json.dumps(header)))
    return {"path": str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path), "config": cfg["id"], "split": split, "task": task, "k": k,
            "instruction": ep["instruction"], "controller": header["controller"], "perturbation": pert,
            "success": res["success"], "ticks": T, "mb": round(path.stat().st_size / 1e6, 2),
            "wall_s": round(time.time() - t0, 1)}
