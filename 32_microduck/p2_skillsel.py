"""Round 7 Stage 3: skill selection on the shared sensors-only map (design/R7_CAMPAIGN.md; Astra's option 2).

    ../.venv/bin/python p2_skillsel.py extract                       # decision set from the bot's set-lamp demos
    ../.venv/bin/python p2_skillsel.py train learned --seed 0         # MLX selector  -> results/r7/skillsel/learned_s0.*
    ../.venv/bin/python p2_skillsel.py train shuffled --seed 0        # shuffled-label control
    ../.venv/bin/python p2_skillsel.py freq                           # (prev skill, lamp belief) most-frequent control
    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python r7_eval.py select \
        sel_learned_s0=selector:results/r7/skillsel/learned_s0.npz sel_program=selector:program \
        sel_freq=selector:freq:results/r7/skillsel/freq.json
    OMP_NUM_THREADS=1 P2_PROCS=4 ../bin/gpu_run ../.venv/bin/python p2_skillsel.py dagger learned_s0 --n 300   # 3B round

The learner decides, at every macro-skill boundary, which of six macro-skills runs next: scan (the bot's
look-around: learn_floor the first time), explore (walk to the nearest frontier / ring point, then scan),
approach (walk to the look point 0.6 m before the remembered plate and face it), press (the bot's
press_button + 12 settle ticks), look_lamp (turn to the remembered lamp and read it; from where it was first
seen if hidden), done. The map, skills, perception and budgets are the bot's (supplied, disclosed); the learner
never receives the bot's chosen goal or program stage. Its input at a decision is the 58-float belief vector
(belief.py, from the bot's memory and pose), the target, the previous macro-skill and whether it reported ok,
the budget fraction used, and the lamp-vs-target bits (unknown / match / mismatch): 72 floats.
Decision labels come from the bot's own set-lamp demos (train split), segmented by the recorded activity and
skill columns plus the bot's events; the activity strings are used only to find the segment boundaries, never
as an input. Controls: the same MLP on shuffled labels; the most-frequent next skill given (previous skill,
lamp belief); "program": the bot's set_lamp rule written over the same features (the adapter's ceiling and the
3B DAgger labeller). Evaluation: r7_eval.py with weights 'selector:<spec>'.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from belief import DIM, OFF_LAMP, BeliefState, belief_vector, replay_poses, snapshot_of, vectors_from_recording  # noqa: E402

OUT = HERE / "results" / "r7" / "skillsel"
SKILLS = ("scan", "explore", "approach", "press", "look_lamp", "done")
SCAN, EXPLORE, APPROACH, PRESS, LOOK, DONE = range(6)
FEAT = DIM + 2 + 7 + 1 + 1 + 3          # 72
BUDGET = 900


# ------------------------------------------------------------------------------------------- features
def lamp_bits(vec, target):
    """unknown / match / mismatch of the belief's lamp one-hot against the target."""
    if vec[OFF_LAMP] > 0.5:
        return np.array([1.0, 0.0, 0.0], np.float32)
    on = vec[OFF_LAMP + 1] > 0.5
    return np.array([0.0, 1.0, 0.0], np.float32) if on == (target == "on") else np.array([0.0, 0.0, 1.0], np.float32)


def features(vec, target, prev, prev_ok, tick, budget=BUDGET):
    f = np.zeros(FEAT, np.float32)
    f[:DIM] = vec
    f[DIM + (0 if target == "on" else 1)] = 1.0
    f[DIM + 2 + (6 if prev is None else prev)] = 1.0
    f[DIM + 9] = float(bool(prev_ok))
    f[DIM + 10] = min(tick / budget, 1.0)
    f[DIM + 11:DIM + 14] = lamp_bits(vec, target)
    return f


# ------------------------------------------------------------------------------------------- extraction
def macro_segments(H, z):
    """[(start_tick, end_tick_exclusive, skill_index, ok)] from a bot-driven set_lamp recording."""
    av, sv = H["activity_vocab"], H["skill_vocab"]
    act = [av[i] for i in z["activity"]]
    sk = [sv[i] for i in z["skill"]]
    T = len(act)
    checks = [e["tick"] for e in H["events"] if e["event"] == "lamp_check" and e.get("state") in ("on", "off")]
    first_check = checks[0] if checks else 10 ** 9
    labels, bounds = [], []
    prev, find_go_seen, prev_act = None, False, None
    for t in range(T):
        a, s = act[t], sk[t]
        new_seg = False
        if a.startswith("set_lamp>learn_floor"):
            L = SCAN
        elif a.startswith("set_lamp>find"):
            if a.endswith(">go"):
                L = EXPLORE
                if not prev_act or not prev_act.endswith(">go"):
                    new_seg = True
                find_go_seen = True
            else:
                L = EXPLORE if find_go_seen else SCAN
                if L == SCAN and prev_act is not None and not prev_act.startswith("set_lamp>find"):
                    new_seg = True
        elif a == "set_lamp>scan_here":
            L = EXPLORE
        elif a == "set_lamp>go":
            L = EXPLORE if t < first_check else APPROACH
            if L == EXPLORE and prev_act != a:
                new_seg = True
        elif a.startswith("set_lamp>look_at_lamp"):
            L = LOOK
        elif s == "press_button":
            L = PRESS
        elif t == T - 1 and float(z["cmd_bot"][t, 7]) > 0.5:
            L = DONE
        elif prev in (APPROACH, PRESS) and s in ("stop", "turn_to"):
            L = prev
        elif prev is None:
            L = SCAN
        else:
            L = prev
        if not a.startswith("set_lamp>find"):
            find_go_seen = False                         # a later find() starts afresh with its own scan
        if L != prev or new_seg:
            bounds.append(t)
        labels.append(L)
        prev, prev_act = L, a
    bounds.append(T)
    segs = []
    observed = [str(o) for o in z["observed"]]
    for a, b in zip(bounds, bounds[1:]):
        L = labels[a]
        ok = True
        if L in (APPROACH, PRESS):
            ok = any(o.startswith(("go_to: ok", "press_button: ok")) for o in observed[a:b]) if L == APPROACH else \
                any(o.startswith("press_button: ok") for o in observed[a:b])
        elif L == LOOK:
            ok = any(a <= e["tick"] < b + 1 and e["event"] == "lamp_check" and e.get("state") in ("on", "off") for e in H["events"])
        segs.append((a, b, L, ok))
    return segs


def decisions_from_recording(path):
    with np.load(path, allow_pickle=False) as z:
        H = json.loads(str(z["header"]))
        if H["task"] != "set_lamp" or H.get("dagger"):
            return []
        memory = json.loads(str(z["memory"]))
        poses = replay_poses(z["body"], z["imu_sub"])
        V, _ = vectors_from_recording(memory, poses)
        segs = macro_segments(H, z)
    target = H["instruction"]["target"]
    out, prev, prev_ok = [], None, False
    for a, b, L, ok in segs:
        out.append({"x": features(V[a], target, prev, prev_ok, a, H.get("budget_ticks", BUDGET)).tolist(), "y": L,
                    "tick": a, "config": H["config_id"], "prev": prev})
        prev, prev_ok = L, ok
    return out


def extract(out=OUT / "decisions.json"):
    rows = [json.loads(l) for l in open(HERE / "episodes" / "index.jsonl")]
    paths = [HERE / r["path"] for r in rows if r["task"] == "set_lamp" and "error" not in r]
    dec = []
    for p in paths:
        dec += decisions_from_recording(p)
    counts = {SKILLS[k]: int(sum(d["y"] == k for d in dec)) for k in range(6)}
    trans = {}
    for d in dec:
        key = f"{'start' if d['prev'] is None else SKILLS[d['prev']]}->{SKILLS[d['y']]}"
        trans[key] = trans.get(key, 0) + 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"episodes": len(paths), "decisions": len(dec), "counts": counts, "transitions": trans,
                               "feat": FEAT, "skills": SKILLS, "rows": dec}))
    print(json.dumps({"episodes": len(paths), "decisions": len(dec), "counts": counts, "transitions": trans}), flush=True)
    return dec


# ------------------------------------------------------------------------------------------- selectors
def program_choice(f):
    """The bot's set_lamp rule over the same features (hand-written; the adapter's ceiling and the DAgger labeller)."""
    prev = int(np.argmax(f[DIM + 2:DIM + 9]))
    prev = None if prev == 6 else prev
    unknown, match, mismatch = (f[DIM + 11] > 0.5, f[DIM + 12] > 0.5, f[DIM + 13] > 0.5)
    button = f[0 * 6 + 5] > 0.5
    crossed, changed = f[DIM - 2] > 0.5, f[DIM - 1] > 0.5
    frac = f[DIM + 10]
    if prev is None:
        return SCAN
    if prev == PRESS:
        return LOOK
    if prev == LOOK:
        if crossed:                   # the lamp was not re-seen after the press: assume it worked, as the bot does
            return DONE
        return DONE if match else (APPROACH if button else EXPLORE)
    if unknown:
        return DONE if frac > 0.67 else EXPLORE
    if match:
        return DONE
    if prev == APPROACH:
        return PRESS
    return APPROACH if button else EXPLORE


class MLP:
    def __init__(self, seed=0, hidden=64):
        import mlx.core as mx
        import mlx.nn as nn
        mx.random.seed(seed)
        self.net = nn.Sequential(nn.Linear(FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def fit(self, X, Y, steps=3000, batch=64, lr=1e-3, seed=0):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        rng = np.random.default_rng(seed)
        opt = optim.Adam(learning_rate=lr)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        self.mu, self.sd = mu, sd
        Xn = ((X - mu) / sd).astype(np.float32)

        def loss_fn(net, x, y):
            return nn.losses.cross_entropy(net(x), y, reduction="mean")
        vg = nn.value_and_grad(self.net, loss_fn)
        for step in range(steps):
            idx = rng.integers(len(Xn), size=batch)
            loss, g = vg(self.net, mx.array(Xn[idx]), mx.array(Y[idx]))
            opt.update(self.net, g)
            mx.eval(self.net.parameters(), opt.state)
        return float(loss)

    def predict(self, X):
        import mlx.core as mx
        Xn = ((np.asarray(X, np.float32) - self.mu) / self.sd).astype(np.float32)
        return np.array(mx.argmax(self.net(mx.array(Xn)), axis=-1))

    def save(self, path):
        import mlx.core as mx
        from mlx.utils import tree_flatten
        mx.save_safetensors(str(path), dict(tree_flatten(self.net.parameters())))
        np.savez(str(path).replace(".safetensors", ".npz"), mu=self.mu, sd=self.sd)

    @classmethod
    def load(cls, path):
        import mlx.core as mx
        m = cls()
        m.net.load_weights(str(path).replace(".npz", ".safetensors"))
        mx.eval(m.net.parameters())
        z = np.load(str(path).replace(".safetensors", ".npz"))
        m.mu, m.sd = z["mu"], z["sd"]
        return m


def train(kind, seed=0, decisions=OUT / "decisions.json", extra=None, steps=3000):
    d = json.loads(Path(decisions).read_text())
    rows = d["rows"] + (json.loads(Path(extra).read_text())["rows"] if extra else [])
    X = np.array([r["x"] for r in rows], np.float32)
    Y = np.array([r["y"] for r in rows], np.int32)
    if kind == "shuffled":
        Y = np.random.default_rng(1000 + seed).permutation(Y)
    m = MLP(seed)
    loss = m.fit(X, Y, steps=steps, seed=seed)
    acc = float((m.predict(X) == Y).mean())
    name = f"{kind}_s{seed}" + ("_dagger" if extra else "")
    OUT.mkdir(parents=True, exist_ok=True)
    m.save(OUT / f"{name}.safetensors")
    rep = {"kind": kind, "seed": seed, "decisions": len(rows), "steps": steps, "final_loss": loss, "train_accuracy": acc,
           "program_agreement": float(np.mean([program_choice(x) == y for x, y in zip(X, Y)])) if kind == "learned" else None,
           "extra": extra}
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep), flush=True)
    return rep


def freq_table(decisions=OUT / "decisions.json"):
    """Most-frequent next skill given (previous skill, lamp belief vs target): the belief-conditioned control."""
    d = json.loads(Path(decisions).read_text())
    counts = {}
    for r in d["rows"]:
        x = np.array(r["x"], np.float32)
        key = f"{'start' if r['prev'] is None else r['prev']}|{int(np.argmax(x[DIM + 11:DIM + 14]))}"
        counts.setdefault(key, [0] * 6)[r["y"]] += 1
    table = {k: int(np.argmax(v)) for k, v in counts.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "freq.json").write_text(json.dumps({"table": table, "counts": counts}, indent=1))
    print(json.dumps({k: SKILLS[v] for k, v in table.items()}), flush=True)
    return table


def load_chooser(spec):
    """'program' | 'freq:<json>' | '<path to learned/shuffled .npz>' -> f(features) -> skill index."""
    if spec == "program":
        return program_choice
    if spec.startswith("freq:"):
        table = json.loads(Path(spec[5:]).read_text())["table"]

        def choose(f):
            prev = int(np.argmax(f[DIM + 2:DIM + 9]))
            key = f"{'start' if prev == 6 else prev}|{int(np.argmax(f[DIM + 11:DIM + 14]))}"
            return int(table.get(key, DONE))
        return choose
    m = MLP.load(spec)
    return lambda f: int(m.predict(f[None])[0])


# --------------------------------------------------------------------------------------------- adapter
class SelectorBot:
    """The reference bot's body of skills, map and memory with the learner choosing the next macro-skill."""

    def __init__(self, chooser, params=None, labeller=None):
        from refbot import RefBot
        self.bot = RefBot(params)
        self.chooser, self.labeller = chooser, labeller
        self.state = BeliefState()
        self.vec = None
        self.decisions = []
        self.seen_from = None
        self.floor_fitted = False

    def _sense_hook(self):
        b = self.bot
        snap = snapshot_of(b.mem, b.P["confirm"])
        self.vec, _ = belief_vector(snap, b.pose(), b.tick, self.state)
        if self.seen_from is None and b.mem.lamp_state() is not None:
            self.seen_from = list(b.pose()[:2])

    def run(self, iface, instruction, budget):
        import numpy as np  # noqa: F811
        from skills import look, press_button, stop, turn_to
        b = self.bot
        ctx = b.ctx(iface)
        target = instruction["target"]
        prev, prev_ok = None, False
        orig_sense = b.sense

        def sense(obs):
            orig_sense(obs)
            self._sense_hook()
        b.sense = sense
        self._sense_hook()
        idle = 0          # consecutive decisions that consumed no tick (look_lamp with no remembered lamp): a chooser
        while b.tick < budget - 2:   # that repeats such a choice would loop forever, so the third in a row costs a tick
            f = features(self.vec, target, prev, prev_ok, b.tick, budget)
            k = int(self.chooser(f))
            rec = {"tick": b.tick, "skill": k, "x": f.tolist()}
            if self.labeller is not None:
                rec["label"] = int(self.labeller(f))
            self.decisions.append(rec)
            if k == DONE:
                yield ctx.cmd(done=True)
                return
            t_before = b.tick
            ok = True
            if k == SCAN:
                if not self.floor_fitted:
                    yield from b.learn_floor(ctx)
                    self.floor_fitted = b.floor.centres is not None
                else:
                    yield from b.scan_here(ctx, pitch=b.P["learn_pitch"])
            elif k == EXPLORE:                     # the bot's find() / lamp-search step: nearest frontier, then scan
                pts = b.frontier_points() or b.explore_points()
                if pts:
                    x, y, _ = b.pose()
                    p = min(pts, key=lambda q: np.hypot(*(q - [x, y])))
                    lamp_known = b.mem.lamp_xy() is not None
                    pitch = b.P["scan_pitch"] if lamp_known else 0.0
                    want = (lambda: b.mem.best("button") is not None) if lamp_known else (lambda: b.mem.lamp_state() is not None)
                    yield from b.go(ctx, lambda p=p: p, stop_radius=0.15, max_ticks=250 if lamp_known else 250, pitch=pitch)
                    yield from b.scan_here(ctx, pitch=pitch, stop_when=want)
                else:
                    yield from b.drive(stop(ctx, max_ticks=3))
                    ok = False
            elif k == APPROACH:
                bt = b.mem.best("button")
                if bt is None:
                    yield from b.drive(stop(ctx, max_ticks=3))
                    ok = False
                else:
                    x, y, _ = b.pose()
                    bxy = np.array(bt["xy"])
                    u = (bxy - [x, y]) / max(np.linalg.norm(bxy - [x, y]), 1e-6)
                    look_pt = bxy - 0.6 * u
                    if b.is_free(look_pt):
                        d = yield from b.go(ctx, lambda: look_pt, stop_radius=0.12, max_ticks=250, pitch=0.45)
                        ok = d < 0.3
                    x, y, _ = b.pose()
                    b2 = np.array(b.mem.best("button")["xy"])
                    yield from b.drive(turn_to(ctx, np.arctan2(b2[1] - y, b2[0] - x), tol=0.2))
                    for _ in range(4):
                        yield from b.drive(stop(ctx, max_ticks=1))
            elif k == PRESS:
                bt = b.mem.best("button")
                if bt is None:
                    yield from b.drive(stop(ctx, max_ticks=3))
                    ok = False
                else:
                    ctx.head = (0.0, 0.45, 0.0, 0.0)
                    rep = yield from b.drive(press_button(ctx, bt["xy"]))
                    b.event("press", ok=rep["ok"], reason=rep["reason"])
                    ok = bool(rep["ok"])
                    for _ in range(12):
                        yield from b.drive(stop(ctx, max_ticks=1))
            elif k == LOOK:
                new = yield from b.look_at_lamp(ctx)
                if new is None and self.seen_from is not None and b.tick < budget - 120:
                    new = yield from b.look_at_lamp(ctx, frm=self.seen_from)
                b.event("lamp_check", state=new)
                ok = new is not None
            idle = idle + 1 if b.tick == t_before else 0
            if idle >= 3:                 # the guard (amendment 3): only a would-have-hung loop ever reaches it
                yield from b.drive(stop(ctx, max_ticks=1))
                idle = 0
            prev, prev_ok = k, ok
        yield ctx.cmd(done=True)


def selector_controller(spec, budget, params=None, labeller=None):
    """A run_episode-compatible controller factory; returns (factory, bot) like refbot.controller."""
    chooser = load_chooser(spec)
    sel = SelectorBot(chooser, params, labeller)
    bot = sel.bot

    def factory(iface, instruction):
        gen = sel.run(iface, instruction, budget)
        bot.task_gen = gen
        try:
            cmd = next(gen)
            while True:
                bot.proposed = cmd
                obs = yield bot.stabilise(cmd)
                cmd = gen.send(obs)
        except StopIteration:
            return
    bot.selector = sel
    return factory, bot


# ----------------------------------------------------------------------------------------------- DAgger
def _dagger_job(args):
    spec, cfg, idx = args
    from interface import Interface
    from room import World
    from tasks import Judge, make_episode, start_episode
    ep = make_episode(cfg, "set_lamp", 0)
    w = World(cfg)
    start_episode(w, ep)
    iface = Interface(w)
    judge = Judge(w, ep)
    fac, bot = selector_controller(spec, BUDGET, labeller=program_choice)
    gen = fac(iface, ep["instruction"])
    try:
        cmd = next(gen)
        while not iface.done and iface.tick_count < BUDGET:
            obs = iface.tick(cmd)
            judge.update(iface)
            if iface.done:
                break
            cmd = gen.send(obs)
    except StopIteration:
        pass
    finally:
        if iface._renderer is not None:
            iface._renderer.close()
    res = judge.result(iface)
    rows = [{"x": d["x"], "y": d["label"], "tick": d["tick"], "config": cfg["id"], "prev": None, "chosen": d["skill"]}
            for d in bot.selector.decisions]
    return {"config": cfg["id"], "success": bool(res["success"]), "press_needed": bool(res["needed_presses"]),
            "decisions": rows, "agree": int(sum(d["skill"] == d["label"] for d in bot.selector.decisions))}


def dagger(name, n=300, procs=4):
    """One decision-level DAgger round: the learned selector drives set_lamp on train configs, the program labels
    every decision it faced; the labelled decisions go to results/r7/skillsel/dagger_<name>.json."""
    import os
    from multiprocessing import Pool
    cfgs = json.loads((HERE / "splits" / "train.json").read_text())["configs"][:n]
    spec = str(OUT / f"{name}.npz")
    with Pool(int(os.environ.get("P2_PROCS", procs))) as pool:
        out = pool.map(_dagger_job, [(spec, c, i) for i, c in enumerate(cfgs)], chunksize=1)
    rows = [d for o in out for d in o["decisions"]]
    rep = {"selector": name, "episodes": len(out), "decisions": len(rows), "agree": int(sum(o["agree"] for o in out)),
           "successes": int(sum(o["success"] for o in out)),
           "press_needed_successes": [int(sum(o["success"] for o in out if o["press_needed"])), int(sum(o["press_needed"] for o in out))],
           "rows": rows}
    (OUT / f"dagger_{name}.json").write_text(json.dumps(rep))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}), flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("extract", "train", "freq", "dagger"))
    ap.add_argument("name", nargs="?", default="learned")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default=None, help="train: a dagger_*.json decision set to add")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--steps", type=int, default=3000)
    a = ap.parse_args()
    if a.cmd == "extract":
        extract()
    elif a.cmd == "train":
        train(a.name, a.seed, extra=a.extra, steps=a.steps)
    elif a.cmd == "freq":
        freq_table()
    else:
        dagger(a.name, a.n)


if __name__ == "__main__":
    main()
