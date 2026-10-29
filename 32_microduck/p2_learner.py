"""Phase 2: M5's recurrent policy with configurable pixels and bot-belief auxiliary heads.

No simulator state is an input. The optional done connection reads the *predicted* match
probabilities, including during training; it never gets the auxiliary target labels.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from interface import Command
from learner import Policy, body_vec, cmd_targets, decode as decode_regression, instr_vec


TURN_IN_PLACE = 0.55       # the bot's turn-in-place rate; wz below ~0.5 does not start a turn on this body
TURN_WZ = (0.0, TURN_IN_PLACE, -TURN_IN_PLACE)   # turn-head classes: none / left (wz > 0) / right


# T3: discrete walking avoids averaging stop/walk commands into the body's gait dead zone.
# Amendment 6: a discrete turn head does the same for turning in place (none / left / right -> 0 / +-0.55).
def decode(reg, done_logit, vx_head="regress", walk_logit=None, turn_logits=None):
    cmd = decode_regression(reg, done_logit)
    if vx_head == "binary":
        if walk_logit is None:
            raise ValueError("binary vx decoding requires a walk logit")
        # sigmoid(logit) > 0.5 iff logit > 0; equality means stop.
        cmd = Command(vx=0.3 if walk_logit > 0.0 else 0.0, vy=cmd.vy, wz=cmd.wz, head=cmd.head, done=cmd.done)
    elif vx_head != "regress":
        raise ValueError("vx_head must be regress or binary")
    if turn_logits is not None and cmd.vx < 0.15:          # standing: turn in place at the bot's rate, or not
        cmd = Command(vx=cmd.vx, vy=cmd.vy, wz=TURN_WZ[int(np.argmax(turn_logits))], head=cmd.head, done=cmd.done)
    return cmd


def prep_frame(frame, img=64):
    """Live: 128 -> record.down's 96 -> img. Recorded frames start at the same 96."""
    if img not in (64, 96):
        raise ValueError("img must be 64 or 96")
    if frame.shape[0] == 128:
        from record import down
        frame = down(frame)
    if frame.shape[:2] != (img, img):
        frame = np.asarray(Image.fromarray(frame).resize((img, img), Image.Resampling.BOX))
    return frame


def policy_config(weights=None, cfg=None):
    """Read an explicit dict, a weights-specific .json, p2_config.json, or checkpoint meta."""
    if cfg is None:
        p = Path(weights)
        for candidate in (p.with_suffix(".json"), p.parent / "p2_config.json", p.parent / "meta.json"):
            if candidate.exists():
                cfg = json.loads(candidate.read_text())
                break
        else:
            raise FileNotFoundError(f"No Phase 2 config beside {p}; pass cfg explicitly")
    cfg = cfg.get("cfg", cfg)
    out = {"img": int(cfg.get("img", 64)),
           "aux": bool(cfg.get("aux", cfg.get("aux_weight", 1.0) > 0)),
           "done_from_aux": bool(cfg.get("done_from_aux", False)),
           "hidden": int(cfg.get("hidden", 256)),
           "act": str(cfg.get("act", "relu")),
           "vx_head": str(cfg.get("vx_head", "regress")),  # T3: old configs retain regression
           "turn_head": str(cfg.get("turn_head", "regress"))}  # amendment 6
    if out["vx_head"] not in ("regress", "binary"):
        raise ValueError("vx_head must be regress or binary")
    if out["turn_head"] not in ("regress", "discrete"):
        raise ValueError("turn_head must be regress or discrete")
    if out["act"] not in ("relu", "leaky"):
        raise ValueError("act must be relu or leaky")
    if out["img"] not in (64, 96):
        raise ValueError("img must be 64 or 96")
    if out["done_from_aux"] and not out["aux"]:
        raise ValueError("done_from_aux requires auxiliary heads")
    return out


class Policy2(Policy):
    """Same encoder/GRU as Policy: NHWC frames and batch-major (B, T, F) GRU input.

    The inherited stride-2 convolutions give 64 -> 32 -> 16 -> 8 -> 4 or
    96 -> 48 -> 24 -> 12 -> 6. Aux outputs are logits, not probabilities.
    T3: binary mode returns (B, T) walk logits in aux['walk'], even with aux=False;
    the return tuple and six regression slots remain unchanged.
    """
    def __init__(self, img=64, aux=True, done_from_aux=False, hidden=256, act="relu", vx_head="regress",
                 turn_head="regress"):
        if img not in (64, 96):
            raise ValueError("img must be 64 or 96")
        if done_from_aux and not aux:
            raise ValueError("done_from_aux requires auxiliary heads")
        if vx_head not in ("regress", "binary"):
            raise ValueError("vx_head must be regress or binary")
        super().__init__(hidden=hidden)
        self.image_size = img
        self.act = act        # "leaky": leaky ReLU (slope 0.05) in the image encoder, so conv units can't die
        # (Phase 2, P2-A: with plain ReLU the last conv layer went silent for every frame within ~250 steps on
        # some seeds, in both trainers; then the policy is blind. M5's runs happened to survive.)
        self.has_aux = aux
        self.done_from_aux = done_from_aux
        if img == 96:
            self.img = nn.Linear(6 * 6 * 64, 192)
        if aux:
            self.lamp = nn.Linear(128, 3)
            self.match = nn.Linear(128, 3)
        if done_from_aux:
            self.done = nn.Linear(128 + 3, 1)
        # T3: retain the six regression slots and legacy parameter names; slot 0 is unused in binary mode.
        self.vx_head = vx_head
        if vx_head == "binary":
            self.walk = nn.Linear(128, 1)
        if turn_head not in ("regress", "discrete"):
            raise ValueError("turn_head must be regress or discrete")
        self.turn_head = turn_head            # amendment 6: (B, T, 3) none/left/right logits in aux['turn']
        if turn_head == "discrete":
            self.turn = nn.Linear(128, 3)

    def encode(self, frames, body, instr):
        if self.act == "relu":
            return super().encode(frames, body, instr)
        f = lambda v: nn.leaky_relu(v, 0.05)  # noqa: E731
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        for c in (self.c1, self.c2, self.c3, self.c4):
            x = f(c(x))
        x = f(self.img(x.reshape(B * T, -1))).reshape(B, T, -1)
        return mx.concatenate([x, nn.relu(self.bod(body)), nn.relu(self.ins(instr))], -1)

    def encoder_health(self, frames):
        """Fraction of last-conv channels that are positive for at least one frame."""
        x = frames
        for c in (self.c1, self.c2, self.c3, self.c4):
            x = nn.relu(c(x)) if self.act == "relu" else nn.leaky_relu(c(x), 0.05)
        c4 = (x.reshape(-1, x.shape[-1]).max(0) > 0).mean()
        return float(c4)

    def __call__(self, frames, body, instr, h0=None):
        z = self.encode(frames, body, instr)
        hs = self.gru(z, hidden=h0)                         # (B, T, hidden), as in M5
        y = nn.relu(self.h1(mx.concatenate([hs, nn.relu(self.ins(instr))], -1)))
        aux = {"lamp": self.lamp(y), "match": self.match(y)} if self.has_aux else {}
        if self.vx_head == "binary":             # T3: action logit, independent of auxiliary supervision
            aux["walk"] = self.walk(y)[..., 0]
        if self.turn_head == "discrete":
            aux["turn"] = self.turn(y)
        dy = mx.concatenate([y, mx.softmax(aux["match"], axis=-1)], -1) if self.done_from_aux else y
        return mx.tanh(self.reg(y)), self.done(dy)[..., 0], aux, hs[:, -1]


class P2Controller:
    """PolicyController's generator protocol, optional gaze reflex, and per-tick latency list."""
    def __init__(self, weights, cfg=None, gaze=False, device="cpu"):
        from bodymodel import BodyModel
        from gaze import Gaze
        from odometry import Odometry
        if device == "cpu":
            mx.set_default_device(mx.cpu)
        self.cfg = policy_config(weights, cfg)
        self.model = Policy2(**self.cfg)
        self.model.load_weights(str(weights))
        mx.eval(self.model.parameters())
        self.gaze_on = gaze
        self.Gaze, self.BodyModel, self.Odometry = Gaze, BodyModel, Odometry
        self.latency = []

    def factory(self, iface, instruction):
        import time
        from bodymodel import rot_z, tilt_rotation
        gz, bm, od = self.Gaze(), self.BodyModel(), self.Odometry()
        ins = mx.array(instr_vec(instruction)[None, None])
        h = None
        obs = iface.observe()
        while True:
            t0 = time.perf_counter()
            fr = mx.array(prep_frame(obs.frame(), self.cfg["img"])[None, None].astype(np.float32) / 255.0)
            bd = mx.array(body_vec(obs.body)[None, None])
            reg, dl, aux, h = self.model(fr, bd, ins, h)
            mx.eval(reg, dl, h)
            walk = float(aux["walk"][0, 0]) if self.cfg["vx_head"] == "binary" else None
            turn = np.array(aux["turn"][0, 0]) if self.cfg["turn_head"] == "discrete" else None
            cmd = decode(np.array(reg[0, 0]), float(dl[0, 0]), self.cfg["vx_head"], walk, turn)  # T3, amendment 6
            self.latency.append(time.perf_counter() - t0)
            if self.gaze_on:
                oy, op = gz.offsets(obs.sim_time)
                hd = list(cmd.head)
                cmd = Command(vx=cmd.vx, vy=0.0, wz=cmd.wz,
                              head=(hd[0], hd[1] + op, hd[2] + oy, hd[3]), done=cmd.done)
            obs = yield cmd
            if obs is None:
                obs = iface.observe()
            if self.gaze_on:
                od.update(obs)
                _, _, cm = bm.fk(obs.body["joint_pos"])
                gz.update(obs, rot_z(od.psi) @ tilt_rotation(obs.body["gravity"]) @ cm)
