"""M5: the first learned mind, a small recurrent policy in MLX (CNN encoder + GRU).

Inputs per 10 Hz tick (the same Observation every controller gets, plus the instruction):
  frame       head camera, downsampled to IMG x IMG, uint8 -> [0, 1]
  body        gyro 3, gravity 3, joint pos - default 14, joint vel 14 (34, normalised)
  instruction one-hot: task (3) + target (pad colour 4, lamp on/off 2) = 9
No previous command is fed in (the brief warns that history inputs invite copying it); the GRU state
carries what the policy needs to remember.
Outputs: vx, wz (tanh-scaled to the interface bounds), 4 head-pose deltas (tanh x 0.8 caps), and a
`done` logit. vy is the interface's fixed-zero slot. Labels are the bot's own commands (cmd_bot,
before the chicken-head reflex); at run time the same reflex (gaze.py) is applied to the policy's
head commands, so both minds get the same body.
"""
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from PIL import Image

from interface import BOUNDS, Command

HERE = Path(__file__).resolve().parent
IMG = 64
TASKS = ("go_and_return", "set_lamp", "find_ball")
TARGETS = ("red", "green", "blue", "yellow", "on", "off")
HEAD_B = np.array(BOUNDS["head"], np.float32)
VX_LO, VX_HI = BOUNDS["vx"]
WZ = BOUNDS["wz"][1]
BODY_SCALE = np.concatenate([np.full(3, 2.0), np.full(3, 1.0), np.full(14, 0.5), np.full(14, 4.0)]).astype(np.float32)


def instr_vec(instruction):
    v = np.zeros(len(TASKS) + len(TARGETS), np.float32)
    v[TASKS.index(instruction["task"])] = 1
    if instruction.get("target") is not None:
        v[len(TASKS) + TARGETS.index(instruction["target"])] = 1
    return v


def prep_frame(frame):
    """Any square uint8 RGB frame -> IMG x IMG uint8. A live 128 x 128 render first goes to the
    recorder's 96 x 96 (record.down), so live and training frames take the same two box steps."""
    if frame.shape[0] == 128:
        from record import down
        frame = down(frame)
    if frame.shape[0] != IMG:
        frame = np.asarray(Image.fromarray(frame).resize((IMG, IMG), Image.Resampling.BOX))
    return frame


def body_vec(body):
    if isinstance(body, dict):
        body = np.concatenate([body["gyro"], body["gravity"], body["joint_pos"], body["joint_vel"]])
    return (np.asarray(body, np.float32) / BODY_SCALE).astype(np.float32)


def cmd_targets(cmd8):
    """[vx, vy, wz, h0..h3, done] -> normalised regression targets (6) and done (1)."""
    c = np.asarray(cmd8, np.float32)
    vx = (c[..., 0] - (VX_HI + VX_LO) / 2) / ((VX_HI - VX_LO) / 2)
    wz = c[..., 2] / WZ
    head = c[..., 3:7] / HEAD_B
    return np.concatenate([vx[..., None], wz[..., None], head], -1).clip(-1, 1), c[..., 7]


def decode(reg, done_logit):
    reg = np.asarray(reg, np.float32)
    vx = reg[0] * (VX_HI - VX_LO) / 2 + (VX_HI + VX_LO) / 2
    return Command(vx=float(vx), vy=0.0, wz=float(reg[1] * WZ), head=tuple(float(h) for h in reg[2:6] * HEAD_B),
                   done=bool(done_logit > 0.0))


class Policy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)     # 64 -> 32
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)    # 32 -> 16
        self.c3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)    # 16 -> 8
        self.c4 = nn.Conv2d(64, 64, 3, stride=2, padding=1)    # 8 -> 4
        self.img = nn.Linear(4 * 4 * 64, 192)
        self.bod = nn.Linear(34, 64)
        self.ins = nn.Linear(len(TASKS) + len(TARGETS), 32)
        self.gru = nn.GRU(192 + 64 + 32, hidden)
        self.h1 = nn.Linear(hidden + 32, 128)
        self.reg = nn.Linear(128, 6)
        self.done = nn.Linear(128, 1)
        self.hidden = hidden

    def encode(self, frames, body, instr):
        """frames (B, T, H, W, 3) float in [0,1]; body (B, T, 34); instr (B, T, 9) -> (B, T, F)."""
        B, T = frames.shape[:2]
        x = frames.reshape(B * T, *frames.shape[2:])
        x = nn.relu(self.c1(x))
        x = nn.relu(self.c2(x))
        x = nn.relu(self.c3(x))
        x = nn.relu(self.c4(x))
        x = nn.relu(self.img(x.reshape(B * T, -1))).reshape(B, T, -1)
        return mx.concatenate([x, nn.relu(self.bod(body)), nn.relu(self.ins(instr))], -1)

    def __call__(self, frames, body, instr, h0=None):
        z = self.encode(frames, body, instr)
        hs = self.gru(z, hidden=h0)                       # (B, T, hidden)
        y = nn.relu(self.h1(mx.concatenate([hs, nn.relu(self.ins(instr))], -1)))
        return mx.tanh(self.reg(y)), self.done(y)[..., 0], hs[:, -1]


class PolicyController:
    """Runs a trained Policy through the frozen interface (one GRU step per tick), with the same
    chicken-head reflex as the bot. Usable as a tasks.run_episode controller factory."""

    def __init__(self, weights, gaze=False, device="cpu"):
        from bodymodel import BodyModel
        from gaze import Gaze
        from odometry import Odometry
        if device == "cpu":          # batch-1 inference of this small net is faster on the CPU (0.3 vs 0.7 ms)
            mx.set_default_device(mx.cpu)
        self.model = Policy()
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
            fr = mx.array(prep_frame(obs.frame())[None, None].astype(np.float32) / 255.0)
            bd = mx.array(body_vec(obs.body)[None, None])
            reg, dl, h = self.model(fr, bd, ins, h)
            mx.eval(reg, dl, h)
            cmd = decode(np.array(reg[0, 0]), float(dl[0, 0]))
            self.latency.append(time.perf_counter() - t0)
            if self.gaze_on:
                oy, op = gz.offsets(obs.sim_time)
                hd = list(cmd.head)
                cmd = Command(vx=cmd.vx, vy=0.0, wz=cmd.wz, head=(hd[0], hd[1] + op, hd[2] + oy, hd[3]), done=cmd.done)
            obs = yield cmd
            if obs is None:
                obs = iface.observe()
            if self.gaze_on:
                od.update(obs)
                _, _, cm = bm.fk(obs.body["joint_pos"])
                gz.update(obs, rot_z(od.psi) @ tilt_rotation(obs.body["gravity"]) @ cm)
