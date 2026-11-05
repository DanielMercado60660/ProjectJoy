"""Round 7: a sensors-only *belief* of where the plate, lamp, mat, pads and ball are, as one fixed-layout vector
per 10 Hz tick (design/R7_CAMPAIGN.md, Stage 1A). One builder serves both uses:

  training labels   from a recorded episode: the bot's (or shadow bot's) confirmed memory tracks per tick
                    (record.py `memory`, in the bot's odometry frame) plus a leg-odometry pose replayed from the
                    stored IMU samples and joint encoders (odometry.py, anchors off);
  deployment        P2Controller runs BeliefTracker on its own observations: the same Odometry (anchors off),
                    CameraPose, perception.detect and refbot.Memory the bot uses, then the same vector builder.

No simulator truth enters anywhere here. Truth is used only by p2_belief_labels.py to *score* label error.

Layout (DIM = 58):
  per class in CLASSES (8 x 6): sin / cos of the relative bearing to the remembered track, log1p(distance),
      in-view bit (track seen this tick), log-age of the last sighting (capped at AGE_CAP, 1.0 when absent),
      present flag (a confirmed track exists);
  lamp belief (4): one-hot unknown / on / off of the latest confirmed lamp track (lamp_on / lamp_off with
      n >= confirm; sticky: the bot's own lamp_state() also demands a fresh run of >= 3 consecutive sightings
      and so flickers to "unknown" after an isolated re-sighting; amendment 2), and its log-age;
  own pose in the odometry frame (4): x / 2, y / 2, sin yaw, cos yaw;
  flags (2): crossed_button (the odometry position came within CROSS_M of the remembered plate since the last
      lamp sighting) and lamp_changed_since_crossing (the first sighting after a crossing differed from the
      belief held before it).
Tick convention (record.py / refbot.controller): the bot first senses at tick 1, so memory[0] is empty and the
odometry pose at tick 0 is the origin; the snapshot at tick t already contains tick t's detections.
"""
import numpy as np

CLASSES = ("button", "lamp", "mat", "pad_red", "pad_green", "pad_blue", "pad_yellow", "ball")
PER_CLASS = 6
AGE_CAP = 300
CROSS_M = 0.15
OFF_LAMP = len(CLASSES) * PER_CLASS          # 48
OFF_POSE = OFF_LAMP + 4                      # 52
OFF_FLAGS = OFF_POSE + 4                     # 56
DIM = OFF_FLAGS + 2                          # 58
VERSION = "r7_belief_v2"
_LOG_CAP = float(np.log1p(AGE_CAP))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def pick_tracks(snapshot):
    """One track per class from a memory snapshot ({"tracks": [[cls, x, y, n, last], ...], ...}): lamp_on /
    lamp_off merge into "lamp" (latest sighting wins; its class is the lamp belief); other classes take the
    most-sighted track, then the latest. Returns {class: (x, y, last_seen_tick, cls)}."""
    best = {}
    for row in snapshot.get("tracks", []):
        cls, x, y, n, last = row[0], float(row[1]), float(row[2]), int(row[3]), int(row[4])
        key = "lamp" if cls.startswith("lamp") else cls
        if key not in CLASSES:
            continue
        score = (last, n) if key == "lamp" else (n, last)
        cur = best.get(key)
        if cur is None or score > cur[0]:
            best[key] = (score, (x, y, last, cls))
    return {k: v[1] for k, v in best.items()}


class BeliefState:
    """The two running flags; restart it wherever the odometry frame restarts."""
    def __init__(self):
        self.crossed = 0
        self.lamp_changed = 0
        self.last_lamp_tick = -1
        self.last_lamp_state = None
        self.state_before_cross = None


def belief_vector(snapshot, pose, t, state):
    """Vector (DIM,) float32 and per-class last-seen ticks (8,) int64 (-1 when absent) at tick t."""
    x, y, psi = float(pose[0]), float(pose[1]), float(pose[2])
    v = np.zeros(DIM, np.float32)
    last_seen = np.full(len(CLASSES), -1, np.int64)
    tr = pick_tracks(snapshot)
    for k, cls in enumerate(CLASSES):
        o = PER_CLASS * k
        if cls in tr:
            tx, ty, last, _ = tr[cls]
            b = wrap(np.arctan2(ty - y, tx - x) - psi)
            v[o] = np.sin(b)
            v[o + 1] = np.cos(b)
            v[o + 2] = np.log1p(np.hypot(tx - x, ty - y))
            v[o + 3] = 1.0 if last == t else 0.0
            v[o + 4] = np.log1p(min(max(t - last, 0), AGE_CAP)) / _LOG_CAP
            v[o + 5] = 1.0
            last_seen[k] = last
        else:
            v[o + 4] = 1.0
    lamp = None if "lamp" not in tr else ("on" if tr["lamp"][3] == "lamp_on" else "off", tr["lamp"][2])
    if lamp is None:
        v[OFF_LAMP] = 1.0
    else:
        v[OFF_LAMP + (1 if lamp[0] == "on" else 2)] = 1.0
        v[OFF_LAMP + 3] = np.log1p(min(max(t - int(lamp[1]), 0), AGE_CAP)) / _LOG_CAP
        if int(lamp[1]) != state.last_lamp_tick:            # a sighting newer than the one last folded in
            if state.crossed:
                state.lamp_changed = int(state.state_before_cross is not None and lamp[0] != state.state_before_cross)
                state.crossed = 0
            state.last_lamp_tick, state.last_lamp_state = int(lamp[1]), lamp[0]
    if "button" in tr and not state.crossed and t > state.last_lamp_tick:   # a crossing is only AFTER the last sighting
        bx, by, _, _ = tr["button"]
        if np.hypot(bx - x, by - y) < CROSS_M:
            state.crossed = 1
            state.state_before_cross = state.last_lamp_state
            state.lamp_changed = 0
    v[OFF_POSE:OFF_POSE + 4] = (x / 2.0, y / 2.0, np.sin(psi), np.cos(psi))
    v[OFF_FLAGS] = float(state.crossed)
    v[OFF_FLAGS + 1] = float(state.lamp_changed)
    return v, last_seen


def snapshot_of(mem, confirm):
    """record.mem_snapshot's view of a refbot.Memory (confirmed tracks and the latest lamp sighting)."""
    tr = [[t["cls"], round(t["xy"][0], 3), round(t["xy"][1], 3), t["n"], t["last"]] for t in mem.tracks if t["n"] >= confirm]
    st = mem.lamp_state()
    return {"tracks": tr, "lamp": None if st is None else [st[0], st[1]]}


class BeliefTracker:
    """Deployment side: the bot's own sensing chain with anchors off and no map, run on the learner's observations.
    update(obs) after every interface tick (never at tick 0); vector() gives the current belief."""

    def __init__(self):
        from bodymodel import BodyModel
        from odometry import Odometry
        from refbot import PARAMS, Memory
        self.P = dict(PARAMS, anchor_gain=0.0)
        self.body = BodyModel()
        self.odo = Odometry(self.body)
        self.mem = Memory(self.P, grid=None)
        self.state = BeliefState()
        self.tick = 0
        self.dets = []
        self.poses = [(0.0, 0.0, 0.0)]       # odometry pose per tick (index = tick)

    def update(self, obs):
        from perception import CameraPose, detect
        self.tick = int(obs.tick)
        self.odo.update(obs)
        cam = CameraPose(self.body, obs, self.odo.pose())
        self.dets = detect(obs.frame(), cam)
        self.mem.update(self.dets, self.tick, cam.pos[:2])
        self.mem.lamp_update(self.dets, self.tick, cam.pos[:2])
        self.poses.append(tuple(float(v) for v in self.odo.pose()))
        return self.dets

    def snapshot(self):
        return snapshot_of(self.mem, self.P["confirm"])

    def vector(self):
        return belief_vector(self.snapshot(), self.odo.pose(), self.tick, self.state)


def replay_poses(body, imu_sub, body_model=None, restart_at=None):
    """Leg-odometry poses (T, 3) replayed from a recording's `body` (34) and `imu_sub` (5 x 6) arrays, anchors
    off. Tick 0 is the origin; the pose at tick t has consumed obs_1..obs_t. With restart_at (a takeover_fresh
    tick) a new odometry starts there, as the fresh bot's did."""
    from types import SimpleNamespace
    from bodymodel import BodyModel
    from odometry import Odometry
    bm = body_model or BodyModel()
    T = len(body)
    poses = np.zeros((T, 3), np.float64)
    odo = Odometry(bm)
    for t in range(1, T):
        if restart_at is not None and t == restart_at:
            odo = Odometry(bm)
            poses[t] = 0.0
            continue
        obs = SimpleNamespace(body={"gyro_substeps": imu_sub[t, :, :3], "gravity_substeps": imu_sub[t, :, 3:],
                                    "joint_pos": body[t, 6:20], "gravity": body[t, 3:6]})
        poses[t] = odo.update(obs)
    return poses


def vectors_from_recording(memory, poses, restart_at=None):
    """Per-tick vectors (T, DIM) and last-seen ticks (T, 8) from a recording's memory snapshots and replayed poses."""
    T = len(memory)
    if len(poses) != T:
        raise ValueError("memory and poses must have the same tick count")
    V = np.zeros((T, DIM), np.float32)
    L = np.full((T, len(CLASSES)), -1, np.int64)
    st = BeliefState()
    for t in range(T):
        if restart_at is not None and t == restart_at:
            st = BeliefState()
        V[t], L[t] = belief_vector(memory[t], poses[t], t, st)
    return V, L
