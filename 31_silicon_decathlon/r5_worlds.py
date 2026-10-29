"""Two closed-loop tasks with no MLX imports. All controllers see the same sensors.

Car: original strict-wall track geometry and bicycle equations, integrated at
<=10 ms so actions can take effect between the original 100 ms observations.
Tracker: original 40-frame world, with an added bounded-speed pursuit cursor.
"""
import math
import numpy as np
from r5_numpy import render
from r5_tracker_sim import BASE, simulate_one, occ_visible_frac


def move_toward(position, target, speed, dt):
    delta = target-position
    norm = np.linalg.norm(delta, axis=-1, keepdims=True)
    return position + delta * np.minimum(1, speed*dt/np.maximum(norm, 1e-12))


class CarWorld:
    hz = 10.0

    def __init__(self, seed, split="fresh"):
        from models.grand_prix_sim import make_track
        self.track = make_track(seed % 12 if split == "old" else seed, "heldout")
        self.xy = self.track.center[0].copy()
        self.yaw = float(np.arctan2(*self.track.tangent[0][::-1]))
        self.v = self.t = self.progress = self.lat = 0.0
        self.idx = 0
        self.frac = 0.0
        self.action = np.zeros(2)
        self.finished = self.crashed = False
        self.finish_time = self.crash_time = None
        self.trace = []
        self._next_trace = 0.0

    def advance(self, target_time):
        while self.t < target_time - 1e-10:
            dt = min(0.01, target_time-self.t)
            if not (self.finished or self.crashed):
                steer, pedal = self.action
                a = np.clip(pedal*(6 if pedal > 0 else 12), -9.81, 9.81)-6/1600*self.v**2
                vn = max(0, self.v+a*dt)
                vm = (vn+self.v)/2
                omega = np.clip(vm*math.tan(steer*0.45)/2.6, -9.81/max(vm, 1), 9.81/max(vm, 1))
                self.yaw += omega*dt
                self.xy += vm*dt*np.array([math.cos(self.yaw), math.sin(self.yaw)])
                self.v = vn
                tr = self.track
                ii = (self.idx + np.arange(-4, 7)) % len(tr.center)
                aa, bb = tr.center[ii], tr.center[(ii+1) % len(tr.center)]
                vv = bb-aa
                ff = np.clip(np.sum((self.xy-aa)*vv, axis=1)/np.sum(vv*vv, axis=1), 0, 1)
                jj = np.argmin(np.sum((aa+ff[:, None]*vv-self.xy)**2, axis=1))
                ni, nf = int(ii[jj]), float(ff[jj])
                spacing = tr.length/len(tr.center)
                ds = ((ni+nf-self.idx-self.frac)*spacing+tr.length/2) % tr.length-tr.length/2
                self.progress += ds
                self.idx, self.frac = ni, nf
                d = self.xy-aa[jj]
                self.lat = float((vv[jj, 0]*d[1]-vv[jj, 1]*d[0])/np.linalg.norm(vv[jj]))
                if abs(self.lat) > tr.half_width-1:
                    self.crashed, self.crash_time = True, self.t+dt
                elif self.progress >= tr.length:
                    self.finished, self.finish_time = True, self.t+dt
            self.t += dt
            if self.t >= self._next_trace:
                self.trace.append([self.t, *self.xy.tolist(), self.v, self.progress,
                                   int(self.crashed), int(self.finished)])
                self._next_trace += 0.1

    def capture(self, tick):
        tr = self.track
        aa = np.concatenate([tr.left, tr.right])
        bb = np.concatenate([np.roll(tr.left, -1, axis=0), np.roll(tr.right, -1, axis=0)])
        seg = bb-aa
        ray = self.yaw+np.deg2rad([-90, -60, -35, -18, -6, 6, 18, 35, 60, 90])
        dr = np.stack([np.cos(ray), np.sin(ray)], 1)
        off = aa-self.xy
        den = dr[:, 0, None]*seg[None, :, 1]-dr[:, 1, None]*seg[None, :, 0]
        good = np.abs(den) > 1e-12
        safe = np.where(good, den, 1)
        rr = (off[:, 0]*seg[:, 1]-off[:, 1]*seg[:, 0])[None]/safe
        u = (off[None, :, 0]*dr[:, 1, None]-off[None, :, 1]*dr[:, 0, None])/safe
        distance = np.where(good & (rr >= 0) & (u >= 0) & (u <= 1), rr, 60).min(1)
        return {"obs": np.r_[np.minimum(distance, 60)/60, self.v/30, self.action].astype(np.float32)[None]}

    def apply(self, out):
        a = np.asarray(out["action"], dtype=float).reshape(2)
        if not np.isfinite(a).all():
            raise ValueError("Nonfinite car action")
        self.action = np.clip(a, -1, 1)

    def score(self):
        return dict(finished=self.finished, crashed=self.crashed, finish_s=self.finish_time,
                    crash_s=self.crash_time, progress_fraction=float(self.progress/self.track.length),
                    horizon_s=self.t)

    def geometry(self):
        return dict(left=self.track.left.tolist(), right=self.track.right.tolist())


class TrackerWorld:
    hz = 30.0

    def __init__(self, seed, split="fresh"):
        self.seed = seed
        if split == "old":
            with np.load("../15_object_permanence/data/test_long.npz") as z:
                i = 500+seed % 50
                self.pos, self.present, self.occ, self.colors = [z[k][i].copy() for k in
                                                               ("pos", "present", "occ", "colors")]
        else:
            cfg = dict(BASE, n_occ=(1, 1), occ_w=(18, 26), min_long_occ=8)
            self.pos, self.present, self.occ, self.colors, _ = simulate_one(np.random.default_rng(seed), cfg)
        self.vis = occ_visible_frac(self.pos[..., 0], [o for o in self.occ if o[0] >= 0])
        self.t = 0.0
        # Equal starting information, no ground-truth initialization for either arm.
        self.cursor = np.full((3, 2), 32.0)
        self.target = self.cursor.copy()
        self.estimate = self.cursor.copy()
        self.trace = []
        self._next_trace = 0.0
        self.err_time = self.track_time = self.hidden_time = self.hidden_err = 0.0
        self.close_time = 0.0

    def truth(self, t):
        frame = min(t*self.hz, 39)
        i = int(frame)
        f = frame-i
        return self.pos[i]*(1-f)+self.pos[min(i+1, 39)]*f

    def advance(self, target_time):
        while self.t < target_time - 1e-10:
            dt = min(0.005, target_time-self.t)
            self.cursor = move_toward(self.cursor, self.target, speed=90, dt=dt)
            self.t += dt
            truth = self.truth(self.t)
            pursuit = np.linalg.norm(self.cursor-truth, axis=1)
            tracking = np.linalg.norm(self.estimate-truth, axis=1)
            live = self.present
            self.err_time += float(pursuit[live].mean())*dt
            self.track_time += float(tracking[live].mean())*dt
            self.close_time += float((pursuit[live] <= 5).mean())*dt
            hidden = live & (self.vis[min(int(self.t*self.hz), 39)] == 0)
            self.hidden_time += int(hidden.sum())*dt
            self.hidden_err += float(tracking[hidden].sum())*dt
            if self.t >= self._next_trace:
                self.trace.append([self.t, truth.tolist(), self.cursor.tolist(), self.estimate.tolist()])
                self._next_trace += 1/self.hz

    def capture(self, tick):
        # Same noise for the same nominal frame across arms; rendered at actual world time.
        frame = render(self.truth(self.t)[None, None], self.present[None], self.occ[None],
                       self.colors[None], rng=np.random.default_rng(self.seed*100003+tick))
        return {"frame": np.ascontiguousarray(frame[:, 0])}

    def apply(self, out):
        raw = np.asarray(out["out"])
        assert raw.shape == (1, 3, 5)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite tracker output")
        self.estimate = raw[0, :, :2].astype(float)*32+32
        self.target = np.clip(self.estimate, 0, 64)

    def score(self):
        return dict(pursuit_error_px=self.err_time/max(self.t, 1e-12),
                    tracking_error_px=self.track_time/max(self.t, 1e-12),
                    hidden_tracking_error_px=self.hidden_err/self.hidden_time if self.hidden_time else None,
                    within_5px_fraction=self.close_time/max(self.t, 1e-12), horizon_s=self.t)

    def geometry(self):
        return dict(occ=self.occ.tolist(), present=self.present.tolist())


def make_world(event, seed, split="fresh"):
    return {"car": CarWorld, "tracker": TrackerWorld}[event](seed, split)
