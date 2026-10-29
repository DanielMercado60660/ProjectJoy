"""Queue-level traffic simulator for a 2x2 grid of signalised intersections (batched, numpy).

Layout (north is up):

            col 0 ("Elm St", medium)    col 1 ("Quiet Lane", side street)
    row 0   I0 ------------------------ I1        <- "1st Ave" arterial (heavy, east-west)
            |                           |
    row 1   I2 ------------------------ I3        <- "2nd Ave" arterial (heavy, east-west)

Every intersection has four incoming lanes, named by travel direction:
    0 = southbound (from north), 1 = northbound (from south)   -> served by phase 0 (NS green)
    2 = eastbound  (from west),  3 = westbound  (from east)    -> served by phase 1 (EW green)
Lane index = 4 * intersection + direction. Vehicles drive straight through (no turns).
8 lanes start at the edge of the map (Poisson arrivals); 8 are internal links fed by a neighbour.

Time step = 2 s. Saturation flow = 1 vehicle per step per green lane. A phase change costs
YELLOW steps of all-red. Internal links take TRAVEL steps to drive and hold at most LINK_CAP
vehicles; when a link is full, its upstream green lane cannot discharge ("spillback").
The two phases never conflict: the simulator only ever serves one phase at a time, so the
policy cannot pick an unsafe signal state.
"""
import numpy as np

DT = 2.0              # seconds per step
T_STEPS = 1800        # one hour
K = 5                 # sim steps per control decision (10 s)
YELLOW = 2            # all-red steps when switching
TRAVEL = 5            # steps to drive an internal link (10 s)
LINK_CAP = 16         # vehicles an internal link can hold
EDGE_CAP = 400        # edge queues (practically unbounded; overflow is counted as rejected)
CAP = 512             # ring-buffer size
N_INT, N_LANES = 4, 16
WAIT_BIN_S = 2.0
WAIT_NBINS = 1200     # histogram up to 2400 s


def _layout():
    down = np.full(N_LANES, -1, dtype=np.int64)
    for i in range(N_INT):
        r, c = divmod(i, 2)
        if r == 0: down[4 * i + 0] = 4 * (i + 2) + 0     # southbound I0->I2, I1->I3
        if r == 1: down[4 * i + 1] = 4 * (i - 2) + 1     # northbound
        if c == 0: down[4 * i + 2] = 4 * (i + 1) + 2     # eastbound
        if c == 1: down[4 * i + 3] = 4 * (i - 1) + 3     # westbound
    upstream = np.full(N_LANES, -1, dtype=np.int64)
    for l in range(N_LANES):
        if down[l] >= 0:
            upstream[down[l]] = l
    edge = np.where(upstream < 0)[0]          # 8 lanes fed from outside
    return down, upstream, edge


DOWN, UPSTREAM, EDGE_LANES = _layout()
INTERNAL_SRC = np.where(DOWN >= 0)[0]
LANE_CAP = np.where(UPSTREAM < 0, EDGE_CAP, LINK_CAP)
LANE_PHASE = np.array([0, 0, 1, 1] * N_INT)   # which phase serves each lane
LANE_INT = np.repeat(np.arange(N_INT), 4)
# The quiet side street: north/south lanes at the column-1 intersections (I1, I3).
QUIET_LANES = np.array([4 * 1 + 0, 4 * 1 + 1, 4 * 3 + 0, 4 * 3 + 1])
DIR_NAMES = ["S-bound", "N-bound", "E-bound", "W-bound"]
LANE_NAMES = [f"I{i} {DIR_NAMES[d]}" for i in range(N_INT) for d in range(4)]

# Base demand per edge lane, vehicles per step (x1800 = veh/h).
def base_rates():
    rate = np.zeros(N_LANES)
    for l in EDGE_LANES:
        i, d = divmod(l, 4)
        c = i % 2
        if d >= 2:
            rate[l] = 0.30                     # arterial: ~540 veh/h per direction
        elif c == 0:
            rate[l] = 0.17                     # Elm St: ~310 veh/h
        else:
            rate[l] = 0.03                     # Quiet Lane: ~54 veh/h (one car every ~67 s)
    return rate


def demand(pattern, rng=None):
    """Arrival-rate table (T_STEPS, 16) for one episode.

    pattern: 'steady', 'rush', or 'train' (random scale, random rush peak/time/asymmetry).
    """
    t = np.arange(T_STEPS)
    rate = base_rates()
    art = np.array([LANE_PHASE[l] == 1 for l in range(N_LANES)])
    elm = np.array([(LANE_PHASE[l] == 0) and (LANE_INT[l] % 2 == 0) for l in range(N_LANES)])
    if pattern == "steady":
        scale, peak, tc, east_bias = 1.0, 1.0, 900, 0.0
    elif pattern == "rush":
        scale, peak, tc, east_bias = 1.0, 1.9, 900, 0.0
    elif pattern == "train":
        scale = rng.uniform(0.75, 1.15)
        peak = rng.uniform(1.0, 2.1) if rng.random() < 0.75 else 1.0
        tc = rng.uniform(450, 1350)
        east_bias = rng.uniform(-0.3, 0.3)
    else:
        raise ValueError(pattern)
    bump = 1.0 + (peak - 1.0) * np.exp(-0.5 * ((t - tc) / 300.0) ** 2)   # sigma = 10 min
    R = np.tile(rate * scale, (T_STEPS, 1))
    R[:, art] *= bump[:, None]
    R[:, elm] *= (1.0 + 0.35 * (bump - 1.0))[:, None]
    for l in EDGE_LANES:
        if LANE_PHASE[l] == 1:
            R[:, l] *= (1.0 + east_bias) if (l % 4) == 2 else (1.0 - east_bias)
    return R


class Grid:
    """B independent copies of the 2x2 grid, stepped together."""

    def __init__(self, B, record=False):
        self.B = B
        self.record = record

    def reset(self, rates, seed):
        """rates: (B, T_STEPS, 16) arrival rates. seed: int for the arrival RNG."""
        B = self.B
        self.rates = rates
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.buf = np.zeros((B, N_LANES, CAP), np.float64)   # ready time of each queued vehicle
        self.head = np.zeros((B, N_LANES), np.int64)
        self.cnt = np.zeros((B, N_LANES), np.int64)           # vehicles on lane (driving + queued)
        self.ready = np.zeros((B, N_LANES), np.int64)         # vehicles at the stop line
        self.delay = np.zeros((B, N_LANES, TRAVEL + 1), np.int64)
        self.phase = np.ones((B, N_INT), np.int64)            # start with arterials green
        self.yellow = np.zeros((B, N_INT), np.int64)
        self.since_switch = np.zeros((B, N_INT), np.int64)
        # stats
        self.dep = np.zeros((B, N_LANES), np.int64)
        self.wsum = np.zeros((B, N_LANES))
        self.wmax = np.zeros((B, N_LANES))
        self.exited = np.zeros(B, np.int64)
        self.entered = np.zeros(B, np.int64)
        self.rejected = np.zeros(B, np.int64)
        self.blocked = np.zeros((B, N_LANES), np.int64)
        self.full_steps = np.zeros((B, N_LANES), np.int64)
        self.switches = np.zeros((B, N_INT), np.int64)
        if self.record:
            self.hist = np.zeros((B, N_LANES, WAIT_NBINS), np.int64)
            self.trace_q = []      # per step (B,16) vehicles at the stop line
            self.trace_c = []      # per step (B,16) vehicles on lane
            self.trace_p = []      # per step (B,4) phase (-1 while yellow)
            self.trace_w = []      # per step (B,16) front-vehicle wait, s
        return self

    # ----- helpers -----
    def head_wait(self):
        """Seconds the vehicle at the front of each lane has waited (0 if nobody is waiting)."""
        ht = np.take_along_axis(self.buf, self.head[..., None], axis=2)[..., 0]
        return np.where(self.ready > 0, (self.t - ht) * DT, 0.0)

    def green(self):
        g = self.phase[:, LANE_INT] == LANE_PHASE[None, :]
        return g & (self.yellow[:, LANE_INT] == 0)

    def _push(self, mask, val):
        ok = mask & (self.cnt < LANE_CAP[None, :])
        b, l = np.nonzero(ok)
        idx = (self.head[b, l] + self.cnt[b, l]) % CAP
        self.buf[b, l, idx] = val
        self.cnt[b, l] += 1
        return ok, mask & ~ok

    # ----- one simulation step -----
    def sim_step(self):
        t = self.t
        slot = t % (TRAVEL + 1)
        self.ready += self.delay[:, :, slot]
        self.delay[:, :, slot] = 0
        # 1. discharge
        g = self.green()
        space = np.ones((self.B, N_LANES), bool)
        space[:, INTERNAL_SRC] = self.cnt[:, DOWN[INTERNAL_SRC]] < LINK_CAP
        want = g & (self.ready > 0)
        move = want & space
        self.blocked += want & ~space
        b, l = np.nonzero(move)
        ht = self.buf[b, l, self.head[b, l]]
        w = (t - ht) * DT
        np.add.at(self.wsum, (b, l), w)
        np.maximum.at(self.wmax, (b, l), w)
        if self.record:
            bins = np.minimum((w / WAIT_BIN_S).astype(np.int64), WAIT_NBINS - 1)
            np.add.at(self.hist, (b, l, bins), 1)
        self.dep += move
        self.head[b, l] = (self.head[b, l] + 1) % CAP
        self.cnt -= move
        self.ready -= move
        exit_mask = move.copy()
        exit_mask[:, INTERNAL_SRC] = False
        self.exited += exit_mask.sum(1)
        # vehicles moving on to the next link
        md = np.zeros_like(move)
        md[:, DOWN[INTERNAL_SRC]] = move[:, INTERNAL_SRC]
        self._push(md, t + TRAVEL)
        self.delay[:, :, (t + TRAVEL) % (TRAVEL + 1)] += md
        # 2. arrivals at the edge
        lam = self.rates[:, t, :][:, EDGE_LANES]
        n = self.rng.poisson(lam)
        for k in range(int(n.max(initial=0))):
            m = np.zeros((self.B, N_LANES), bool)
            m[:, EDGE_LANES] = n > k
            ok, rej = self._push(m, t)
            self.ready += ok
            self.entered += ok.sum(1)
            self.rejected += rej.sum(1)
        self.full_steps[:, INTERNAL_SRC] += self.cnt[:, DOWN[INTERNAL_SRC]] >= LINK_CAP
        # 3. signal clocks
        self.yellow = np.maximum(self.yellow - 1, 0)
        self.since_switch += 1
        self.t += 1
        if self.record:
            self.trace_q.append(self.ready.copy())
            self.trace_c.append(self.cnt.copy())
            self.trace_p.append(np.where(self.yellow > 0, -1, self.phase))
            self.trace_w.append(self.head_wait())

    def step_decision(self, want_phase):
        """Apply desired phase per intersection, run K sim steps.

        Returns per-intersection reward pieces summed over the K steps:
        (queue, sq_wait) where queue = vehicles at the stop line, sq_wait = sum (head wait / 60 s)^2.
        """
        want_phase = np.asarray(want_phase, np.int64)
        sw = (want_phase != self.phase) & (self.yellow == 0)
        self.phase = np.where(sw, want_phase, self.phase)
        self.yellow = np.where(sw, YELLOW, self.yellow)
        self.since_switch = np.where(sw, 0, self.since_switch)
        self.switches += sw
        q = np.zeros((self.B, N_INT))
        sq = np.zeros((self.B, N_INT))
        for _ in range(K):
            if self.t >= T_STEPS:
                break
            self.sim_step()
            q += self.ready.reshape(self.B, N_INT, 4).sum(2)
            hw = np.minimum(self.head_wait() / 60.0, 5.0)     # capped at 5 min for the reward
            sq += (hw ** 2).reshape(self.B, N_INT, 4).sum(2)
        return q, sq

    @property
    def done(self):
        return self.t >= T_STEPS

    # ----- observation for learned controllers -----
    OBS_DIM = 4 * 4 + 2 + 1 + 1 + 4

    def obs(self):
        B = self.B
        hw = self.head_wait()
        down_occ = np.zeros((B, N_LANES))
        down_occ[:, INTERNAL_SRC] = self.cnt[:, DOWN[INTERNAL_SRC]] / LINK_CAP
        feats = np.stack([
            self.ready / 10.0,
            (self.cnt - self.ready) / 10.0,     # vehicles still driving towards the stop line
            hw / 60.0,
            down_occ,
        ], -1).reshape(B, N_INT, 16)
        ph = np.eye(2)[self.phase]
        extra = np.concatenate([
            ph,
            (self.yellow > 0)[..., None].astype(float),
            np.minimum(self.since_switch * DT / 60.0, 3.0)[..., None],
            np.broadcast_to(np.eye(N_INT), (B, N_INT, N_INT)),
        ], -1)
        return np.concatenate([np.minimum(feats, 10.0), extra], -1).astype(np.float32)

    # ----- end-of-episode metrics -----
    def metrics(self):
        """Per-env dict of arrays. Waits in seconds. Vehicles still waiting at the end count with
        their (censored) wait so far, so a starved street cannot hide."""
        hw_end = self.head_wait()
        wmax = np.maximum(self.wmax, hw_end)
        dep = np.maximum(self.dep, 1)
        lane_mean = self.wsum / dep
        lane_mean = np.where(self.dep > 0, lane_mean, hw_end)
        tot_dep = self.dep.sum(1)
        m = dict(
            throughput=self.exited.astype(float),                       # veh exited / hour
            entered=self.entered.astype(float),
            in_network_end=self.cnt.sum(1).astype(float),
            mean_wait=self.wsum.sum(1) / np.maximum(tot_dep, 1),        # s per signal crossing
            max_wait=wmax.max(1),
            quiet_mean_wait=self.wsum[:, QUIET_LANES].sum(1) / np.maximum(self.dep[:, QUIET_LANES].sum(1), 1),
            quiet_max_wait=wmax[:, QUIET_LANES].max(1),
            quiet_stuck_end=self.ready[:, QUIET_LANES].sum(1).astype(float),
            worst_approach_mean_wait=lane_mean.max(1),
            jain_fairness=(lane_mean.sum(1) ** 2) / (N_LANES * (lane_mean ** 2).sum(1) + 1e-9),
            spillback_blocked_steps=self.blocked.sum(1).astype(float),
            link_full_frac=self.full_steps[:, INTERNAL_SRC].mean(1) / T_STEPS,
            rejected=self.rejected.astype(float),
            switches_per_int=self.switches.mean(1).astype(float),
        )
        m["lane_mean_wait"] = lane_mean
        m["lane_max_wait"] = wmax
        if self.record:
            m["p95_wait"] = np.array([_pct(self.hist[b].sum(0), 95) for b in range(self.B)])
        return m


def _pct(h, q):
    c = np.cumsum(h)
    if c[-1] == 0:
        return 0.0
    return float(np.searchsorted(c, q / 100.0 * c[-1]) * WAIT_BIN_S)


def run_episode(controller, rates, seed, record=False):
    """Run one batch of episodes with a controller(env) -> phases (B,4). Returns the env."""
    env = Grid(rates.shape[0], record=record).reset(rates, seed)
    while not env.done:
        env.step_decision(controller(env))
    return env
