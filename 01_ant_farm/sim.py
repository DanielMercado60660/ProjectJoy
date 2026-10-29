"""The Ant Farm simulator: many parallel grid terrariums, many ants each.

Everything that changes during an episode lives in a `World` (plain numpy arrays,
including a per-world xorshift RNG state), so a world can be saved, restored, forked
and replayed bit-for-bit. The hot loops are numba, parallel over worlds only, with
ants inside one world processed in a fixed order -> results do not depend on the
number of threads.

Interface used by every colony (hand-written, PPO, ES, JS port):
    obs   = observe(world)                    float32 [B, N, OBS_DIM]  (local only)
    probs = some_policy(obs)                  move probs [B,N,7], deposit probs [B,N,3]
    move, dep = sample_actions(world, pm, pd) (uses the world's own RNG)
    ev = step(world, move, dep)               per-ant events for rewards / metrics

Actions
    move: 0..4 = turn -90,-45,0,+45,+90 degrees then step forward; 5 = U-turn and step;
          6 = stay (rest, cheaper than moving)
    deposit: 0 = none, 1 = "home" pheromone, 2 = "food" pheromone (both cost a little energy)
Pick-up and drop are automatic contact events: an empty-handed ant that steps on food
picks up one unit; a carrying ant that enters the nest drops it (delivery).
"""
import numpy as np
from numba import njit, prange

# ---------------------------------------------------------------- constants
H = W = 64
NEST_R = 2            # nest is a disc of this radius (13 cells)
FOOD_R = 2            # each food source is a disc of this radius
FOOD_PER_CELL = 40    # -> ~520 units per source (not depleted by a good colony in 1000 steps)
DIRS = np.array([[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]], np.int32)  # (dy,dx), +1 = clockwise
TURNS = np.array([-2, -1, 0, 1, 2, 4, 0], np.int32)
N_MOVE, N_DEP = 7, 3
STAY = 6

C_MOVE = 1.0 / 300.0  # energy per step moved (a full ant can walk 300 steps)
C_STAY = 1.0 / 600.0   # energy per step resting (basal metabolism: a colony that only rests starves by ~step 900)
C_DEP = 1.0 / 3000.0  # energy per pheromone deposit
FOOD_VALUE = 1.0       # energy a delivered food unit adds to the colony store
STORE0_PER_ANT = 0.5   # initial colony store (energy units) per ant
DEP_AMOUNT = 1.0
P_MAX = 30.0
EVAP = np.array([0.006, 0.012], np.float32)  # per-step evaporation [home, food]
DIFF = 0.10                                   # fraction exchanged with 4-neighbours per step

N_SENS_D, N_SENS_K, N_CH = 2, 5, 5   # sensor distance 1..2, 5 forward directions, channels
SENS_DIM = N_SENS_D * N_SENS_K * N_CH  # 50
OBS_DIM = SENS_DIM + 9
# obs index helpers
CH_HOME, CH_FOOD, CH_OBST, CH_FOODITEM, CH_NEST = range(5)
I_CUR_HOME, I_CUR_FOOD, I_CARRY, I_ENERGY, I_CMP_F, I_CMP_S, I_NDIST, I_ATNEST, I_CLOCK = range(SENS_DIM, SENS_DIM + 9)


def sidx(d, k, c):
    """obs index of sensor at distance d (0->1 cell, 1->2 cells), turn k (0..4 = -2..+2), channel c"""
    return (d * N_SENS_K + k) * N_CH + c


# ---------------------------------------------------------------- rng
@njit(inline="always")
def _rand_u64(rng, b):
    x = rng[b]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    rng[b] = x
    return x * np.uint64(2685821657736338717)


@njit(inline="always")
def _rand_f(rng, b):
    return float(_rand_u64(rng, b) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


def _seed_state(seed):
    # splitmix64 so nearby seeds give unrelated streams; never 0
    z = (int(seed) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    z ^= z >> 31
    return np.uint64(z or 1)


# ---------------------------------------------------------------- world
class World:
    """Batch of B terrariums with N ants each. All state is in numpy arrays."""
    FIELDS = ["obst", "food", "pher", "nest", "pos", "head", "carry", "energy", "alive", "clock",
              "store", "rng", "t", "delivered", "picked", "spent", "deaths", "food0", "seeds", "fsrc"]

    def __init__(self, B, N):
        self.B, self.N = B, N
        self.obst = np.zeros((B, H, W), np.uint8)
        self.food = np.zeros((B, H, W), np.int16)
        self.pher = np.zeros((B, 2, H, W), np.float32)
        self.nest = np.zeros((B, 2), np.int32)
        self.pos = np.zeros((B, N, 2), np.int32)
        self.head = np.zeros((B, N), np.int32)
        self.carry = np.zeros((B, N), np.uint8)
        self.energy = np.ones((B, N), np.float32)
        self.alive = np.ones((B, N), np.uint8)
        self.clock = np.zeros((B, N), np.int32)
        self.store = np.zeros(B, np.float32)
        self.rng = np.ones(B, np.uint64)
        self.t = np.zeros(B, np.int32)
        self.delivered = np.zeros(B, np.float64)
        self.picked = np.zeros(B, np.float64)
        self.spent = np.zeros(B, np.float64)
        self.deaths = np.zeros(B, np.int32)
        self.food0 = np.zeros(B, np.int32)
        self.seeds = np.zeros(B, np.int64)
        self.fsrc = np.zeros((B, 2, 2), np.int32)  # food-source centres: for metrics/route-blocking only, never observed

    # -- persistence -------------------------------------------------------
    def state_dict(self):
        return {k: getattr(self, k).copy() for k in self.FIELDS}

    def load_state_dict(self, d):
        for k in self.FIELDS:
            setattr(self, k, d[k].copy())
        self.B, self.N = self.pos.shape[:2]

    def save(self, path):
        np.savez_compressed(path, **self.state_dict())

    @classmethod
    def load(cls, path):
        d = dict(np.load(path))
        w = cls(d["pos"].shape[0], d["pos"].shape[1])
        w.load_state_dict(d)
        return w

    def copy(self):
        w = World(self.B, self.N)
        w.load_state_dict(self.state_dict())
        return w

    def subset(self, idx):
        w = World(len(idx), self.N)
        w.load_state_dict({k: getattr(self, k)[idx].copy() for k in self.FIELDS})
        return w

    def nest_mask(self, b):
        yy, xx = np.mgrid[0:H, 0:W]
        return (yy - self.nest[b, 0]) ** 2 + (xx - self.nest[b, 1]) ** 2 <= NEST_R ** 2


# ---------------------------------------------------------------- layout generation
def _disc(cy, cx, r):
    yy, xx = np.mgrid[0:H, 0:W]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def _reachable(obst, src, dsts):
    """BFS with 8-connectivity on free cells."""
    seen = np.zeros_like(obst, bool)
    q = [tuple(src)]
    seen[src[0], src[1]] = True
    i = 0
    while i < len(q):
        y, x = q[i]; i += 1
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and not obst[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return all(seen[d[0], d[1]] for d in dsts)


def make_layout(seed, family="train"):
    """Procedural terrarium. Families:
    train   : food 10-22 cells from nest, 2-5 random rectangular rocks
    heldout : food 24-30 cells from nest (farther than any training layout), long thin walls
              (a different obstacle family), seeds drawn from a disjoint range by the caller
    """
    r = np.random.default_rng(seed)
    for _ in range(200):
        obst = np.zeros((H, W), np.uint8)
        nest = r.integers(18, 46, size=2) if family == "train" else r.integers(22, 42, size=2)
        dmin, dmax = (10, 22) if family == "train" else (24, 30)
        foods, a0 = [], r.uniform(0, 2 * np.pi)
        ok = True
        for j in range(2):
            for _try in range(100):
                a = a0 + (0 if j == 0 else r.uniform(np.pi / 3, 5 * np.pi / 3))
                d = r.uniform(dmin, dmax)
                fy, fx = int(round(nest[0] + d * np.sin(a))), int(round(nest[1] + d * np.cos(a)))
                if 4 <= fy < H - 4 and 4 <= fx < W - 4:
                    foods.append((fy, fx)); break
                a0 = r.uniform(0, 2 * np.pi)
            else:
                ok = False
        if not ok:
            continue
        keep = _disc(*nest, NEST_R + 3)
        for f in foods:
            keep |= _disc(*f, FOOD_R + 2)
        if family == "train":
            for _k in range(r.integers(2, 6)):
                h, w = r.integers(2, 5), r.integers(4, 15)
                if r.random() < 0.5:
                    h, w = w, h
                y, x = r.integers(0, H - h), r.integers(0, W - w)
                obst[y:y + h, x:x + w] = 1
        else:
            for _k in range(r.integers(2, 4)):
                L = r.integers(14, 26)
                y, x = r.integers(4, H - 4), r.integers(4, W - 4)
                if r.random() < 0.5:
                    obst[y:y + 2, max(0, x - L // 2):x + L // 2] = 1
                else:
                    obst[max(0, y - L // 2):y + L // 2, x:x + 2] = 1
        obst[keep] = 0
        if _reachable(obst, nest, foods):
            food = np.zeros((H, W), np.int16)
            for f in foods:
                food[_disc(*f, FOOD_R) & (obst == 0)] = FOOD_PER_CELL
            return obst, food, nest.astype(np.int32), foods
    raise RuntimeError("could not generate layout")


def reset(world, seeds, family="train", layouts=None):
    """(Re)initialise every world from integer seeds (layout + ant RNG both derive from the seed)."""
    B, N = world.B, world.N
    for b in range(B):
        if layouts is not None:
            obst, food, nest, foods = layouts[b]
        else:
            obst, food, nest, foods = make_layout(int(seeds[b]), family)
        world.fsrc[b] = np.asarray(foods, np.int32)
        world.obst[b] = obst
        world.food[b] = food
        world.nest[b] = nest
        world.rng[b] = _seed_state(int(seeds[b]) * 7919 + 17)
        world.seeds[b] = int(seeds[b])
        world.food0[b] = int(food.sum())
    world.pher[:] = 0
    world.carry[:] = 0
    world.energy[:] = 1.0
    world.alive[:] = 1
    world.clock[:] = 0
    world.store[:] = STORE0_PER_ANT * N
    world.t[:] = 0
    world.delivered[:] = 0; world.picked[:] = 0; world.spent[:] = 0; world.deaths[:] = 0
    _place_ants(world.pos, world.head, world.nest, world.rng)
    return world


@njit(parallel=True, cache=True)
def _place_ants(pos, head, nest, rng):
    B, N = head.shape
    for b in prange(B):
        for i in range(N):
            while True:
                dy = int(_rand_f(rng, b) * (2 * NEST_R + 1)) - NEST_R
                dx = int(_rand_f(rng, b) * (2 * NEST_R + 1)) - NEST_R
                if dy * dy + dx * dx <= NEST_R * NEST_R:
                    break
            pos[b, i, 0] = nest[b, 0] + dy
            pos[b, i, 1] = nest[b, 1] + dx
            head[b, i] = int(_rand_f(rng, b) * 8)


# ---------------------------------------------------------------- observation
@njit(inline="always")
def _ptrans(p):
    return np.log1p(20.0 * p) * 0.25


@njit(parallel=True, cache=True)
def _observe(obst, food, pher, nest, pos, head, carry, energy, alive, clock, dirs, out):
    B, N = head.shape
    for b in prange(B):
        ny, nx = nest[b, 0], nest[b, 1]
        for i in range(N):
            for j in range(out.shape[2]):
                out[b, i, j] = 0.0
            if alive[b, i] == 0:
                continue
            y, x, h = pos[b, i, 0], pos[b, i, 1], head[b, i]
            for d in range(2):
                for k in range(5):
                    hh = (h + k - 2) % 8
                    cy = y + (d + 1) * dirs[hh, 0]
                    cx = x + (d + 1) * dirs[hh, 1]
                    base = (d * 5 + k) * 5
                    if cy < 0 or cy >= H or cx < 0 or cx >= W or obst[b, cy, cx]:
                        out[b, i, base + 2] = 1.0
                        continue
                    out[b, i, base + 0] = _ptrans(pher[b, 0, cy, cx])
                    out[b, i, base + 1] = _ptrans(pher[b, 1, cy, cx])
                    out[b, i, base + 3] = 1.0 if food[b, cy, cx] > 0 else 0.0
                    if (cy - ny) * (cy - ny) + (cx - nx) * (cx - nx) <= NEST_R * NEST_R:
                        out[b, i, base + 4] = 1.0
            o = 50
            out[b, i, o + 0] = _ptrans(pher[b, 0, y, x])
            out[b, i, o + 1] = _ptrans(pher[b, 1, y, x])
            out[b, i, o + 2] = float(carry[b, i])
            out[b, i, o + 3] = energy[b, i]
            # path-integration compass: unit vector to the nest in the ant's own frame
            vy, vx = float(ny - y), float(nx - x)
            dist = np.sqrt(vy * vy + vx * vx)
            if dist > 0:
                fy, fx = float(dirs[h, 0]), float(dirs[h, 1])
                fn = np.sqrt(fy * fy + fx * fx)
                fy /= fn; fx /= fn
                ry, rx = fx, -fy  # heading rotated +90deg (clockwise on screen, y down)
                out[b, i, o + 4] = (vy * fy + vx * fx) / dist
                out[b, i, o + 5] = (vy * ry + vx * rx) / dist
            out[b, i, o + 6] = dist / 32.0
            out[b, i, o + 7] = 1.0 if dist * dist <= NEST_R * NEST_R else 0.0
            out[b, i, o + 8] = min(clock[b, i], 400) / 100.0


def observe(world, out=None):
    if out is None:
        out = np.empty((world.B, world.N, OBS_DIM), np.float32)
    _observe(world.obst, world.food, world.pher, world.nest, world.pos, world.head, world.carry,
             world.energy, world.alive, world.clock, DIRS, out)
    return out


# ---------------------------------------------------------------- actions
@njit(parallel=True, cache=True)
def _sample(pm, pd, alive, rng, move, dep):
    B, N, A = pm.shape
    for b in prange(B):
        for i in range(N):
            if alive[b, i] == 0:
                move[b, i] = STAY; dep[b, i] = 0
                continue
            u = _rand_f(rng, b); c = 0.0; a = A - 1
            for j in range(A):
                c += pm[b, i, j]
                if u < c:
                    a = j; break
            move[b, i] = a
            u = _rand_f(rng, b); c = 0.0; a = pd.shape[2] - 1
            for j in range(pd.shape[2]):
                c += pd[b, i, j]
                if u < c:
                    a = j; break
            dep[b, i] = a


def sample_actions(world, pm, pd):
    move = np.empty((world.B, world.N), np.int32)
    dep = np.empty((world.B, world.N), np.int32)
    _sample(np.ascontiguousarray(pm, np.float32), np.ascontiguousarray(pd, np.float32), world.alive, world.rng, move, dep)
    return move, dep


# ---------------------------------------------------------------- dynamics
@njit(parallel=True, cache=True)
def _step(obst, food, pher, nest, pos, head, carry, energy, alive, clock, store, t,
          delivered, picked, spent, deaths, move, dep, dirs, turns, evap, diff, dep_on,
          ev_deliver, ev_pick, ev_die, ev_cost, tmp):
    B, N = head.shape
    for b in prange(B):
        ny, nx = nest[b, 0], nest[b, 1]
        for i in range(N):
            ev_deliver[b, i] = 0.0; ev_pick[b, i] = 0.0; ev_die[b, i] = 0.0; ev_cost[b, i] = 0.0
            if alive[b, i] == 0:
                continue
            y, x = pos[b, i, 0], pos[b, i, 1]
            a = move[b, i]
            cost = 0.0
            if a == STAY:
                cost += C_STAY
            else:
                h = (head[b, i] + turns[a]) % 8
                head[b, i] = h
                cy, cx = y + dirs[h, 0], x + dirs[h, 1]
                if 0 <= cy < H and 0 <= cx < W and obst[b, cy, cx] == 0:
                    y, x = cy, cx
                    pos[b, i, 0] = y; pos[b, i, 1] = x
                cost += C_MOVE
            dk = dep[b, i]
            if dk > 0 and dep_on:
                p = pher[b, dk - 1, y, x] + DEP_AMOUNT
                pher[b, dk - 1, y, x] = p if p < P_MAX else P_MAX
                cost += C_DEP
            clock[b, i] += 1
            at_nest = (y - ny) * (y - ny) + (x - nx) * (x - nx) <= NEST_R * NEST_R
            if carry[b, i] == 0 and food[b, y, x] > 0:
                food[b, y, x] -= 1
                carry[b, i] = 1
                clock[b, i] = 0
                ev_pick[b, i] = 1.0
                picked[b] += 1.0
            elif carry[b, i] == 1 and at_nest:
                carry[b, i] = 0
                store[b] += FOOD_VALUE
                ev_deliver[b, i] = 1.0
                delivered[b] += 1.0
            if at_nest:
                if carry[b, i] == 0:
                    clock[b, i] = 0
                need = 1.0 - energy[b, i]
                take = need if need < store[b] else store[b]
                if take > 0:
                    store[b] -= take
                    energy[b, i] += take
            energy[b, i] -= cost
            spent[b] += cost
            ev_cost[b, i] = cost
            if energy[b, i] <= 0.0:
                energy[b, i] = 0.0
                alive[b, i] = 0
                ev_die[b, i] = 1.0
                deaths[b] += 1
                if carry[b, i] == 1:  # the load is dropped where the ant falls
                    food[b, y, x] += 1
                    carry[b, i] = 0
        # pheromone: diffuse (reflecting at walls/edges) then evaporate
        for c in range(2):
            for yy in range(H):
                for xx in range(W):
                    if obst[b, yy, xx]:
                        tmp[b, yy, xx] = 0.0
                        continue
                    p = pher[b, c, yy, xx]
                    s = 0.0
                    s += pher[b, c, yy - 1, xx] if yy > 0 and obst[b, yy - 1, xx] == 0 else p
                    s += pher[b, c, yy + 1, xx] if yy < H - 1 and obst[b, yy + 1, xx] == 0 else p
                    s += pher[b, c, yy, xx - 1] if xx > 0 and obst[b, yy, xx - 1] == 0 else p
                    s += pher[b, c, yy, xx + 1] if xx < W - 1 and obst[b, yy, xx + 1] == 0 else p
                    v = ((1.0 - diff) * p + diff * 0.25 * s) * (1.0 - evap[c])
                    tmp[b, yy, xx] = v if v > 1e-4 else 0.0
            for yy in range(H):
                for xx in range(W):
                    pher[b, c, yy, xx] = tmp[b, yy, xx]
        t[b] += 1


class Events:
    def __init__(self, B, N):
        self.deliver = np.zeros((B, N), np.float32)
        self.pick = np.zeros((B, N), np.float32)
        self.die = np.zeros((B, N), np.float32)
        self.cost = np.zeros((B, N), np.float32)


def step(world, move, dep, ev=None, pheromones_on=True):
    if ev is None or ev.deliver.shape != (world.B, world.N):
        ev = Events(world.B, world.N)
    tmp = np.empty((world.B, H, W), np.float32)
    _step(world.obst, world.food, world.pher, world.nest, world.pos, world.head, world.carry,
          world.energy, world.alive, world.clock, world.store, world.t, world.delivered,
          world.picked, world.spent, world.deaths, np.ascontiguousarray(move, np.int32),
          np.ascontiguousarray(dep, np.int32), DIRS, TURNS, EVAP, np.float32(DIFF), bool(pheromones_on),
          ev.deliver, ev.pick, ev.die, ev.cost, tmp)
    return ev


# ---------------------------------------------------------------- route blocking
def block_route(world, b, half_len=9, food_xy=None):
    """Drop a 2-cell-thick wall across the line nest->food (perpendicular, centred on the
    midpoint). Default target = the food source with the most food-pheromone mass nearby
    (i.e. the trail the colony is actually using). Returns the wall mask."""
    nest = world.nest[b].astype(float)
    if food_xy is None:
        food_xy = busiest_source(world, b)
    f = np.asarray(food_xy, float)
    v = f - nest
    L = np.linalg.norm(v)
    u = v / L
    perp = np.array([-u[1], u[0]])
    mid = nest + 0.5 * v
    mask = np.zeros((H, W), bool)
    for s in np.linspace(-half_len, half_len, 8 * half_len + 1):
        for o in (-0.5, 0.0, 0.5):
            p = mid + s * perp + o * u
            yy, xx = int(round(p[0])), int(round(p[1]))
            if 0 <= yy < H and 0 <= xx < W:
                mask[yy, xx] = True
    # thicken to block diagonal slips
    m2 = mask.copy()
    m2[1:, :] |= mask[:-1, :]
    m2[:, 1:] |= mask[:, :-1]
    keep = _disc(*world.nest[b], NEST_R + 2)
    for (fy, fx) in food_sources(world, b):
        keep |= _disc(fy, fx, FOOD_R + 2)
    m2 &= ~keep
    world.obst[b][m2] = 1
    world.pher[b][:, m2] = 0
    return m2


def food_sources(world, b):
    """Centres of the food discs. Used only by evaluation code (route blocking, metrics),
    never by any policy."""
    return [tuple(int(v) for v in f) for f in world.fsrc[b]]


def busiest_source(world, b):
    best, bf = -1.0, None
    for (fy, fx) in food_sources(world, b):
        m = _disc(fy, fx, 8)
        s = world.pher[b, 1][m].sum()
        if s > best:
            best, bf = s, (fy, fx)
    return bf


def make_worlds(seeds, N, family="train"):
    w = World(len(seeds), N)
    return reset(w, np.asarray(seeds), family)
