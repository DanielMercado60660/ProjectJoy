"""Rooms & Mechanisms: a vectorized, reward-free gridworld (numpy + numba).

Layout: 3x3 rooms, each with a 5x5 interior, walls 1 cell thick -> 19x19 grid.
The agent sees only a 7x7 egocentric window (not rotated; actions are absolute moves).

Room graph (rooms named Rrc):

    R00 --open-- R01 ==button-door== R02
     |            #box in doorway      |
    open         R11 (dead end,       open
     |               lever C)          |
    R10          (walls)              R12
     #locked door (key from R02)       #button-door (button in R20!)
    R20 --open-- R21                  R22 (lever D)

Mechanisms, easy -> hard:
  0 lever A (R00) lights lamp A in the same room
  1 button B (R01) opens the adjacent door R01|R02
  2 box     sits in the R01|R11 doorway; push it down into R11 to get through
  3 key     lies in R02
  4 unlock  the locked door R10|R20 (walk into it while holding the key; key consumed)
  5 button E (R20) opens the far door R12|R22 (effect is not visible from R20)
  6 lever C (R11) lights lamp C
  7 lever D (R22) lights lamp D
These 8 "useful interactions" are counted per episode, separately from novelty.

Optional noisy TV (noisy_tv=True): a 3-cell screen on the floor of R10 whose cells show a fresh
random colour (1 of 8) every step. It is learnable-free novelty: nothing about it can be predicted.

Actions: 0 up, 1 down, 2 left, 3 right, 4 interact (with the cell the agent faces).
Facing = direction of the last move action. No extrinsic reward anywhere.
"""
import numpy as np
from numba import njit, prange

# ----------------------------------------------------------------- cell types
UNSEEN, FLOOR, WALL, DOOR_OPEN, DOOR_LOCKED, DOOR_SWITCH, KEY, BUTTON_UP, BUTTON_DOWN, \
    LEVER_OFF, LEVER_ON, LAMP_OFF, LAMP_ON, BOX, TV = range(15)
N_TYPES = 15
N_COLORS = 10          # 0 = no room colour (walls/doors), 1..9 = room id + 1
N_TV = 8               # TV pixel values
VIEW = 7
HALF = VIEW // 2
CELL_FEAT = N_TYPES + N_COLORS + N_TV
OBS_DIM = VIEW * VIEW * CELL_FEAT + 1 + 4   # + has_key + facing one-hot
N_ACTIONS = 5
N_INTER = 8
INTER_NAMES = ["lever A (lamp)", "button B (near door)", "push box", "pick up key",
               "unlock door", "button E (far door)", "lever C", "lever D"]
N_ROOMS = 9
ROOM_INT = 5
H = W = 3 * (ROOM_INT + 1) + 1  # 19
DIRS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int64)


def _room_origin(r, c):
    return 1 + r * (ROOM_INT + 1), 1 + c * (ROOM_INT + 1)


def build_layout(noisy_tv=False):
    """Returns (grid template int8 [H,W], room-id map int8 [H,W], start pos, mechanism coords)."""
    g = np.full((H, W), WALL, np.int8)
    room = np.full((H, W), -1, np.int8)
    for r in range(3):
        for c in range(3):
            y0, x0 = _room_origin(r, c)
            g[y0:y0 + ROOM_INT, x0:x0 + ROOM_INT] = FLOOR
            room[y0:y0 + ROOM_INT, x0:x0 + ROOM_INT] = r * 3 + c
    mid = ROOM_INT // 2

    def hdoor(r, c):  # door between (r,c) and (r,c+1)
        y0, x0 = _room_origin(r, c)
        return y0 + mid, x0 + ROOM_INT

    def vdoor(r, c):  # door between (r,c) and (r+1,c)
        y0, x0 = _room_origin(r, c)
        return y0 + ROOM_INT, x0 + mid

    m = {}
    g[hdoor(0, 0)] = DOOR_OPEN                      # R00-R01
    g[vdoor(0, 0)] = DOOR_OPEN                      # R00-R10
    m["door_B"] = hdoor(0, 1); g[m["door_B"]] = DOOR_SWITCH   # R01-R02
    m["box_door"] = vdoor(0, 1); g[m["box_door"]] = DOOR_OPEN  # R01-R11, box sits in it
    m["locked"] = vdoor(1, 0); g[m["locked"]] = DOOR_LOCKED    # R10-R20
    g[vdoor(0, 2)] = DOOR_OPEN                      # R02-R12
    m["door_E"] = vdoor(1, 2); g[m["door_E"]] = DOOR_SWITCH   # R12-R22
    g[hdoor(2, 0)] = DOOR_OPEN                      # R20-R21

    def at(r, c, dy, dx):
        y0, x0 = _room_origin(r, c)
        return y0 + dy, x0 + dx

    m["lever_A"] = at(0, 0, 0, 0); g[m["lever_A"]] = LEVER_OFF
    m["lamp_A"] = at(0, 0, 0, 4); g[m["lamp_A"]] = LAMP_OFF
    m["button_B"] = at(0, 1, 1, 4); g[m["button_B"]] = BUTTON_UP   # right next to door_B
    m["box"] = m["box_door"]; g[m["box"]] = BOX
    m["key"] = at(0, 2, 4, 4); g[m["key"]] = KEY
    m["button_E"] = at(2, 0, 4, 0); g[m["button_E"]] = BUTTON_UP
    m["lever_C"] = at(1, 1, 4, 2); g[m["lever_C"]] = LEVER_OFF
    m["lamp_C"] = at(1, 1, 0, 0); g[m["lamp_C"]] = LAMP_OFF
    m["lever_D"] = at(2, 2, 2, 4); g[m["lever_D"]] = LEVER_OFF
    m["lamp_D"] = at(2, 2, 0, 4); g[m["lamp_D"]] = LAMP_OFF
    tv = np.zeros((0, 2), np.int64)
    if noisy_tv:
        tv = np.array([at(1, 0, 2, 4), at(1, 0, 3, 4), at(1, 0, 4, 4)], np.int64)  # east wall of R10
        for y, x in tv:
            g[y, x] = TV
    start = np.array(at(0, 0, 2, 2), np.int64)
    # wall/door cells carry no room colour; objects on floor keep their room colour
    return g, room, start, m, tv


# ----------------------------------------------------------------- numba core
@njit(cache=True)
def _write_obs(obs_row, grid, room, tv_val, y, x, facing, has_key):
    obs_row[:] = 0.0
    k = 0
    for dy in range(-HALF, HALF + 1):
        for dx in range(-HALF, HALF + 1):
            yy = y + dy
            xx = x + dx
            base = k * CELL_FEAT
            if yy < 0 or yy >= H or xx < 0 or xx >= W:
                obs_row[base + UNSEEN] = 1.0
                obs_row[base + N_TYPES] = 1.0
            else:
                t = grid[yy, xx]
                obs_row[base + t] = 1.0
                obs_row[base + N_TYPES + room[yy, xx] + 1] = 1.0
                if t == TV:
                    obs_row[base + N_TYPES + N_COLORS + tv_val[yy, xx]] = 1.0
            k += 1
    obs_row[VIEW * VIEW * CELL_FEAT] = has_key
    obs_row[VIEW * VIEW * CELL_FEAT + 1 + facing] = 1.0


@njit(cache=True)
def _reset_one(i, grid, template, pos, facing, has_key, t, visited, rooms, inter, inter_t,
               unprod, tvsteps, repeat, start, room):
    grid[i] = template
    pos[i, 0] = start[0]; pos[i, 1] = start[1]
    facing[i] = 1
    has_key[i] = 0
    t[i] = 0
    visited[i] = False
    rooms[i] = False
    inter[i] = False
    inter_t[i] = -1
    unprod[i] = 0
    tvsteps[i] = 0
    repeat[i] = 0
    visited[i, start[0], start[1]] = True
    rooms[i, room[start[0], start[1]]] = True


@njit(parallel=True, cache=True)
def _step(actions, grid, template, room, tv_val, tv_cells, pos, facing, has_key, t, visited,
          rooms, inter, inter_t, unprod, tvsteps, repeat, start, mech, max_steps, obs, done,
          ep_stats, ep_inter_t, ep_visited, rng_tv):
    n = actions.shape[0]
    for i in prange(n):
        a = actions[i]
        y = pos[i, 0]; x = pos[i, 1]
        new_event = False
        if a < 4:
            facing[i] = a
            dy = (-1, 1, 0, 0)[a]; dx = (0, 0, -1, 1)[a]
            ny = y + dy; nx = x + dx
            c = grid[i, ny, nx]
            if c == FLOOR or c == DOOR_OPEN:
                y = ny; x = nx
            elif c == KEY:
                grid[i, ny, nx] = FLOOR
                has_key[i] = 1
                y = ny; x = nx
                if not inter[i, 3]:
                    inter[i, 3] = True; inter_t[i, 3] = t[i]; new_event = True
            elif c == DOOR_LOCKED and has_key[i] == 1:
                grid[i, ny, nx] = DOOR_OPEN
                has_key[i] = 0
                if not inter[i, 4]:
                    inter[i, 4] = True; inter_t[i, 4] = t[i]; new_event = True
            elif c == BOX:
                by = ny + dy; bx = nx + dx
                bc = grid[i, by, bx]
                if bc == FLOOR or bc == DOOR_OPEN:
                    # what lies under the box: the doorway if it started there, else floor
                    if ny == mech[2, 0] and nx == mech[2, 1]:
                        grid[i, ny, nx] = DOOR_OPEN
                    else:
                        grid[i, ny, nx] = template[ny, nx] if template[ny, nx] != BOX else FLOOR
                    grid[i, by, bx] = BOX
                    y = ny; x = nx
                    if not inter[i, 2]:
                        inter[i, 2] = True; inter_t[i, 2] = t[i]; new_event = True
        else:
            f = facing[i]
            dy = (-1, 1, 0, 0)[f]; dx = (0, 0, -1, 1)[f]
            fy = y + dy; fx = x + dx
            c = grid[i, fy, fx]
            ev = -1
            if c == LEVER_OFF or c == LEVER_ON:
                on = c == LEVER_OFF
                grid[i, fy, fx] = LEVER_ON if on else LEVER_OFF
                # which lever? mech rows: 0 lever_A/lamp_A(row 8), 6 lever_C, 7 lever_D
                if fy == mech[0, 0] and fx == mech[0, 1]:
                    ev = 0; ly = mech[8, 0]; lx = mech[8, 1]
                elif fy == mech[6, 0] and fx == mech[6, 1]:
                    ev = 6; ly = mech[9, 0]; lx = mech[9, 1]
                else:
                    ev = 7; ly = mech[10, 0]; lx = mech[10, 1]
                grid[i, ly, lx] = LAMP_ON if on else LAMP_OFF
            elif c == BUTTON_UP:
                grid[i, fy, fx] = BUTTON_DOWN
                if fy == mech[1, 0] and fx == mech[1, 1]:
                    ev = 1; grid[i, mech[11, 0], mech[11, 1]] = DOOR_OPEN
                else:
                    ev = 5; grid[i, mech[12, 0], mech[12, 1]] = DOOR_OPEN
            elif c == BUTTON_DOWN:
                repeat[i] += 1
            if ev >= 0:
                if not inter[i, ev]:
                    inter[i, ev] = True; inter_t[i, ev] = t[i]; new_event = True
                else:
                    repeat[i] += 1
        pos[i, 0] = y; pos[i, 1] = x
        if not visited[i, y, x]:
            visited[i, y, x] = True
            new_event = True
        if room[y, x] >= 0:
            rooms[i, room[y, x]] = True
        if not new_event:
            unprod[i] += 1
        # noisy TV: fresh random pixels every step; count steps with the screen in view
        if tv_cells.shape[0] > 0:
            seen = False
            for j in range(tv_cells.shape[0]):
                ty = tv_cells[j, 0]; tx = tv_cells[j, 1]
                tv_val[i, ty, tx] = rng_tv[i, j]
                if abs(ty - y) <= HALF and abs(tx - x) <= HALF:
                    seen = True
            if seen:
                tvsteps[i] += 1
        t[i] += 1
        done[i] = t[i] >= max_steps
        if done[i]:
            nr = 0
            for r in range(N_ROOMS):
                nr += rooms[i, r]
            ni = 0
            for e in range(N_INTER):
                ni += inter[i, e]
            nc = 0
            for yy in range(H):
                for xx in range(W):
                    nc += visited[i, yy, xx]
                    ep_visited[i, yy, xx] = visited[i, yy, xx]
            ep_stats[i, 0] = nr
            ep_stats[i, 1] = ni
            ep_stats[i, 2] = nc
            ep_stats[i, 3] = unprod[i]
            ep_stats[i, 4] = tvsteps[i]
            ep_stats[i, 5] = repeat[i]
            ep_inter_t[i] = inter_t[i]
            _reset_one(i, grid, template, pos, facing, has_key, t, visited, rooms, inter, inter_t,
                       unprod, tvsteps, repeat, start, room)
        _write_obs(obs[i], grid[i], room, tv_val[i], pos[i, 0], pos[i, 1], facing[i], has_key[i])


@njit(cache=True)
def _obs_all(obs, grid, room, tv_val, pos, facing, has_key):
    for i in range(obs.shape[0]):
        _write_obs(obs[i], grid[i], room, tv_val[i], pos[i, 0], pos[i, 1], facing[i], has_key[i])


# ----------------------------------------------------------------- python wrapper
EP_FIELDS = ["rooms", "interactions", "coverage", "unproductive", "tv_steps", "repeat_interactions"]


class VecRooms:
    """N independent copies of the world, stepped together. Fixed-length episodes."""

    def __init__(self, n, max_steps=512, noisy_tv=False, seed=0):
        self.n, self.max_steps, self.noisy_tv = n, max_steps, noisy_tv
        g, room, start, m, tv = build_layout(noisy_tv)
        self.template, self.room, self.start, self.mech_dict, self.tv_cells = g, room, start, m, tv
        order = ["lever_A", "button_B", "box_door", "key", "locked", "button_E", "lever_C",
                 "lever_D", "lamp_A", "lamp_C", "lamp_D", "door_B", "door_E"]
        self.mech = np.array([m[k] for k in order], np.int64)
        self.rng = np.random.default_rng(seed)
        self.grid = np.repeat(g[None], n, 0)
        self.tv_val = np.zeros((n, H, W), np.int64)
        self.pos = np.zeros((n, 2), np.int64)
        self.facing = np.zeros(n, np.int64)
        self.has_key = np.zeros(n, np.int64)
        self.t = np.zeros(n, np.int64)
        self.visited = np.zeros((n, H, W), np.bool_)
        self.rooms = np.zeros((n, N_ROOMS), np.bool_)
        self.inter = np.zeros((n, N_INTER), np.bool_)
        self.inter_t = np.full((n, N_INTER), -1, np.int64)
        self.unprod = np.zeros(n, np.int64)
        self.tvsteps = np.zeros(n, np.int64)
        self.repeat = np.zeros(n, np.int64)
        self.obs = np.zeros((n, OBS_DIM), np.float32)
        self.done = np.zeros(n, np.bool_)
        self.ep_stats = np.zeros((n, len(EP_FIELDS)), np.int64)
        self.ep_inter_t = np.full((n, N_INTER), -1, np.int64)
        self.ep_visited = np.zeros((n, H, W), np.bool_)
        for i in range(n):
            _reset_one(i, self.grid, self.template, self.pos, self.facing, self.has_key, self.t,
                       self.visited, self.rooms, self.inter, self.inter_t, self.unprod,
                       self.tvsteps, self.repeat, self.start, self.room)
        _obs_all(self.obs, self.grid, self.room, self.tv_val, self.pos, self.facing, self.has_key)

    def reset(self):
        return self.obs.copy()

    def step(self, actions):
        """Returns obs [n,OBS_DIM], done [n] bool, and list of finished-episode stat dicts."""
        rng_tv = self.rng.integers(0, N_TV, size=(self.n, max(len(self.tv_cells), 1)))
        _step(np.asarray(actions, np.int64), self.grid, self.template, self.room, self.tv_val,
              self.tv_cells, self.pos, self.facing, self.has_key, self.t, self.visited, self.rooms,
              self.inter, self.inter_t, self.unprod, self.tvsteps, self.repeat, self.start,
              self.mech, self.max_steps, self.obs, self.done, self.ep_stats, self.ep_inter_t,
              self.ep_visited, rng_tv)
        eps = []
        if self.done.any():
            for i in np.nonzero(self.done)[0]:
                d = dict(zip(EP_FIELDS, self.ep_stats[i].tolist()))
                d["inter_t"] = self.ep_inter_t[i].tolist()
                eps.append(d)
        return self.obs.copy(), self.done.copy(), eps

    # state snapshot for checkpoints (so a resumed run continues the same episodes)
    _STATE = ["grid", "tv_val", "pos", "facing", "has_key", "t", "visited", "rooms", "inter",
              "inter_t", "unprod", "tvsteps", "repeat", "obs"]

    def get_state(self):
        s = {k: getattr(self, k).copy() for k in self._STATE}
        s["rng"] = self.rng.bit_generator.state
        return s

    def set_state(self, s):
        for k in self._STATE:
            getattr(self, k)[...] = s[k]
        self.rng.bit_generator.state = s["rng"]

    def n_floor_cells(self):
        return int(np.sum(self.template != WALL))


def render_rgb(grid, room, pos, tv_val=None, cell=12):
    """Full-map RGB image (for demos only; the agent never sees this)."""
    room_cols = np.array([[236, 230, 214], [214, 232, 222], [222, 222, 240], [240, 222, 222],
                          [226, 236, 206], [214, 228, 238], [238, 228, 206], [228, 214, 236],
                          [206, 232, 232]], np.uint8)
    col = {WALL: (60, 60, 70), DOOR_OPEN: (190, 170, 130), DOOR_LOCKED: (200, 60, 60),
           DOOR_SWITCH: (120, 80, 160), KEY: (240, 200, 0), BUTTON_UP: (230, 120, 30),
           BUTTON_DOWN: (120, 70, 20), LEVER_OFF: (70, 110, 200), LEVER_ON: (40, 200, 240),
           LAMP_OFF: (90, 90, 60), LAMP_ON: (255, 250, 120), BOX: (150, 100, 50)}
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):
        for x in range(W):
            t = grid[y, x]
            if t == FLOOR:
                img[y, x] = room_cols[room[y, x]]
            elif t == TV:
                v = 0 if tv_val is None else tv_val[y, x]
                rng = np.random.default_rng(v + 7)
                img[y, x] = rng.integers(0, 255, 3)
            else:
                img[y, x] = col.get(int(t), (255, 0, 255))
    img = np.kron(img, np.ones((cell, cell, 1), np.uint8))
    if pos is not None:
        y, x = pos
        c0 = cell // 4
        img[y * cell + c0:(y + 1) * cell - c0, x * cell + c0:(x + 1) * cell - c0] = (220, 30, 120)
    return img
