"""Room configs: defaults, the family rules for each split factor, the generators and the validator.

The named factors (brief M1) each get three pairwise-disjoint families: train / dev / test.
  instances     same layout and appearance rules as training; unseen seeds, start poses, lamp states
  arrangements  reserved placements of the home mat, button, occluder, partition and pads
  appearances   reserved shade bands, floor textures, wall colours and lighting
  distractor    (lessons file, item 4) a flickering screen on a wall; dev and test only, never trained on
Layout and appearance come from separate RNG streams, and appearance fields are visual only
(colours, textures, lights), so appearance can never shift physics (tests/test_m1.py checks this).
Pad colour *categories* (red/green/blue/yellow) are fixed; appearance only moves their shades.
Which category sits on which pad is layout.
"""
import colorsys
import copy

import numpy as np

from room import MUJOCO_VERSION, PAD_COLOURS, config_hash

ROOM = {"size": [3.0, 3.0], "wall_height": 0.4, "wall_thickness": 0.05}
# 5 mm plate, 5 mm travel; pressed at 2.5 mm (3.75 N, about half the duck's 7.2 N weight); chosen in m1_physics_checks.py.
BUTTON = {"size": 0.24, "thickness": 0.005, "height": 0.005, "travel": 0.005, "stiffness": 1500.0,
          "preload": 0.0, "damping": 2.0, "mass": 0.03, "press_depth": 0.0025}
HOME_SIZE, PAD_SIZE = 0.40, 0.30
OCCLUDER = {"length": 0.40, "thickness": 0.04, "height": 0.20}
PARTITION = {"thickness": 0.04, "height": 0.30}
INTERIOR = 0.9          # |x|, |y| <= 0.9 m: at least 0.6 m from every wall
DUCK_CLEARANCE = 0.15   # metres kept free around the duck's path in the reachability check
OBSTACLE_GAP = 2 * DUCK_CLEARANCE   # occluder <-> partition: no overlap and no slit the duck can't pass
WALL_ITEM_GAP = 0.05    # lamp / distractor screen kept this far (edge to face) from the partition's wall joint

# Hue centres (degrees) of the colour categories the instructions name. The closest pair of centres
# is button/blue (45 deg); with independent jitter the closest drawn shades are 33 deg apart within
# the training family and 25 deg across families (REVIEW_M1.md: in CIELAB every reserved shade is still
# nearest its own training category).
PAD_HUE = {"red": 0.0, "yellow": 55.0, "green": 120.0, "blue": 220.0}
BALL_HUE, BUTTON_HUE = 300.0, 175.0

# Room cells (1 m squares): 6 7 8 / 3 4 5 / 0 1 2, i.e. id = 3*row + col from the SW corner.
CORNERS, EDGES_NS, EDGES_EW, CENTRE = (0, 2, 6, 8), (1, 7), (3, 5), (4,)

# ------------------------------------------------------------------------------- family rules
ARRANGEMENT_FAMILIES = {
    "train": {"home": CORNERS, "button": CORNERS, "occluder": EDGES_NS + EDGES_EW,
              "partition": "wall_NS", "pads_interior": (0, 0)},
    "dev":   {"home": EDGES_NS, "button": EDGES_EW, "occluder": CORNERS,
              "partition": "wall_EW", "pads_interior": (1, 1)},
    "test":  {"home": EDGES_EW, "button": EDGES_NS, "occluder": CENTRE,
              "partition": "free", "pads_interior": (2, 4)},
}
APPEARANCE_FAMILIES = {
    # hue_offset: jitter of every category hue (pads, button, ball); s, v: their saturation/value
    "train": {"hue_offset": (-6, 6), "s": (0.75, 0.95), "v": (0.75, 0.95),
              "floor_builtin": "checker", "floor_mark": "edge", "floor_hue": (190, 250), "floor_s": (0.05, 0.15),
              "floor_v": (0.22, 0.38), "texrepeat": (5, 8),
              "wall_hue": (30, 60), "wall_s": (0.0, 0.10), "wall_v": (0.70, 0.85),
              "home_s": (0.0, 0.05), "home_v": (0.85, 0.95),
              "light_tilt_deg": (0, 15), "diffuse": (0.55, 0.70), "ambient": (0.20, 0.30)},
    "dev":   {"hue_offset": (8, 14), "s": (0.55, 0.72), "v": (0.75, 0.95),
              "floor_builtin": "flat", "floor_mark": "random", "floor_hue": (20, 50), "floor_s": (0.10, 0.20),
              "floor_v": (0.40, 0.52), "texrepeat": (5, 8),
              "wall_hue": (200, 240), "wall_s": (0.05, 0.15), "wall_v": (0.55, 0.69),
              "home_s": (0.0, 0.05), "home_v": (0.70, 0.80),
              "light_tilt_deg": (30, 45), "diffuse": (0.45, 0.54), "ambient": (0.31, 0.40)},
    "test":  {"hue_offset": (-14, -8), "s": (0.75, 0.95), "v": (0.50, 0.68),
              "floor_builtin": "checker", "floor_mark": "none", "floor_hue": (90, 150), "floor_s": (0.15, 0.25),
              "floor_v": (0.53, 0.65), "texrepeat": (2, 3),
              "wall_hue": (300, 360), "wall_s": (0.0, 0.08), "wall_v": (0.35, 0.50),
              "home_s": (0.15, 0.25), "home_v": (0.82, 0.92),
              "light_tilt_deg": (16, 29), "diffuse": (0.75, 0.85), "ambient": (0.10, 0.15)},
}


def hsv(h_deg, s, v):
    return [round(float(c), 4) for c in colorsys.hsv_to_rgb((h_deg % 360) / 360.0, s, v)]


def rgb_to_hsv_deg(rgb):
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return h * 360.0, s, v


# ----------------------------------------------------------------------------------- appearance
def sample_appearance(rng, family):
    """Visual-only parameters for one config, from APPEARANCE_FAMILIES[family]."""
    f = APPEARANCE_FAMILIES[family]
    u = lambda lo_hi: float(rng.uniform(*lo_hi))  # noqa: E731

    def shade(centre):
        return hsv(centre + u(f["hue_offset"]), u(f["s"]), u(f["v"]))

    fh, fs, fv = u(f["floor_hue"]), u(f["floor_s"]), u(f["floor_v"])
    wall = hsv(u(f["wall_hue"]), u(f["wall_s"]), u(f["wall_v"]))
    tilt, az = np.radians(u(f["light_tilt_deg"])), float(rng.uniform(0, 2 * np.pi))
    d, a = u(f["diffuse"]), u(f["ambient"])
    return {
        "family": family,
        "floor": {"builtin": f["floor_builtin"], "mark": f["floor_mark"], "rgb1": hsv(fh, fs, fv),
                  "rgb2": hsv(fh, fs, fv * 0.8), "markrgb": hsv(fh, fs * 0.5, min(1.0, fv * 1.3)),
                  "random": 0.03 if f["floor_mark"] == "random" else 0.0,
                  "texrepeat": int(rng.integers(f["texrepeat"][0], f["texrepeat"][1] + 1)), "reflectance": 0.05},
        "sky": {"rgb1": hsv(fh, 0.1, 0.7), "rgb2": [0.2, 0.2, 0.25]},
        "wall_rgb": wall, "partition_rgb": [round(c * 0.85, 4) for c in wall],
        "occluder_rgb": [round(c * 0.7, 4) for c in wall],
        "home_rgb": hsv(40.0, u(f["home_s"]), u(f["home_v"])),
        "pad_rgb": {c: shade(PAD_HUE[c]) for c in PAD_COLOURS},
        "button_rgb": shade(BUTTON_HUE), "ball_rgb": shade(BALL_HUE),
        "lamp_off_rgb": [0.15, 0.15, 0.15], "lamp_on_rgb": [1.0, 0.95, 0.6], "lamp_on_emission": 0.9,
        "light": {"dir": [round(float(np.sin(tilt) * np.cos(az)), 4), round(float(np.sin(tilt) * np.sin(az)), 4),
                          round(float(-np.cos(tilt)), 4)],
                  "diffuse": [round(d, 4)] * 3, "ambient": [round(a, 4)] * 3, "castshadow": True,
                  "headlight_ambient": [0.2, 0.2, 0.2], "headlight_diffuse": [0.3, 0.3, 0.3]},
    }


def default_appearance():
    """A plain look for hand-made test configs."""
    return sample_appearance(np.random.default_rng(0), "train")


# -------------------------------------------------------------------------------------- layout
def cell_of(xy):
    col = int(np.clip(np.floor(xy[0] + 1.5), 0, 2))
    row = int(np.clip(np.floor(xy[1] + 1.5), 0, 2))
    return 3 * row + col


def _in_cell(rng, cell, limit, margin=0.1):
    """Uniform point in a cell, `margin` inside its borders and within |x|, |y| <= limit."""
    row, col = divmod(cell, 3)
    lo = np.maximum([col - 1.5 + margin, row - 1.5 + margin], -limit)
    hi = np.minimum([col - 0.5 - margin, row - 0.5 - margin], limit)
    assert (hi > lo).all(), (cell, limit)
    return rng.uniform(lo, hi).round(3)


def marker_limit(half):
    """Largest |x|, |y| for a flat marker of half-size `half` (0.20 m from the walls, plus 2 cm)."""
    return ROOM["size"][0] / 2 - 0.20 - half - 0.02


OCCLUDER_LIMIT = 0.95   # occluder centre >= 0.55 m from walls, so both sides stay reachable


def is_interior(xy):
    return abs(xy[0]) <= INTERIOR and abs(xy[1]) <= INTERIOR


def _partition(rng, kind):
    hx = ROOM["size"][0] / 2
    if kind == "wall_NS":   # perpendicular to the N or S wall, touching it
        L = float(rng.uniform(1.0, 1.3))
        side = rng.choice([-1, 1])
        return {"pos": [round(float(rng.uniform(-0.6, 0.6)), 3), round(float(side * (hx - L / 2)), 3)],
                "yaw": round(np.pi / 2, 6), "length": round(L, 3), "attached": "N" if side > 0 else "S",
                **PARTITION}
    if kind == "wall_EW":
        L = float(rng.uniform(1.0, 1.3))
        side = rng.choice([-1, 1])
        return {"pos": [round(float(side * (hx - L / 2)), 3), round(float(rng.uniform(-0.6, 0.6)), 3)],
                "yaw": 0.0, "length": round(L, 3), "attached": "E" if side > 0 else "W", **PARTITION}
    L = float(rng.uniform(0.8, 1.1))   # free-standing, >= 0.45 m from every wall at both ends
    yaw = float(rng.choice([0.0, np.pi / 2]))
    lim = hx - 0.45 - L / 2
    along = float(rng.uniform(-lim, lim))
    across = float(rng.uniform(-0.5, 0.5))
    pos = [along, across] if yaw == 0.0 else [across, along]
    return {"pos": [round(p, 3) for p in pos], "yaw": round(yaw, 6), "length": round(L, 3),
            "attached": None, **PARTITION}


def _point_free(X, Y, obst, xy):
    r, c = int((xy[1] - Y[0, 0]) / GRID + 0.5), int((xy[0] - X[0, 0]) / GRID + 0.5)
    return 0 <= r < obst.shape[0] and 0 <= c < obst.shape[1] and not obst[r, c]


def _marker_fits(X, Y, obst20, xy, half, placed):
    """Marker clear of obstacles by 0.20 m and of already placed markers by a 0.15 m gap."""
    if (_rect_mask(X, Y, xy, half, half, 0.0) & obst20).any():
        return False
    return all(max(abs(xy[0] - p[0]), abs(xy[1] - p[1])) - half - h >= 0.15 for p, h in placed)


def sample_layout(rng, family, tries=200):
    """Physical layout from ARRANGEMENT_FAMILIES[family], placed object by object (partition,
    occluder + ball, home, button, pads), each re-drawn until it fits. validate() still has the last
    word (reachability). Returns None if some object can't be placed (caller redraws)."""
    f = ARRANGEMENT_FAMILIES[family]
    lay = {"family": family, "room": copy.deepcopy(ROOM), "partition": _partition(rng, f["partition"])}
    home_cell = int(rng.choice(f["home"]))
    button_cell = int(rng.choice([c for c in f["button"] if c != home_cell]))
    occ_cell = int(rng.choice([c for c in f["occluder"] if c not in (home_cell, button_cell)]))
    # occluder: both 0.30 m side points must be free of walls/partition with the duck's clearance
    lay["occluder"] = None
    for _ in range(tries):
        occ = _in_cell(rng, occ_cell, OCCLUDER_LIMIT)
        yaw = float(rng.uniform(0, np.pi))
        cand = {**OCCLUDER, "pos": occ.tolist(), "yaw": round(yaw, 4), "cell": occ_cell}
        lay["occluder"] = {**cand, "pos": [9.0, 9.0]}  # keep it out of its own clearance test
        X, Y, obst = footprints(lay, pad=DUCK_CLEARANCE)
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        if all(_point_free(X, Y, obst, occ + sg * 0.30 * normal) for sg in (-1, 1)) \
                and _occluder_partition_gap_ok({**lay, "occluder": cand}):
            lay["occluder"] = cand
            break
    else:
        return None
    occ, yaw = np.array(lay["occluder"]["pos"]), lay["occluder"]["yaw"]
    normal, along = np.array([-np.sin(yaw), np.cos(yaw)]), np.array([np.cos(yaw), np.sin(yaw)])
    side = rng.choice([-1, 1])
    lay["ball"] = {"pos": (occ + side * 0.30 * normal + rng.uniform(-0.1, 0.1) * along).round(3).tolist()}
    X, Y, obst20 = footprints(lay, pad=0.20)
    placed = [(lay["ball"]["pos"], 0.10)]   # keep markers off the ball
    for key, cell, half in (("home", home_cell, HOME_SIZE / 2), ("button", button_cell, BUTTON["size"] / 2)):
        for _ in range(tries):
            xy = _in_cell(rng, cell, marker_limit(half))
            if _marker_fits(X, Y, obst20, xy, half, placed):
                break
        else:
            return None
        placed.append((xy.tolist(), half))
        lay[key] = {"pos": xy.tolist(), "cell": cell,
                    **({"size": HOME_SIZE} if key == "home" else BUTTON)}
    n_pads = int(rng.choice([3, 4]))
    colours = list(rng.permutation(PAD_COLOURS)[:n_pads])
    lo, hi = f["pads_interior"]
    n_int = int(rng.integers(lo, min(hi, n_pads) + 1))
    lay["pads"] = []
    lim = marker_limit(PAD_SIZE / 2)
    for i, c in enumerate(colours):
        for _ in range(tries * 5):
            xy = rng.uniform(-lim, lim, 2).round(3)
            if is_interior(xy) == (i < n_int) and _marker_fits(X, Y, obst20, xy, PAD_SIZE / 2, placed):
                break
        else:
            return None
        placed.append((xy.tolist(), PAD_SIZE / 2))
        lay["pads"].append({"color": str(c), "pos": xy.tolist(), "size": PAD_SIZE})
    for _ in range(tries):
        lay["lamp"] = {"wall": str(rng.choice(["N", "S", "E", "W"])),
                       "offset": round(float(rng.uniform(-1.0, 1.0)), 3), "height": 0.30}
        if _clear_of_partition_joint(lay, lay["lamp"]["wall"], lay["lamp"]["offset"], LAMP_HALF_WIDTH):
            break
    else:
        return None
    return lay


LAMP_HALF_WIDTH = 0.05   # room.py's lamp box is 0.10 m wide along the wall


def _occluder_partition_gap_ok(lay, gap=OBSTACLE_GAP):
    """Occluder footprint grown by `gap` (as a box) must not touch the partition: no interpenetration
    and no slit narrower than the duck's clearance between them (review M1)."""
    p, o = lay["partition"], lay["occluder"]
    if p is None:
        return True
    X, Y, _ = footprints({**lay, "partition": None, "occluder": {**o, "pos": [9.0, 9.0]}})
    mo = _rect_mask(X, Y, o["pos"], o["length"] / 2, o["thickness"] / 2, o["yaw"], gap)
    mp = _rect_mask(X, Y, p["pos"], p["length"] / 2, p["thickness"] / 2, p["yaw"])
    return not (mo & mp).any()


def _clear_of_partition_joint(lay, wall, offset, half_width):
    """A wall-mounted item (lamp, distractor screen) must not sit in the partition's joint with that wall."""
    p = lay["partition"]
    if p is None or p["attached"] != wall:
        return True
    joint = p["pos"][0] if wall in ("N", "S") else p["pos"][1]
    return abs(offset - joint) >= half_width + p["thickness"] / 2 + WALL_ITEM_GAP


def sample_instance(rng, layout):
    """Unseen-instance part: start pose on the home mat and the initial lamp state."""
    hp = np.array(layout["home"]["pos"])
    start = (hp + rng.uniform(-0.08, 0.08, 2)).round(3)
    return {"start": {"pos": start.tolist(), "yaw": round(float(rng.uniform(-np.pi, np.pi)), 4)},
            "lamp_initial_on": bool(rng.random() < 0.5)}


def sample_distractor(rng, layout, tries=200):
    walls = [w for w in ("N", "S", "E", "W") if w != layout["lamp"]["wall"]]
    for _ in range(tries):
        d = {"wall": str(rng.choice(walls)), "offset": round(float(rng.uniform(-0.9, 0.9)), 3),
             "height": 0.22, "width": 0.30, "tall": 0.20}
        if _clear_of_partition_joint(layout, d["wall"], d["offset"], d["width"] / 2):
            return d
    raise RuntimeError("no place for the distractor screen")


# ------------------------------------------------------------------------------- validation
GRID = 0.02


def _rect_mask(X, Y, centre, half_len, half_wid, yaw, pad=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = X - centre[0], Y - centre[1]
    u, v = c * dx + s * dy, -s * dx + c * dy
    return (np.abs(u) <= half_len + pad) & (np.abs(v) <= half_wid + pad)


def footprints(layout, pad=0.0):
    """Boolean masks on a 2 cm grid: obstacles (walls, partition, occluder) and walkable markers."""
    hx, hy = layout["room"]["size"][0] / 2, layout["room"]["size"][1] / 2
    xs = np.arange(-hx + GRID / 2, hx, GRID)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    obst = (np.abs(X) >= hx - pad) | (np.abs(Y) >= hy - pad)
    if layout["partition"] is not None:
        p = layout["partition"]
        obst |= _rect_mask(X, Y, p["pos"], p["length"] / 2, p["thickness"] / 2, p["yaw"], pad)
    o = layout["occluder"]
    obst |= _rect_mask(X, Y, o["pos"], o["length"] / 2, o["thickness"] / 2, o["yaw"], pad)
    return X, Y, obst


def _reachable(X, Y, free, start_xy):
    from collections import deque
    n = X.shape[0]
    idx = lambda xy: (int((xy[1] - Y[0, 0]) / GRID + 0.5), int((xy[0] - X[0, 0]) / GRID + 0.5))  # noqa: E731
    seen = np.zeros_like(free)
    s = idx(start_xy)
    if not free[s]:
        return seen, idx
    q = deque([s])
    seen[s] = True
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < n and free[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen, idx


def validate(cfg):
    """Return a list of problems (empty = valid): marker overlaps, occluder/partition overlap or slit,
    lamp/screen in the partition's wall joint, clearances, reachability, family rules."""
    lay = cfg["layout"]
    problems = []
    markers = [("home", lay["home"]["pos"], lay["home"]["size"] / 2),
               ("button", lay["button"]["pos"], lay["button"]["size"] / 2)]
    markers += [(f"pad_{p['color']}", p["pos"], p["size"] / 2) for p in lay["pads"]]
    # markers: no overlap with each other (0.15 m gap), 0.20 m from obstacles, inside the room
    for i, (n1, p1, h1) in enumerate(markers):
        for n2, p2, h2 in markers[i + 1:]:
            gap = max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1])) - h1 - h2
            if gap < 0.15:
                problems.append(f"{n1}/{n2} gap {gap:.2f}")
    X, Y, obst_wide = footprints(lay, pad=0.20)
    for n, p, h in markers:
        m = _rect_mask(X, Y, p, h, h, 0.0)
        if (m & obst_wide).any():
            problems.append(f"{n} within 0.20 m of a wall/partition/occluder")
    ball = np.array(lay["ball"]["pos"])
    for n, p, h in markers:
        if max(abs(ball[0] - p[0]), abs(ball[1] - p[1])) < h + 0.08:
            problems.append(f"ball on {n}")
    _, _, obst_ball = footprints(lay, pad=0.04)
    bi = (int((ball[1] - Y[0, 0]) / GRID + 0.5), int((ball[0] - X[0, 0]) / GRID + 0.5))
    if obst_ball[bi]:
        problems.append("ball inside an obstacle")
    # reachability with the duck's clearance, from the start pose
    _, _, obst = footprints(lay, pad=DUCK_CLEARANCE)
    seen, idx = _reachable(X, Y, ~obst, lay["start"]["pos"])
    for n, p, _ in markers:
        if not seen[idx(p)]:
            problems.append(f"{n} unreachable")
    o = lay["occluder"]
    if not _occluder_partition_gap_ok(lay):
        problems.append(f"occluder within {OBSTACLE_GAP:.2f} m of the partition (or overlapping it)")
    if not _clear_of_partition_joint(lay, lay["lamp"]["wall"], lay["lamp"]["offset"], LAMP_HALF_WIDTH):
        problems.append("lamp in the partition's wall joint")
    dist = cfg.get("distractor")
    if dist is not None and (dist["wall"] == lay["lamp"]["wall"]
                             or not _clear_of_partition_joint(lay, dist["wall"], dist["offset"], dist["width"] / 2)):
        problems.append("distractor on the lamp's wall or in the partition's wall joint")
    normal = np.array([-np.sin(o["yaw"]), np.cos(o["yaw"])])
    for sgn in (-1, 1):  # both sides of the occluder must be reachable (to look behind it)
        pt = np.array(o["pos"]) + sgn * 0.30 * normal
        r, c = idx(pt)
        if not (0 <= r < seen.shape[0] and 0 <= c < seen.shape[1]) or not seen[r, c]:
            problems.append(f"occluder side {sgn:+d} unreachable")
    problems += check_family_rules(cfg)
    return problems


def check_family_rules(cfg):
    """Does the config obey the family rules it claims? (Used by validate() and check_splits.py.)"""
    lay, app = cfg["layout"], cfg["appearance"]
    f = ARRANGEMENT_FAMILIES[lay["family"]]
    out = []
    for key in ("home", "button", "occluder"):
        if cell_of(lay[key]["pos"]) not in f[key]:
            out.append(f"{key} cell {cell_of(lay[key]['pos'])} not in {lay['family']} family")
    p = lay["partition"]
    kind = "free" if p["attached"] is None else ("wall_NS" if p["attached"] in "NS" else "wall_EW")
    if kind != f["partition"]:
        out.append(f"partition {kind} not {f['partition']}")
    n_int = sum(is_interior(pd["pos"]) for pd in lay["pads"])
    lo, hi = f["pads_interior"]
    if not lo <= n_int <= hi:
        out.append(f"{n_int} interior pads not in {lo}..{hi}")
    a = APPEARANCE_FAMILIES[app["family"]]
    for cat, rgb in [*app["pad_rgb"].items(), ("button", app["button_rgb"]), ("ball", app["ball_rgb"])]:
        centre = {"button": BUTTON_HUE, "ball": BALL_HUE}.get(cat, PAD_HUE.get(cat))
        h, s, v = rgb_to_hsv_deg(rgb)
        off = (h - centre + 180) % 360 - 180
        if not (a["hue_offset"][0] - 0.6 <= off <= a["hue_offset"][1] + 0.6
                and a["s"][0] - 0.01 <= s <= a["s"][1] + 0.01 and a["v"][0] - 0.01 <= v <= a["v"][1] + 0.01):
            out.append(f"{cat} shade outside {app['family']} band")
    if app["floor"]["builtin"] != a["floor_builtin"] or app["floor"]["mark"] != a["floor_mark"]:
        out.append("floor texture not in family")
    return out


# ------------------------------------------------------------------------------- assembly
def make_config(cid, layout_family, appearance_family, seeds, distractor=False, max_tries=500):
    """Draw a valid config. `seeds` = dict of independent integer seeds: layout, appearance, instance.
    Invalid layout draws are re-drawn from the same layout stream (the appearance stream is untouched)."""
    lrng = np.random.default_rng([seeds["layout"], 1])
    arng = np.random.default_rng([seeds["appearance"], 2])
    irng = np.random.default_rng([seeds["instance"], 3])
    drng = np.random.default_rng([seeds["instance"], 4])
    appearance = sample_appearance(arng, appearance_family)
    for attempt in range(max_tries):
        layout = sample_layout(lrng, layout_family)
        if layout is None:
            continue
        layout.update(sample_instance(irng, layout))
        cfg = {"id": cid, "mujoco_version": MUJOCO_VERSION, "seed": int(seeds["instance"]),
               "seeds": dict(seeds), "layout": layout, "appearance": appearance,
               "distractor": sample_distractor(drng, layout) if distractor else None}
        if not validate(cfg):
            cfg["layout_attempts"] = attempt + 1
            cfg["hash"] = config_hash(cfg)
            return cfg
    raise RuntimeError(f"no valid layout for {cid} in {max_tries} draws")


def hand_config(button_height=None, start=((0.0, 0.0), 0.0), button_pos=(0.6, 0.0), empty=False):
    """A fixed config for tests and physics checks (not part of any split)."""
    b = dict(BUTTON)
    if button_height is not None:
        b.update(height=button_height, travel=button_height, thickness=max(button_height, 0.002),
                 press_depth=0.5 * button_height)
    lay = {
        "family": "hand", "room": copy.deepcopy(ROOM),
        "home": {"pos": [-1.0, -1.0], "size": HOME_SIZE},
        "pads": [{"color": "red", "pos": [1.0, 1.0], "size": PAD_SIZE},
                 {"color": "green", "pos": [-1.0, 1.0], "size": PAD_SIZE},
                 {"color": "blue", "pos": [1.0, -1.0], "size": PAD_SIZE}],
        "partition": None if empty else {"pos": [0.0, 1.0], "yaw": np.pi / 2, "length": 1.0,
                                         "attached": "N", **PARTITION},
        "button": {**b, "pos": list(button_pos)},
        "lamp": {"wall": "E", "offset": 0.0, "height": 0.3},
        "occluder": {**OCCLUDER, "pos": [-0.8, 0.3], "yaw": 0.0},
        "ball": {"pos": [-0.8, 0.6]},
        "start": {"pos": list(start[0]), "yaw": float(start[1])},
        "lamp_initial_on": False,
    }
    if empty:  # objects pushed to a corner, out of the way
        lay["occluder"]["pos"] = [-1.3, 1.3]
        lay["ball"]["pos"] = [-1.3, 1.0]
    cfg = {"id": "hand", "mujoco_version": MUJOCO_VERSION, "seed": 0, "layout": lay,
           "appearance": default_appearance(), "distractor": None}
    cfg["hash"] = config_hash(cfg)
    return cfg
