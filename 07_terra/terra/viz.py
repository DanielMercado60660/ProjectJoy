"""Rendering helpers (matplotlib Agg, files only)."""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from .terrain import GRASS, FOREST, ROCK, WATER, BRIDGE

PALETTE = np.array([
    [0.62, 0.76, 0.42],  # grass
    [0.22, 0.45, 0.25],  # forest
    [0.60, 0.56, 0.52],  # rock
    [0.20, 0.42, 0.78],  # water
    [0.80, 0.52, 0.22],  # bridge
])

ROUTE_COLORS = {
    "Dijkstra": "#ffffff", "A*": "#ffd400", "D* Lite": "#ff3b30", "Weighted A* (w=2)": "#00e5ff",
    "Learned-h A*": "#ff2bd6", "Greedy": "#8a2be2",
}


def hillshade(elev, az=315, alt=45):
    gy, gx = np.gradient(elev * 24.0)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(az), np.radians(alt)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def terrain_rgb(elev, kind):
    base = PALETTE[kind]
    shade = 0.55 + 0.45 * hillshade(elev)
    tint = 0.85 + 0.25 * elev
    rgb = base * shade[..., None] * tint[..., None]
    water = kind == WATER
    rgb[water] = PALETTE[WATER] * (0.9 + 0.1 * shade[water, None])
    return np.clip(rgb, 0, 1)


def overlay_cells(rgb, nodes, w, color, alpha=0.45):
    out = rgb.copy()
    if len(nodes) == 0:
        return out
    idx = np.array(list(nodes))
    r, c = idx // w, idx % w
    out[r, c] = (1 - alpha) * out[r, c] + alpha * np.array(color)
    return out


def plot_path(ax, path, w, color, lw=2.2, label=None, ls="-"):
    if not path:
        return
    p = np.array([divmod(u, w) for u in path])
    ax.plot(p[:, 1], p[:, 0], ls, color="k", lw=lw + 1.6, alpha=0.6, solid_capstyle="round")
    ax.plot(p[:, 1], p[:, 0], ls, color=color, lw=lw, label=label, solid_capstyle="round")


def mark(ax, u, w, color, marker="o", size=90):
    r, c = divmod(u, w)
    ax.scatter([c], [r], s=size, c=color, marker=marker, edgecolors="k", linewidths=1.2, zorder=5)
