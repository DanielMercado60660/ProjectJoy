"""Continuing life: one JSON file holds everything needed to carry on exactly where the duck stopped.

A session file = the full config, the World state (MuJoCo integration state incl. solver warmstart,
the duck's controller buffers, lamp flag, button latch, flicker RNG, step counter) and the bot's
memory (any JSON-able dict). Python's float repr round-trips float64 exactly, so a resumed session
continues bit for bit (tests/test_m1.py and m1_checks.py check this, the latter across processes).
Phase 1 keeps it at that: no learning between sessions.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np

from room import World

FORMAT = "microduck-session/1"


def _jsonable(state):
    s = dict(state)
    s["physics"] = np.asarray(state["physics"], np.float64).tolist()
    if state.get("physics_pre") is not None:
        s["physics_pre"] = np.asarray(state["physics_pre"], np.float64).tolist()
    return s


def save(path, world: World, memory: dict | None = None):
    path = Path(path)
    doc = {"format": FORMAT, "saved": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
           "step_counter": world.step_count, "lamp_on": world.lamp_on,
           "config": world.cfg, "state": _jsonable(world.get_state()), "memory": memory or {}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc))
    tmp.replace(path)   # atomic: a crash mid-write never leaves a half session
    return path


def new(path, cfg, memory: dict | None = None, policy=None):
    """Start a fresh life from a config and write its first session file."""
    world = World(cfg, policy=policy)
    save(path, world, memory)
    return world, dict(memory or {})


def start(path, policy=None):
    """Resume a session: rebuild the world from its config and restore the saved state."""
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} file")
    world = World(doc["config"], policy=policy)
    world.set_state(doc["state"])
    return world, doc["memory"]
