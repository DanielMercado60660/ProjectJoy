"""Phase 2 labels from recorded bot perception only. NumPy + standard library; no MLX.

Read `header` and `memory` with json.loads(str(npz[key])). Pass the ORIGINAL memory
list to inview_labels, then select the cache's ep['ticks'] (first-done/NaN cleaning).
Absence here means 'no confirmed teacher detection this tick', not physical absence.
"""
import warnings

import numpy as np

INVIEW_CLASSES = ("pad_red", "pad_green", "pad_blue", "pad_yellow", "button",
                  "ball", "mat", "lamp_on", "lamp_off")
LABEL_VERSION = "p2_inview_v1"


def inview_labels(header, memory, *, unknown_counts=None):
    """Return (T, 9) uint8 multi-hot labels from confirmed track snapshots.

    A track row is [cls, x, y, n_sightings, last_seen_tick]. The recorder already
    applied confirmation; do not impose another threshold or use remembered lamp state.
    Multiple tracks of one class set one bit; simultaneous on/off detections set both.
    `header` is accepted for the episode API but no header truth/outcome affects labels.

    Unknown classes are counted per track-row occurrence, including stale rows (NOT
    unique objects). Pass a mutable dict/Counter to accumulate counts over episodes.
    Without one, a warning reports this episode's counts so they aren't silently lost.
    """
    lookup = {name: k for k, name in enumerate(INVIEW_CLASSES)}
    labels = np.zeros((len(memory), len(INVIEW_CLASSES)), np.uint8)
    ignored = {}
    for t, snapshot in enumerate(memory):
        for row in snapshot.get("tracks", []):
            if len(row) != 5:
                raise ValueError(f"track at tick {t} must have five fields")
            cls, _, _, _, last_seen = row
            if cls not in lookup:
                ignored[cls] = ignored.get(cls, 0) + 1
            elif last_seen == t:
                labels[t, lookup[cls]] = 1
    if unknown_counts is not None:
        for cls, count in ignored.items():
            unknown_counts[cls] = unknown_counts.get(cls, 0) + count
    elif ignored:
        warnings.warn(f"ignored unknown track classes (track-row occurrences): {ignored}",
                      UserWarning, stacklevel=2)
    return labels


def observable_mask(lamp_lbl, memory, window_start, *, ticks=None):
    """Return a bool mask for window-aligned lamp labels (0 unknown, 1 on, 2 off).

    Unknown (0) passes; masked (-1) does not. A known label passes only if its recorded
    supporting sighting is at/after ORIGINAL window_start. This does not replace the
    press-staleness rule used to construct lamp_lbl, or the existing burn-in/padding M.

    Without `ticks`, memory must already be aligned with lamp_lbl (e.g. both sliced
    to the window). With `ticks`, memory is the full original episode; ticks gives the
    original indices of the window's labels. In a batch built from p2_train's cache:

        ix = ep['ticks'][s:s+n]
        visible = observable_mask(ep['lamp_lbl'][s:s+n], original_memory,
                                  int(ix[0]), ticks=ix)
        M_aux[b, :n] = M[b, :n] * visible

    Window start includes burn-in, since those frames also reach the GRU. Apply this
    mask to lamp/match auxiliaries, not to command/done loss. A sighting dropped as a
    NaN command row is not checked for membership here: this is the requested timestamp
    rule, not a guarantee that every supporting image survived filtering.
    """
    labels = np.asarray(lamp_lbl)
    if labels.ndim != 1 or not np.isin(labels, (-1, 0, 1, 2)).all():
        raise ValueError("lamp_lbl must be a 1D array with labels -1, 0, 1, 2")
    if not isinstance(window_start, (int, np.integer)) or window_start < 0:
        raise ValueError("window_start must be a nonnegative original tick")
    if ticks is None:
        if len(memory) != len(labels):
            raise ValueError("align memory with lamp_lbl, or supply original ticks")
        selected = memory
    else:
        ix = np.asarray(ticks)
        if ix.shape != labels.shape or not np.issubdtype(ix.dtype, np.integer):
            raise ValueError("ticks must be an integer array aligned with lamp_lbl")
        if len(ix) and (np.any(ix < window_start) or np.any(ix >= len(memory)) or np.any(ix[1:] <= ix[:-1])):
            raise ValueError("ticks must increase within the original episode and start at/after window_start")
        selected = [memory[int(t)] for t in ix]
    mask = labels == 0
    for i, (label, snapshot) in enumerate(zip(labels, selected)):
        if label > 0:
            sighting = snapshot.get("lamp")
            mask[i] = sighting is not None and sighting[1] >= window_start
    return mask.astype(np.bool_, copy=False)
