"""Analyse trained populations: accuracy, controls, message efficiency, compositionality,
partner swapping, symbol interventions (the receiver's 'dictionary'), and figures.

  ../.venv/bin/python analyze.py            # reads checkpoints/, writes results/ and media/
"""
from __future__ import annotations

import json
import os
from collections import Counter

import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from alien import (VALUE_NAMES, make_episodes, topographic_similarity, positional_disentanglement,  # noqa: E402
                   mi_matrix, message_stats)
from train import load_model, sender_table, receiver_choices, onehot  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.environ.get("ALIEN_CKPT", os.path.join(HERE, "checkpoints"))
RES = os.environ.get("ALIEN_RES", os.path.join(HERE, "results"))
MEDIA = os.environ.get("ALIEN_MEDIA", os.path.join(HERE, "media"))
N_EVAL = 20000
METHOD_LABEL = {"gumbel": "Gumbel-ST (emergent)", "reinforce": "REINFORCE (emergent)",
                "nocomm": "no communication", "compositional": "hand-designed compositional code",
                "holistic": "random holistic code", "reinforce_decay": "REINFORCE, entropy 0.1→0.01"}
COL = {"gumbel": "#2a6fdb", "reinforce": "#d9534f", "nocomm": "#888888", "compositional": "#2e9e5b",
       "holistic": "#c08a1e", "reinforce_decay": "#8e4ec6"}


def short_name(attr, i):
    v = VALUE_NAMES[attr][i]
    return "".join(w[0].upper() for w in v.split("-")) if attr == "location" else v[:6]


def ckpt(stage, method, tag=""):
    p = os.path.join(CK, f"{stage}_{method}{tag}_s0")
    return p if os.path.exists(p + ".json") else None


def hard_mask(world, cand, tpos):
    """Episodes where some distractor differs from the target in exactly one attribute."""
    o = world.objects[cand]
    tgt = o[np.arange(len(cand)), tpos]
    d = (o != tgt[:, None, :]).sum(-1)
    d[np.arange(len(cand)), tpos] = 99
    return d.min(1) == 1


def heldout_naming(world, tab, picks):
    """How are never-seen combinations named and read? Per member: fraction of held-out objects whose
    message is identical to some seen object's message, and how many attributes (of 3) the receiver's
    pick among all 128 objects shares with the held-out target."""
    coll, ov = [], []
    for k in range(tab.shape[0]):
        seen = {tuple(tab[k, i]) for i in world.train_ids}
        coll.append(float(np.mean([tuple(tab[k, i]) in seen for i in world.test_ids])))
        pk = picks[k, world.test_ids]
        ov.append(float(np.mean((world.objects[pk] == world.objects[world.test_ids]).sum(1))))
    return coll, ov


def acc_on(receiver, world, table, cand, tpos, V):
    tgt = cand[np.arange(len(cand)), tpos]
    return (receiver_choices(receiver, world, table[:, tgt], cand, V) == tpos[None]).mean(1)


def all_object_choices(receiver, world, msgs, V):
    """Receiver picks among ALL objects. msgs: (P, M, L) -> (P, M) chosen object ids."""
    P, M, _ = msgs.shape
    feats = mx.broadcast_to(mx.array(world.feats)[None, None], (P, 1, world.n, world.feats.shape[1]))
    out = []
    for i in range(0, M, 512):
        m = onehot(mx.array(msgs[:, i:i + 512]), V)
        c = mx.broadcast_to(feats, (P, m.shape[1], world.n, world.feats.shape[1]))
        out.append(np.array(mx.argmax(receiver(m, c), axis=-1)))
    return np.concatenate(out, axis=1)


def interventions(receiver, world, table, V):
    """For every object's own message, replace one position with every other symbol and see which
    attributes of the receiver's pick change. Returns
      change[p, a]  = fraction of single-symbol edits at position p that change attribute a
      decode[p][v]  = for each attribute, Counter of the attribute value picked when position p = v
    for each member (lists over P)."""
    P, N, L = table.shape
    A = world.objects.shape[1]
    edits, meta = [], []
    for o in range(N):
        for p in range(L):
            for v in range(V):
                m = table[:, o].copy()
                m[:, p] = v
                edits.append(m)
                meta.append((o, p, v))
    edits = np.stack(edits, axis=1)                                    # (P, N*L*V, L)
    picks = all_object_choices(receiver, world, edits, V)              # (P, N*L*V)
    meta = np.array(meta)
    out = []
    for k in range(P):
        own = {}
        for (o, p, v), c in zip(meta.tolist(), picks[k].tolist()):
            if v == table[k, o, p]:
                own[o] = c
        change = np.zeros((L, A)); count = np.zeros(L)
        decode = [[[Counter() for _ in range(A)] for _ in range(V)] for _ in range(L)]
        for (o, p, v), c in zip(meta.tolist(), picks[k].tolist()):
            for a in range(A):
                decode[p][v][a][int(world.objects[c, a])] += 1
            if v == table[k, o, p]:
                continue
            count[p] += 1
            change[p] += world.objects[c] != world.objects[own[o]]
        out.append({"change": (change / np.maximum(count, 1)[:, None]),
                    "decode": decode,
                    "self_acc_all128": float(np.mean([own[o] == o for o in range(N)]))})
    return out


def dictionary_table(world, decode, change, V):
    """Readable dictionary: for each position, the attribute its edits change most; for each symbol,
    the attribute value the receiver most reliably reads from it (consistency across all 128 contexts,
    i.e. fraction of messages with this symbol here whose pick has that value)."""
    L = len(decode)
    rows = []
    for p in range(L):
        a = int(np.argmax(change[p]))
        name = world.attr_names[a]
        entries = []
        for v in range(V):
            # the single attribute value this symbol most reliably evokes, scored above chance (1/d)
            best = None
            for b, nm in enumerate(world.attr_names):
                cnt = decode[p][v][b]
                val, n = cnt.most_common(1)[0]
                cons = n / sum(cnt.values())
                score = cons - 1.0 / world.dims[b]
                if best is None or score > best[0]:
                    best = (score, nm, VALUE_NAMES[nm][val], cons)
            entries.append({"symbol": v, "evokes": f"{best[1]}={best[2]}", "consistency": round(best[3], 3),
                            "chance": round(1.0 / world.dims[world.attr_names.index(best[1])], 3)})
        rows.append({"position": p, "controls": name, "edit_changes_attr": [round(x, 3) for x in change[p]],
                     "symbols": entries})
    return rows


def analyse_full():
    results = {"stage": "full"}
    runs = {m: ckpt("full", m) for m in ["gumbel", "reinforce", "nocomm", "compositional", "holistic"]}
    runs["reinforce_decay"] = ckpt("full", "reinforce", "_decay")
    runs = {m: p for m, p in runs.items() if p}
    loaded = {m: load_model(p) for m, p in runs.items()}
    world = next(iter(loaded.values()))[1]
    cfg0 = next(iter(loaded.values()))[2]
    V, L, K = cfg0["V"], cfg0["L"], cfg0["K"]
    tr = make_episodes(world, N_EVAL, K, "train", seed=777, cluster_frac=0.5)
    te = make_episodes(world, N_EVAL, K, "test", seed=778, cluster_frac=0.5)
    perm = np.random.RandomState(5).permutation(N_EVAL)
    tables = {}
    results["world"] = {"dims": world.dims, "n_objects": world.n, "n_train": len(world.train_ids),
                        "n_heldout": len(world.test_ids), "K": K, "L": L, "V": V, "chance": 1 / K,
                        "eval_episodes": N_EVAL,
                        "heldout_objects": [world.describe(i) for i in world.test_ids]}
    results["methods"] = {}
    for m, (model, w, cfg, meta) in loaded.items():
        tab = sender_table(model, world, cfg)
        tables[m] = tab
        r = {"label": METHOD_LABEL[m], "steps": meta["step"], "P": cfg["P"]}
        r["train_acc"] = acc_on(model.receiver, world, tab, *tr, V).tolist()
        r["heldout_acc"] = acc_on(model.receiver, world, tab, *te, V).tolist()
        # difficulty-matched view: held-out episodes have slightly fewer near-miss distractors
        # (44% vs 53% contain a distractor differing in ONE attribute), so also report accuracy on
        # exactly those "hard" episodes for both splits
        for key, (c_, t_) in [("train", tr), ("heldout", te)]:
            hard = hard_mask(world, c_, t_)
            r[f"{key}_acc_hard"] = acc_on(model.receiver, world, tab, c_[hard], t_[hard], V).tolist()
            r[f"{key}_frac_hard"] = float(hard.mean())
        # side-channel control: give each receiver the message of a DIFFERENT episode
        tgt = tr[0][np.arange(N_EVAL), tr[1]]
        sh = receiver_choices(model.receiver, world, tab[:, tgt][:, perm], tr[0], V)
        r["shuffled_message_acc"] = (sh == tr[1][None]).mean(1).tolist()
        r["topsim"] = [topographic_similarity(world.objects, tab[k]) for k in range(cfg["P"])]
        r["posdis"] = [positional_disentanglement(world.objects, tab[k]) for k in range(cfg["P"])]
        r["mi_matrix"] = [mi_matrix(world.objects, tab[k]).round(3).tolist() for k in range(cfg["P"])]
        r["message_stats"] = [message_stats(world.objects, tab[k], V) for k in range(cfg["P"])]
        r["heldout_unique_messages"] = [len({tuple(x) for x in tab[k, world.test_ids].tolist()})
                                        for k in range(cfg["P"])]
        # pick the target out of ALL 128 objects (much harder than K=5)
        picks = all_object_choices(model.receiver, world, tab, V)
        r["all128_acc_train_objs"] = [(picks[k, world.train_ids] == world.train_ids).mean() for k in range(cfg["P"])]
        r["all128_acc_heldout_objs"] = [(picks[k, world.test_ids] == world.test_ids).mean() for k in range(cfg["P"])]
        r["heldout_msg_same_as_a_seen_obj"], r["heldout_pick_attrs_shared"] = heldout_naming(world, tab, picks)
        r["log"] = meta["log"]
        results["methods"][m] = r
        print(f"{m:14s} train {np.mean(r['train_acc']):.3f}  heldout {np.mean(r['heldout_acc']):.3f}  "
              f"hard: {np.mean(r['train_acc_hard']):.3f}/{np.mean(r['heldout_acc_hard']):.3f}  "
              f"shuffled {np.mean(r['shuffled_message_acc']):.3f}  topsim {np.mean(r['topsim']):.3f}", flush=True)

    # partner swapping: sender from member a, receiver from member b (within and across methods)
    swaps = {}
    for ms in ["gumbel", "reinforce", "reinforce_decay", "compositional"]:
        for mr in ["gumbel", "reinforce", "reinforce_decay", "compositional"]:
            if ms not in loaded or mr not in loaded:
                continue
            Ts, rec = tables[ms], loaded[mr][0].receiver
            Pr = loaded[mr][2]["P"]
            M = np.zeros((Ts.shape[0], Pr))
            for a in range(Ts.shape[0]):
                M[a] = acc_on(rec, world, np.broadcast_to(Ts[a][None], (Pr, *Ts[a].shape)), *tr, V)
            swaps[f"{ms}->{mr}"] = M.round(4).tolist()
    results["partner_swap"] = swaps

    # interventions on the emergent protocols
    results["interventions"] = {}
    for m in ["gumbel", "reinforce", "reinforce_decay", "compositional"]:
        if m not in loaded:
            continue
        iv = interventions(loaded[m][0].receiver, world, tables[m], V)
        results["interventions"][m] = [{"change": x["change"].round(3).tolist(),
                                        "self_acc_all128": x["self_acc_all128"],
                                        "dictionary": dictionary_table(world, x["decode"], x["change"], V)}
                                       for x in iv]
        if m == "gumbel":
            gumbel_iv = iv
    results["_tables"] = {m: t.tolist() for m, t in tables.items()}
    return results, loaded, world, tables, (gumbel_iv if "gumbel" in loaded else None)


def analyse_color():
    out = {}
    for m in ["gumbel", "reinforce", "nocomm"]:
        p = ckpt("color", m)
        if not p:
            continue
        model, world, cfg, meta = load_model(p)
        tr = make_episodes(world, N_EVAL, cfg["K"], "train", seed=777)
        tab = sender_table(model, world, cfg)
        acc = acc_on(model.receiver, world, tab, *tr, cfg["V"])
        out[m] = {"train_acc": acc.tolist(), "P": cfg["P"], "steps": meta["step"],
                  "codebooks": {VALUE_NAMES["color"][o]: [int(tab[k, o, 0]) for k in range(cfg["P"])]
                                for o in range(world.n)},
                  "symbols_used": [len(np.unique(tab[k])) for k in range(cfg["P"])],
                  "log": meta["log"]}
        out[m]["n_perfect"] = int((acc >= 0.99).sum())
        print(f"color {m:10s} acc {np.round(acc, 3)} symbols used {out[m]['symbols_used']}", flush=True)
    # sensitivity sweep (checkpoints/sweep): final accuracy of every member
    sweep = {}
    sd = os.path.join(CK, "sweep")
    if os.path.isdir(sd):
        for f in sorted(os.listdir(sd)):
            if f.startswith("color_") and f.endswith(".json") and not f.endswith(".opt.json"):
                meta = json.load(open(os.path.join(sd, f)))
                accs = meta["log"][-1]["train_acc_members"]
                c = meta["cfg"]
                sweep[f[:-5]] = {"method": c["method"], "tau": c["tau"], "lr": c["lr"],
                                 "entropy": c["gumbel_entropy_coef"] if c["method"] == "gumbel" else c["entropy_coef"],
                                 "acc_members": accs, "n_perfect": int(sum(a >= 0.99 for a in accs))}
                print("sweep", f[:-5], np.round(accs, 3), flush=True)
    out["_sweep"] = sweep
    return out


def analyse_variants():
    """Every other full-stage checkpoint (tagged variants): accuracy, topsim, posdis, message use."""
    out = {}
    for f in sorted(os.listdir(CK)):
        if not (f.startswith("full_") and f.endswith("_s0.json")):
            continue
        name = f[:-len("_s0.json")]
        model, world, cfg, meta = load_model(os.path.join(CK, f[:-5]))
        if not cfg["tag"]:
            continue
        V = cfg["V"]
        # always evaluated on the standard K=5 rounds, whatever K/cluster_frac it was trained with
        tr = make_episodes(world, N_EVAL, 5, "train", seed=777, cluster_frac=0.5)
        te = make_episodes(world, N_EVAL, 5, "test", seed=778, cluster_frac=0.5)
        tab = sender_table(model, world, cfg)
        picks = all_object_choices(model.receiver, world, tab, V)
        r = {"method": cfg["method"], "L": cfg["L"], "V": V, "tau": cfg["tau"],
             "entropy": cfg["gumbel_entropy_coef"] if cfg["method"] == "gumbel" else cfg["entropy_coef"],
             "entropy_final": cfg.get("entropy_final"),
             "train_acc": acc_on(model.receiver, world, tab, *tr, V).tolist(),
             "heldout_acc": acc_on(model.receiver, world, tab, *te, V).tolist(),
             "topsim": [topographic_similarity(world.objects, tab[k]) for k in range(cfg["P"])],
             "posdis": [positional_disentanglement(world.objects, tab[k]) for k in range(cfg["P"])],
             "unique_messages": [message_stats(world.objects, tab[k], V)["unique_messages"] for k in range(cfg["P"])],
             "K_train": cfg["K"], "cluster_frac": cfg["cluster_frac"],
             "all128_acc_train_objs": [float((picks[k, world.train_ids] == world.train_ids).mean()) for k in range(cfg["P"])],
             "all128_acc_heldout_objs": [float((picks[k, world.test_ids] == world.test_ids).mean()) for k in range(cfg["P"])],
             **dict(zip(["heldout_msg_same_as_a_seen_obj", "heldout_pick_attrs_shared"], heldout_naming(world, tab, picks))),
             "log": meta["log"]}
        out[name] = r
        print(f"variant {name:28s} train {np.mean(r['train_acc']):.3f} heldout {np.mean(r['heldout_acc']):.3f} "
              f"topsim {np.mean(r['topsim']):.3f} all128 {np.mean(r['all128_acc_train_objs']):.3f}/"
              f"{np.mean(r['all128_acc_heldout_objs']):.3f}", flush=True)
    return out


# ----------------------------------------------------------------------------- figures
def fig_curves(res, color_res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    for m, r in color_res.items():
        if m.startswith("_"):
            continue
        st = [x["step"] for x in r["log"]]
        mem = np.array([x["train_acc_members"] for x in r["log"]])
        ax.plot(st, mem, color=COL[m], alpha=0.25, lw=0.8)
        ax.plot(st, mem.mean(1), color=COL[m], lw=2, label=METHOD_LABEL[m])
    ax.axhline(0.2, ls=":", color="k", lw=1)
    ax.set(title="Stage 1: colour only (8 colours, 1 symbol of 8)", xlabel="step", ylabel="accuracy (K=5)", ylim=(0, 1.02))
    ax.legend(fontsize=8, loc="lower right")
    for ax, key, title in [(axes[1], "train_acc", "Stage 2: seen combinations"),
                           (axes[2], "heldout_acc", "Stage 2: HELD-OUT combinations")]:
        for m, r in res["methods"].items():
            st = [x["step"] for x in r["log"]]
            mem = np.array([x[key + "_members"] for x in r["log"]])
            ax.plot(st, mem, color=COL[m], alpha=0.2, lw=0.8)
            ax.plot(st, mem.mean(1), color=COL[m], lw=2, label=METHOD_LABEL[m])
        ax.axhline(0.2, ls=":", color="k", lw=1)
        ax.set(title=title, xlabel="step", ylim=(0, 1.02))
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Alien Radio training curves (thin = individual seeds, thick = mean; dotted = chance)")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "training_curves.png"), dpi=130)
    plt.close(fig)


def fig_dictionary(world, table, iv, member, V, fname, title):
    """Symbol-vs-attribute heatmaps: for each message position, how often each symbol is used for
    each attribute value (sender side), plus the receiver-side intervention matrix."""
    L, A = table.shape[1], world.objects.shape[1]
    fig, axes = plt.subplots(L, A + 1, figsize=(3.3 * (A + 1), 2.6 * L),
                             gridspec_kw={"width_ratios": [d for d in world.dims] + [A + 1]})
    for p in range(L):
        for a in range(A):
            ax = axes[p, a]
            d = world.dims[a]
            M = np.zeros((V, d))
            for o in range(world.n):
                M[table[o, p], world.objects[o, a]] += 1
            M = M / M.sum(0, keepdims=True)
            ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(d))
            ax.set_xticklabels([short_name(world.attr_names[a], i) for i in range(d)], rotation=60, fontsize=7)
            ax.set_yticks(range(V))
            ax.set_yticklabels([f"s{v}" for v in range(V)], fontsize=7)
            if p == 0:
                ax.set_title(world.attr_names[a])
            if a == 0:
                ax.set_ylabel(f"position {p}")
        ax = axes[p, A]
        ch = np.array(iv["change"][p]) if iv is not None else np.zeros(A)
        ax.barh(range(A), ch, color="#2a6fdb")
        ax.set_yticks(range(A)); ax.set_yticklabels(world.attr_names, fontsize=8)
        ax.set_xlim(0, 1); ax.invert_yaxis()
        if p == 0:
            ax.set_title("edit this position →\nwhich attribute flips?", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, fname), dpi=130)
    plt.close(fig)


def fig_swap(res):
    keys = [k for k in ["gumbel->gumbel", "reinforce->reinforce", "gumbel->reinforce", "compositional->gumbel"]
            if k in res["partner_swap"]]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.4))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        M = np.array(res["partner_swap"][k])
        ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if M[i, j] < 0.6 else "k")
        s, r = k.split("->")
        ax.set(title=f"sender: {s}\nreceiver: {r}", xlabel="receiver seed", ylabel="sender seed")
    fig.suptitle("Partner swapping (accuracy, K=5, chance 0.20): diagonal = own partner", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "partner_swap.png"), dpi=130)
    plt.close(fig)


HEX = {"red": "#e5484d", "orange": "#f76b15", "yellow": "#f2c200", "green": "#30a46c", "cyan": "#05a2c2",
       "blue": "#3e63dd", "purple": "#8e4ec6", "pink": "#e93d82"}


def draw_obj(ax, world, i, x0, y0, s=1.0, edge=None):
    from matplotlib.patches import Rectangle, Circle, RegularPolygon, Polygon
    c, sh, loc = world.objects[i]
    col = HEX[VALUE_NAMES["color"][c]]
    ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec=edge or "#bbbbbb", lw=3 if edge else 1))
    ax.plot([x0 + s / 2] * 2, [y0 + .05 * s, y0 + .95 * s], color="#dddddd", lw=.6, ls=":")
    ax.plot([x0 + .05 * s, x0 + .95 * s], [y0 + s / 2] * 2, color="#dddddd", lw=.6, ls=":")
    cx = x0 + (0.75 if loc % 2 else 0.25) * s
    cy = y0 + (0.25 if loc >= 2 else 0.75) * s
    r = 0.17 * s
    if sh == 0:
        ax.add_patch(Circle((cx, cy), r, color=col))
    elif sh == 1:
        ax.add_patch(Rectangle((cx - r, cy - r), 2 * r, 2 * r, color=col))
    elif sh == 2:
        ax.add_patch(RegularPolygon((cx, cy), 3, radius=r * 1.25, color=col))
    else:
        pts = [(cx + (r * 1.2 if k % 2 == 0 else r * .5) * np.cos(np.pi / 2 + k * np.pi / 5),
                cy + (r * 1.2 if k % 2 == 0 else r * .5) * np.sin(np.pi / 2 + k * np.pi / 5)) for k in range(10)]
        ax.add_patch(Polygon(pts, color=col))


def fig_rounds(world, model, table, member, V, K, fname):
    """A strip of example rounds: what the sender saw, the message, the receiver's probabilities."""
    rows = []
    for split, seed in [("train", 31), ("train", 32), ("test", 33), ("test", 34), ("test", 35)]:
        c, t = make_episodes(world, 1, K, split, seed=seed)
        rows.append((split, c[0], int(t[0])))
    fig, ax = plt.subplots(figsize=(11, 1.55 * len(rows)))
    ax.set_xlim(0, 11); ax.set_ylim(0, 1.55 * len(rows)); ax.axis("off")
    for r, (split, cand, tpos) in enumerate(rows):
        y = 1.55 * (len(rows) - 1 - r) + 0.3
        tgt = cand[tpos]
        msg = table[tgt]
        P = model.receiver.m1.weight.shape[0]
        m = onehot(mx.array(np.broadcast_to(msg[None, None], (P, 1, len(msg)))), V)
        cf = mx.broadcast_to(mx.array(world.feats[cand])[None, None], (P, 1, K, world.feats.shape[1]))
        prob = np.array(mx.softmax(model.receiver(m, cf), axis=-1))[member, 0]
        draw_obj(ax, world, tgt, 0.1, y, 1.0)
        ax.text(0.6, y - 0.18, ("seen: " if split == "train" else "NEW: ") + world.describe(tgt), ha="center",
                fontsize=7, color="k" if split == "train" else "#d07a00")
        ax.text(1.85, y + 0.5, "  ".join(f"s{v}" for v in msg), ha="center", va="center", fontsize=13,
                family="monospace", color="#7cf2c1", bbox=dict(boxstyle="round", fc="#10151a", ec="none"))
        ax.annotate("", (2.95, y + 0.5), (2.6, y + 0.5), arrowprops=dict(arrowstyle="->", color="#999999"))
        pick = int(np.argmax(prob))
        for k in range(K):
            x = 3.0 + k * 1.6
            edge = ("#2e9e5b" if k == tpos else "#d9534f") if k == pick else None
            draw_obj(ax, world, cand[k], x, y, 1.0, edge=edge)
            ax.add_patch(matplotlib.patches.Rectangle((x, y - 0.17), prob[k], 0.07, color="#2a6fdb"))
            ax.text(x + 1.05, y - 0.15, f"{prob[k]*100:.0f}%" + (" *" if k == tpos else ""), fontsize=7, va="center")
    ax.set_title(f"Gumbel pair (seed {member}): sender's view -> 3-symbol message -> receiver's probabilities over 5 shuffled "
                 "candidates\n(green box = receiver's pick was right, red = wrong, * = true target)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, fname), dpi=130)
    plt.close(fig)


def fig_summary(res):
    ms = list(res["methods"])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(ms))
    for off, key, lab in [(-0.2, "train_acc", "seen combinations"), (0.2, "heldout_acc", "held-out combinations")]:
        vals = [np.mean(res["methods"][m][key]) for m in ms]
        ax.bar(x + off, vals, 0.38, label=lab, color=[COL[m] for m in ms], alpha=1.0 if off < 0 else 0.5,
               edgecolor="k")
        for k, m in enumerate(ms):
            ax.scatter(np.full(len(res["methods"][m][key]), x[k] + off), res["methods"][m][key], s=8, color="k", zorder=3)
    ax.axhline(0.2, ls=":", color="k")
    ax.set_xticks(x); ax.set_xticklabels([METHOD_LABEL[m].replace(" (", "\n(").replace("designed ", "designed\n").replace("random ", "random\n").replace("no comm", "no\ncomm") for m in ms], fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy (K=5)")
    ax.set_title("Solid = seen combinations, faded = held-out combinations; dots = seeds")
    fig.tight_layout()
    fig.savefig(os.path.join(MEDIA, "summary_bars.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True)
    os.makedirs(MEDIA, exist_ok=True)
    color_res = analyse_color()
    res, loaded, world, tables, giv = analyse_full()
    res["color_stage"] = color_res
    res["variants"] = analyse_variants()
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    fig_curves(res, color_res)
    fig_summary(res)
    fig_swap(res)
    V = res["world"]["V"]
    if "gumbel" in tables:
        g = res["methods"]["gumbel"]
        best = int(np.argmax(np.array(g["heldout_acc"]) + 1e-3 * np.array(g["topsim"])))
        res["demo_member"] = best
        fig_dictionary(world, tables["gumbel"][best], res["interventions"]["gumbel"][best], best, V,
                       "dictionary_gumbel.png",
                       f"Gumbel seed {best}: sender's symbol use per attribute value (columns sum to 1), "
                       f"and receiver interventions")
        fig_rounds(world, loaded["gumbel"][0], tables["gumbel"][best], best, V, res["world"]["K"], "example_rounds.png")
    for m in ["reinforce", "compositional"]:
        if m in tables:
            k = int(np.argmax(res["methods"][m]["heldout_acc"]))
            fig_dictionary(world, tables[m][k], res["interventions"][m][k], k, V, f"dictionary_{m}.png",
                           f"{METHOD_LABEL[m]} seed {k}")
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("wrote results/metrics.json and media/*.png")


if __name__ == "__main__":
    main()
