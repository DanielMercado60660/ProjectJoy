"""Phase 2 open-loop lamp diagnostic (results/p2/protocol.json).

    python p2_diag.py <weights> [<weights> ...]      # -> results/p2/diag_<name>.json
On the 80 dev bot demos (episodes/p2_dev, never trained on), the policy is fed the bot's frames, body and
instruction (teacher-forced, GRU carried from tick 0). At the bot's first lamp check (tick c, the bot's
own on/off reading), the bot says done if the lamp matches the target and goes to press otherwise. Reports:
  auc         AUC of the done logit at tick c, match episodes vs mismatch episodes (1 = perfectly tied)
  early_done  mismatch episodes where the done logit is > 0 anywhere in ticks c-29..c
  aux         match/lamp head accuracy on ticks where the bot's belief is known (P2 models only)
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from learner import Policy, body_vec, instr_vec, prep_frame  # noqa: E402

mx.set_default_device(mx.cpu)


def load_model(w):
    if (Path(w).parent / "p2_config.json").exists():
        from p2_learner import Policy2, policy_config
        m = Policy2(**policy_config(w))
        assert m.image_size == 64, "p2_diag feeds 64x64 frames"
    else:
        m = Policy()
    m.load_weights(str(w))
    mx.eval(m.parameters())
    return m


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else None


def episodes():
    for line in open(HERE / "episodes" / "p2_dev" / "index.jsonl"):
        r = json.loads(line)
        z = np.load(HERE / r["path"] if not Path(r["path"]).is_absolute() else r["path"])
        H = json.loads(str(z["header"]))
        chk = [e for e in H["events"] if e["event"] == "lamp_check" and e.get("state") in ("on", "off")]
        if not chk:
            continue
        c = chk[0]["tick"]
        match = chk[0]["state"] == H["instruction"]["target"]
        frames = np.stack([prep_frame(f) for f in z["frames"][:c + 1]])
        body = np.stack([body_vec(b) for b in z["body"][:c + 1]])
        mem = json.loads(str(z["memory"]))[:c + 1]
        yield r, H, c, match, frames, body, mem


def run(w):
    m = load_model(w)
    rows = []
    for r, H, c, match, F, B, mem in episodes():
        n = len(B)
        I = np.broadcast_to(instr_vec(H["instruction"]), (n, 9))
        out = m(mx.array(F[None].astype(np.float32) / 255.0), mx.array(B[None]), mx.array(I[None]))
        dl = np.array(out[1][0])
        row = {"config": r["config"], "c": c, "match": bool(match), "logit_c": float(dl[c]),
               "max_logit_window": float(dl[max(0, c - 29):c + 1].max())}
        if getattr(m, "done_from_aux", False):
            # Counterfactual (Astra's suggestion): force the match input to "matches" vs "mismatch" at tick c
            # and see how far the done logit moves. Near zero = the done head ignores the match belief.
            import mlx.nn as nn
            z = m.encode(mx.array(F[None].astype(np.float32) / 255.0), mx.array(B[None]), mx.array(I[None]))
            hs = m.gru(z)
            y = nn.relu(m.h1(mx.concatenate([hs, nn.relu(m.ins(mx.array(I[None])))], -1)))[:, c]
            lo = {k: float(m.done(mx.concatenate([y, mx.array([v])], -1))[0, 0])
                  for k, v in (("matches", [0.0, 1.0, 0.0]), ("mismatch", [0.0, 0.0, 1.0]))}
            row["cf_done_logit"] = lo
        if len(out) == 4 and out[2]:
            aux = {k: np.array(v[0]).argmax(-1) for k, v in out[2].items()}
            tgt = H["instruction"]["target"]
            pressed = [e["tick"] for e in H["events"] if e["event"] == "press"]
            ok = {"lamp": [], "match": []}
            for t in range(n):
                L = mem[t]["lamp"]
                lp = max([p for p in pressed if p <= t], default=-1)
                if L is None or L[1] <= lp:
                    continue
                ok["lamp"].append(int(aux["lamp"][t] == (1 if L[0] == "on" else 2)))
                ok["match"].append(int(aux["match"][t] == (1 if L[0] == tgt else 2)))
            row["aux_ok"] = {k: [sum(v), len(v)] for k, v in ok.items()}
        rows.append(row)
    pos = [x["logit_c"] for x in rows if x["match"]]
    neg = [x["logit_c"] for x in rows if not x["match"]]
    res = {"weights": str(w), "episodes": len(rows), "match": len(pos), "mismatch": len(neg),
           "auc": None if auc(pos, neg) is None else round(auc(pos, neg), 3),
           "match_done_at_c": int(sum(p > 0 for p in pos)),
           "early_done_mismatch": int(sum(x["max_logit_window"] > 0 for x in rows if not x["match"])), "rows": rows}
    if rows and "cf_done_logit" in rows[0]:
        res["cf_mean_logit_matches_minus_mismatch"] = round(float(np.mean(
            [x["cf_done_logit"]["matches"] - x["cf_done_logit"]["mismatch"] for x in rows])), 3)
    if rows and "aux_ok" in rows[0]:
        res["aux_acc"] = {k: round(sum(x["aux_ok"][k][0] for x in rows) / max(1, sum(x["aux_ok"][k][1] for x in rows)), 3)
                          for k in ("lamp", "match")}
    return res


def main():
    out = HERE / "results" / "p2"
    for w in sys.argv[1:]:
        res = run(w)
        name = Path(w).parent.name if Path(w).name == "weights.safetensors" else Path(w).stem
        if Path(w).name == "weights.safetensors":
            name = Path(w).parent.parent.name + "_" + name
        (out / f"diag_{name}.json").write_text(json.dumps(res, indent=1))
        print(name, {k: v for k, v in res.items() if k != "rows"}, flush=True)


if __name__ == "__main__":
    main()
