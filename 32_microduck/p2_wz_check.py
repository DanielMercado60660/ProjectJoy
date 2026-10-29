"""Phase 2, amendment 5, check B: is the policy's turn command too weak or unsure even in the bot's states?

    python p2_wz_check.py NAME=WEIGHTS [...]     # -> results/p2/wz_check.jsonl (one line per policy)
    python p2_wz_check.py --yaw dagger3            # replay-only: does a commanded turn in place rotate the duck?

Teacher-forced on the 80 dev bot demos (episodes/p2_dev), as p2_vx_check. On the bot's turn-in-place ticks
(|wz| > 0.1, vx == 0; the bot then commands exactly +-0.55): share of policy wz with the bot's sign, mean
|wz|, share with |wz| >= 0.5 (wz 0.3 does not start a turn in place on this body; +-0.5..0.6 does,
results/m0/body_frozen.json). On turning-while-walking ticks: sign share and mean |wz|.

--yaw: on a recorded round's episodes (policy driving), the ticks where the executed command is a turn in
place (vx < 0.1, |wz| > 0.1), binned by |wz|: share where the duck's yaw moved >= 0.05 rad the commanded way
over the next 3 ticks. Uses the replay-only truth column for analysis; never a label or an input.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def teacher_forced(name, w):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from learner import body_vec, instr_vec, prep_frame
    from p2_learner import Policy2, decode, policy_config
    cfg = policy_config(w)
    m = Policy2(**cfg)
    m.load_weights(w)
    P, Bt, Wk = [], [], []
    for line in open(HERE / "episodes" / "p2_dev" / "index.jsonl"):
        r = json.loads(line)
        z = np.load(HERE / r["path"])
        H = json.loads(str(z["header"]))
        F = np.stack([prep_frame(f) for f in z["frames"]])
        B = np.stack([body_vec(b) for b in z["body"]])
        I = np.broadcast_to(instr_vec(H["instruction"]), (len(B), 9))
        reg, dl, aux, _ = m(mx.array(F[None].astype(np.float32) / 255.0), mx.array(B[None]), mx.array(I[None]))
        reg, dl = np.array(reg[0]), np.array(dl[0])
        walk = np.array(aux["walk"][0]) if cfg["vx_head"] == "binary" else None
        turn = np.array(aux["turn"][0]) if cfg["turn_head"] == "discrete" else None
        cb = z["cmd_bot"]
        for t in range(len(cb)):
            if not np.isfinite(cb[t]).all():
                continue
            c = decode(reg[t], dl[t], cfg["vx_head"], None if walk is None else float(walk[t]),
                       None if turn is None else turn[t])
            P.append(c.wz)
            Bt.append(cb[t, 2])
            Wk.append(cb[t, 0])
    P, Bt, Wk = map(np.array, (P, Bt, Wk))
    out = {"policy": name, "weights": w}
    for grp, mask in (("turn_in_place", (np.abs(Bt) > 0.1) & (np.abs(Wk) < 1e-6)),
                      ("turn_while_walking", (np.abs(Bt) > 0.1) & (Wk > 0.15)),
                      ("bot_no_turn", np.abs(Bt) <= 0.1)):
        p, b = P[mask], Bt[mask]
        row = {"ticks": int(mask.sum()), "mean_abs_wz_policy": round(float(np.abs(p).mean()), 3),
               "mean_abs_wz_bot": round(float(np.abs(b).mean()), 3)}
        if grp != "bot_no_turn":
            row["same_sign"] = round(float((np.sign(p) == np.sign(b)).mean()), 3)
        if grp == "turn_in_place":
            row["share_abs_ge_0.5"] = round(float((np.abs(p) >= 0.5).mean()), 3)
            row["share_abs_ge_0.5_and_same_sign"] = round(float(((np.abs(p) >= 0.5) & (np.sign(p) == np.sign(b))).mean()), 3)
            row["abs_wz_quantiles_5_25_50_75_95"] = np.quantile(np.abs(p), [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist()
            # Unsure direction should show up as small |wz| where the sign is wrong.
            wrong = np.sign(p) != np.sign(b)
            row["mean_abs_wz_when_sign_wrong"] = round(float(np.abs(p[wrong]).mean()), 3) if wrong.any() else None
            row["mean_abs_wz_when_sign_right"] = round(float(np.abs(p[~wrong]).mean()), 3) if (~wrong).any() else None
        out[grp] = row
    out["decision"] = {"sign_ok": out["turn_in_place"]["same_sign"] >= 0.9,
                           "strength_ok": out["turn_in_place"]["share_abs_ge_0.5"] >= 0.7}
    return out


def yaw_check(tag):
    bins = [(0.1, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.7)]
    hit = {b: [0, 0] for b in bins}
    for line in open(HERE / "episodes" / f"{tag}_index.jsonl"):
        r = json.loads(line)
        z = np.load(HERE / r["path"])
        ce, tr = z["cmd_exec"], z["truth"]
        yaw = np.unwrap(tr[:, 2])
        for t in range(len(ce) - 3):
            vx, wz = ce[t, 0], ce[t, 2]
            if vx >= 0.1 or abs(wz) <= 0.1:
                continue
            d = (yaw[t + 3] - yaw[t]) * np.sign(wz)
            for b in bins:
                if b[0] < abs(wz) <= b[1]:
                    hit[b][0] += int(d >= 0.05)
                    hit[b][1] += 1
    return {"tag": tag, "rotates_commanded_way_over_3_ticks": {f"{b[0]}-{b[1]}": {"share": round(h[0] / h[1], 3) if h[1] else None,
                                                                              "ticks": h[1]} for b, h in hit.items()}}


def main():
    if sys.argv[1] == "--yaw":
        out = yaw_check(sys.argv[2])
    else:
        out = None
        for arg in sys.argv[1:]:
            name, w = arg.split("=", 1)
            out = teacher_forced(name, w)
            with open(HERE / "results" / "p2" / "wz_check.jsonl", "a") as f:
                f.write(json.dumps(out) + "\n")
            print(json.dumps(out), flush=True)
        return
    with open(HERE / "results" / "p2" / "wz_check.jsonl", "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
