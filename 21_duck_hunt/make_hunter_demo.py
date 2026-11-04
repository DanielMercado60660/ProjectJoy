"""Self-contained browser demo of the hunter game: media/hunter_demo.html (no network needed).
Replays recorded held-out episodes for a few matchups; pick one, play/pause, step, speed.

  ../.venv/bin/python make_hunter_demo.py
"""
import json, os
import numpy as np
import flight as F
import hunter as H

HERE = os.path.dirname(os.path.abspath(__file__))


def episodes(duck_spec, hunt_spec, idx, label):
    C = H.heldout_courses(500)[idx]
    rng = np.random.default_rng(0)
    r = H.play(C, H.make_agent(duck_spec, len(C), rng), H.make_agent(hunt_spec, len(C), rng), seed=7, record=True)
    out = []
    for i in range(len(C)):
        fr = [f for f in r["rec"] if f["alive"][i]]
        res = "HIT" if r["hit"][i] else "SURVIVED" if r["complete"][i] else "CRASHED" if r["crash"][i] else "TIMED OUT"
        out.append(dict(label=label, course=int(idx[i]), result=res, shots=int(r["shots"][i]),
                        W=np.round(C[i], 2).tolist(),
                        p=np.round([f["p"][i] for f in fr], 2).tolist(), c=np.round([f["c"][i] for f in fr], 2).tolist(),
                        z=np.round([f["z"][i] for f in fr], 2).tolist(), k=[int(f["k"][i]) for f in fr],
                        pend=[int(f["pend"][i]) for f in fr], aim=np.round([f["aim"][i] for f in fr], 2).tolist()))
    return out


def main():
    os.chdir(HERE)
    arch = json.load(open("results/r6h_selfplay_archive.json"))
    d_last, h_last = arch["arch_d"][-1], arch["arch_h"][-1]
    s1 = json.load(open("results/r6h_stage1.json"))
    pick = np.array([0, 1, 2, 3])
    eps = []
    eps += episodes(d_last, h_last, pick, f"self-play duck {d_last['name']} vs self-play hunter {h_last['name']}")
    eps += episodes(d_last, "lead", pick, f"self-play duck {d_last['name']} vs scripted lead-KF hunter")
    eps += episodes("straight", h_last, pick, f"scripted PD+ duck vs self-play hunter {h_last['name']}")
    eps += episodes("straight", "lead", pick, "scripted PD+ duck vs scripted lead-KF hunter")
    eps += episodes(s1["duck_s"], "lead", pick, "stage-1 duck (trained vs scripted) vs lead-KF hunter")
    eps += episodes(d_last, "lead_tuned", pick, f"self-play duck {d_last['name']} vs TUNED lead-KF hunter (never trained against)")
    eps += episodes("straight", "lead_tuned", pick, "scripted PD+ duck vs TUNED lead-KF hunter")
    eps += episodes(d_last, "lead_ca", pick, f"self-play duck {d_last['name']} vs HELD-OUT lead_ca hunter (const-accel KF)")
    # preserved failure: the scripted dodger flies itself into the ground when shots come low
    C = H.heldout_courses(500); rng = np.random.default_rng(0)
    r = H.play(C, H.make_agent("dodge_jink", 500, rng), H.make_agent("naive", 500, rng), seed=7)
    crash = np.nonzero(r["crash"])[0][:3]
    eps += episodes("dodge_jink", "naive", np.sort(crash), "FAILURE: scripted dodge_jink duck crashes while dodging (vs naive hunter)")
    summary = json.load(open("results/r6h_eval.json"))["matrix"] if os.path.exists("results/r6h_eval.json") else {}
    html = open("hunter_demo_template.html").read()
    html = html.replace("__EPISODES__", json.dumps(eps, separators=(",", ":")))
    html = html.replace("__CONST__", json.dumps(dict(DT=H.DT, HIT_R=H.HIT_R, WP_R=F.WP_R, T_LIM=H.T_LIM,
                                                     ARENA=list(F.ARENA), DELAY=H.DELAY)))
    os.makedirs("media", exist_ok=True)
    open("media/hunter_demo.html", "w").write(html)
    print("wrote media/hunter_demo.html", len(html) // 1024, "KB,", len(eps), "episodes")


if __name__ == "__main__":
    main()
