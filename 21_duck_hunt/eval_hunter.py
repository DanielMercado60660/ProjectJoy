"""Hunter game evaluation (R6): every duck vs every hunter on 500 held-out courses (new course
seed, fixed wind/noise seed). Held-out opponents (never trained against): dodge_jink, lead_ca.

  ../bin/gpu_run ../.venv/bin/python eval_hunter.py
      -> results/r6h_eval.json, media/r6h_matrix.png
"""
import json, os
import numpy as np
import hunter as H

HERE = os.path.dirname(os.path.abspath(__file__))


def roster():
    arch = json.load(open(os.path.join(HERE, "results", "r6h_selfplay_archive.json")))
    ducks = ["straight", "jink", "dodge", "dodge_jink"] + [s for s in arch["arch_d"] if not isinstance(s, str)]
    hunters = ["naive", "lead", "lead_ca"] + (["lead_tuned"] if "lead_tuned" in H.SCRIPTED else []) + [s for s in arch["arch_h"] if not isinstance(s, str)]
    return ducks, hunters


def name(s):
    return s if isinstance(s, str) else s["name"]


def main(n=500):
    os.chdir(HERE)
    ducks, hunters = roster()
    C = H.heldout_courses(n)
    M = {}
    for d in ducks:
        for h in hunters:
            rng = np.random.default_rng(0)
            r = H.play(C, H.make_agent(d, n, rng), H.make_agent(h, n, rng), seed=7)
            M[f"{name(d)}|{name(h)}"] = H.summarize(r)
        print(name(d), {name(h): round(M[f"{name(d)}|{name(h)}"]["survival"], 3) for h in hunters}, flush=True)
    dn, hn = [name(d) for d in ducks], [name(h) for h in hunters]
    out = dict(n_courses=n, course_seed=5_777_000, play_seed=7, ducks=dn, hunters=hn, matrix=M)
    json.dump(out, open("results/r6h_eval.json", "w"), indent=1)
    plot(out)


def plot(out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dn, hn, M = out["ducks"], out["hunters"], out["matrix"]
    fig, axs = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, key, title, cmap in [(axs[0], "survival", "duck survival rate (completed, not hit)", "RdYlGn"),
                                 (axs[1], "hit_rate", "hunter hit rate", "RdYlGn_r")]:
        A = np.array([[M[f"{d}|{h}"][key] for h in hn] for d in dn])
        im = ax.imshow(A, vmin=0, vmax=1, cmap=cmap)
        ax.set_xticks(range(len(hn))); ax.set_xticklabels(hn, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(dn))); ax.set_yticklabels(dn, fontsize=7)
        for i in range(len(dn)):
            for j in range(len(hn)):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=5.5)
        ax.set_title(title + "; rows: ducks, cols: hunters", fontsize=9)
        ax.set_xlabel("hunter"); ax.set_ylabel("duck")
    plt.tight_layout(); plt.savefig("media/r6h_matrix.png", dpi=110)


if __name__ == "__main__":
    main()
