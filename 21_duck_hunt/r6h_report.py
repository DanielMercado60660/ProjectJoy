"""Markdown tables from results/r6h_eval.json -> results/r6h_tables.md (and stdout).
  ../.venv/bin/python r6h_report.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HO_D, HO_H = "dodge_jink", "lead_ca"          # held-out scripted opponents (never trained against)


def main():
    os.chdir(HERE)
    ev = json.load(open("results/r6h_eval.json"))
    M, dn, hn = ev["matrix"], ev["ducks"], ev["hunters"]
    g = lambda d, h, k: M[f"{d}|{h}"][k]
    out = [f"Held-out courses: {ev['n_courses']} (seed {ev['course_seed']}), play seed {ev['play_seed']}. "
           f"Held-out scripted opponents: duck `{HO_D}`, hunter `{HO_H}` (never trained against).", ""]
    learned_d = [d for d in dn if d.startswith("duck_")]
    learned_h = [h for h in hn if h.startswith("hunter_")]
    final_d, final_h = learned_d[-1], learned_h[-1]

    out += ["### Ducks vs scripted hunters (survival = completed and not hit; hit rate in brackets)", "",
            "| duck | naive | lead (KF) | lead_ca (held out) | lead_tuned | course time of survivors vs lead |", "|---|---|---|---|---|---|"]
    for d in ["straight", "jink", "dodge", HO_D, "duck_s", final_d]:  # noqa
        out.append(f"| {d} | " + " | ".join(f"{g(d, h, 'survival'):.3f} ({g(d, h, 'hit_rate'):.3f})" for h in ["naive", "lead", HO_H, "lead_tuned"])
                   + f" | {g(d, 'lead', 'mean_time_survivors'):.2f} s |")
    out += ["", "### Hunters vs scripted ducks (hit rate; shots per course in brackets)", "",
            "| hunter | straight | jink | dodge | dodge_jink (held out) |", "|---|---|---|---|---|"]
    for h in ["naive", "lead", HO_H, "lead_tuned", "hunter_s", final_h]:
        out.append(f"| {h} | " + " | ".join(f"{g(d, h, 'hit_rate'):.3f} ({g(d, h, 'shots'):.2f})" for d in ["straight", "jink", "dodge", HO_D]) + " |")

    out += ["", "### Self-play generations (hit rate of the hunter column against the duck row)", "",
            "Each duck_g is tested against: the scripted lead and lead_ca hunters, the final hunter, and the mean over "
            "every learned hunter in the archive. Each hunter_g likewise against scripted and learned ducks.", "",
            "| gen | duck_g vs lead | vs lead_ca (HO) | vs lead_tuned | vs final hunter | vs all learned hunters (mean) | "
            "hunter_g vs straight | vs dodge_jink (HO) | vs final duck | vs all learned ducks (mean) |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for i, (d, h) in enumerate(zip(learned_d, learned_h)):
        md = np.mean([g(d, x, "hit_rate") for x in learned_h])
        mh = np.mean([g(x, h, "hit_rate") for x in learned_d])
        out.append(f"| {d.replace('duck_', '')} | {g(d, 'lead', 'hit_rate'):.3f} | {g(d, HO_H, 'hit_rate'):.3f} | {g(d, 'lead_tuned', 'hit_rate'):.3f} | "
                   f"{g(d, final_h, 'hit_rate'):.3f} | {md:.3f} | {g('straight', h, 'hit_rate'):.3f} | "
                   f"{g(HO_D, h, 'hit_rate'):.3f} | {g(final_d, h, 'hit_rate'):.3f} | {mh:.3f} |")
    # final agents vs the whole archive: does the latest beat the past?
    out += ["", "### Final self-play agents vs every archived opponent", "",
            f"`{final_d}` vs hunters (hit rate): " + ", ".join(f"{h} {g(final_d, h, 'hit_rate'):.3f}" for h in hn), "",
            f"`{final_h}` vs ducks (hit rate): " + ", ".join(f"{d} {g(d, final_h, 'hit_rate'):.3f}" for d in dn), ""]
    # course completion / failures of learned ducks
    fails = {d: max(g(d, h, "crash") + g(d, h, "timeout") for h in hn) for d in dn}
    out += ["Worst-case crash+timeout rate per duck over all hunters: " +
            ", ".join(f"{d} {v:.3f}" for d, v in fails.items()), ""]
    txt = "\n".join(out)
    open("results/r6h_tables.md", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main()
