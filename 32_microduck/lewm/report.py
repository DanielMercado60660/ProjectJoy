"""R6 LeWM S1: gate table from results/lewm/eval_*.json and probes.jsonl -> results/lewm/gates.json.

    ../.venv/bin/python lewm/report.py runs/lewm_s1 runs/lewm_rand
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "lewm"


def probe_row(spec):
    rows = [json.loads(l) for l in open(RES / "probes.jsonl")]
    rows = [r for r in rows if r["encoder"] == spec]
    return rows[-1] if rows else None


def main():
    run, ctl = sys.argv[1], sys.argv[2]
    e = json.loads((RES / f"eval_{Path(run).name}.json").read_text())
    c = json.loads((RES / f"eval_{Path(ctl).name}.json").read_text())
    pl, pc = probe_row(f"lewm:{run}"), probe_row(f"rand:{ctl}")
    p2 = probe_row("p2:runs/p2_a2/ckpt_016000/weights.safetensors")
    g1 = e["G1"]
    gates = {
        "G1_no_collapse": {"frac_dims_std_0.5_2": g1["frac_std_in_0.5_2"], "effective_rank": g1["effective_rank"],
                           "sigreg_step1000": g1["sigreg_step1000"], "sigreg_last": g1["sigreg_last"],
                           "pass": bool(g1["frac_std_in_0.5_2"] >= 0.9 and g1["effective_rank"] >= 100
                                        and g1["sigreg_last"] < g1["sigreg_step1000"])},
        "G2_prediction": {"R_heldout": e["G2_heldout"]["R"], "R_p2dev": e["G2_p2dev"]["R"],
                          "control_R_heldout": c["G2_heldout"]["R"], "control_R_p2dev": c["G2_p2dev"]["R"],
                          "pass": bool(e["G2_heldout"]["R"] <= 0.70 and e["G2_p2dev"]["R"] <= 0.85)},
        "G3_action_ranking": {"top1_heldout": e["G3_heldout"]["top1"], "control_top1_heldout": c["G3_heldout"]["top1"],
                              "top1_p2dev": e["G3_p2dev"]["top1"], "control_top1_p2dev": c["G3_p2dev"]["top1"],
                              "chance": e["G3_heldout"]["chance"],
                              "pass": bool(e["G3_heldout"]["top1"] >= 0.5
                                           and e["G3_heldout"]["top1"] >= 3 * c["G3_heldout"]["top1"])},
        "G4_probes": {"lamp_acc": pl["lamp"]["test_acc"], "plate_auc": pl["plate"]["test_auc"],
                      "p2a_s0_lamp_acc": p2["lamp"]["test_acc"], "p2a_s0_plate_auc": p2["plate"]["test_auc"],
                      "control_lamp_acc": pc["lamp"]["test_acc"] if pc else None,
                      "control_plate_auc": pc["plate"]["test_auc"] if pc else None,
                      "pass": bool(pl["lamp"]["test_acc"] >= 0.90 and pl["plate"]["test_auc"] >= p2["plate"]["test_auc"])},
    }
    gates["S1_pass"] = all(v["pass"] for v in gates.values())
    print(json.dumps(gates, indent=1))
    (RES / "gates.json").write_text(json.dumps(gates, indent=1))


if __name__ == "__main__":
    main()
