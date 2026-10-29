"""Offline follow-up report from a complete saved checkpoint; no GUI/network."""
import argparse
from html import escape
import json
from pathlib import Path

from r5_core import Guard
from r5_report import chart, replay_svg
from run_r5 import ROOT, atomic_json, read_json
from run_r5_turn2 import summarize_followup


def render(checkpoint, output, summary_path, max_minutes):
    guard = Guard(max_minutes)
    for name in (output, summary_path):
        path = Path(name)
        if not path.resolve().is_relative_to(ROOT) or path.exists():
            raise ValueError("Choose new report paths inside this project")
    state = read_json(checkpoint)
    summary = summarize_followup(state)
    summary["checkpoint"] = str(Path(checkpoint).resolve())
    # Reuse the original generic chart's four drawing colors/slots, with explicit
    # follow-up labels below. Underlying rows/summary keep their actual names.
    slots = {"same_probes_unit_bayes": "eig", "gain_bayes": "task_voi", "lsq": "lsq", "gain_mle": "tobit"}
    plotted = [dict(g, method=slots[g["method"]]) for g in summary["groups"] if g["method"] in slots]
    n = state["config"]["worlds"]
    small = "TINY PIPELINE CHECK; no headline conclusions. " if n < 32 else ""
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Physics Detective · Unknown actuator gain</title><style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 24px;background:#f7f5ef;color:#20282d}h1{font-size:36px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:16px}.card{padding:18px;background:white;border-radius:10px}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:7px;text-align:right;border-bottom:1px solid #ddd}td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;font-size:12px}</style>',
             '<h1>Can a detective learn its pusher is wrong?</h1>',
             f'<p>{small}{n} fresh paired worlds per family. The only model fix adds unknown actuator gain. Probe histories are reused unchanged; each probe still costs an attempt and a reset.</p>',
             '<p>Green: joint gain/friction Bayes. Blue: unit-gain Bayes on the same probes. Purple: equally validation-tuned joint MLE. Gold: original frozen LSQ.</p>',
             '<p>The gain-error family is now inside the adopted physics model. This is a test on fresh worlds, not a new unmodelled disturbance. MLE intervals shown below are diagnostic Bayesian intervals.</p>']
    for family in ("matched", "gain_shift"):
        guard.check()
        parts.append(f'<h2>{family}</h2><div class="grid">')
        for metric, label in (("success_rate", "Placement success"), ("posterior_coverage90", "90% friction coverage")):
            parts.append(f'<div class="card"><b>{label}</b>{chart(plotted, family, metric, label)}</div>')
        parts.append('</div>')
    parts.append('<p>Vertical lines are Wilson 95% intervals across worlds; budgets are paired. No abstention. Total attempts = probes + 1.</p><h2>All methods</h2><table><tr><th>Family / method / probes</th><th>Success</th><th>Attempts</th><th>μ RMSE</th><th>μ cov90</th><th>gain cov90</th></tr>')
    for g in summary["groups"]:
        mu_cov = f'{g["posterior_coverage90"]:.3f}' if "posterior_coverage90" in g else "—"
        gain_cov = f'{g["gain_coverage90"]:.3f}' if "gain_coverage90" in g else "—"
        parts.append(f'<tr><td>{g["family"]} / {g["method"]} / {g["budget"]}</td><td>{g["success_rate"]:.3f}</td><td>{g["mean_attempts"]:.0f}</td><td>{g["mu_rmse"]:.4f}</td><td>{mu_cov}</td><td>{gain_cov}</td></tr>')
    parts.append('</table>')
    by_key = {(r["world"], r["family"], r["method"], r["budget"]): r for r in state["rows"]}
    corrected = [r for r in state["rows"] if r["family"] == "gain_shift" and r["method"] == "gain_bayes" and r["budget"] == 4]
    saved = []
    recovered = [r for r in corrected if r["success"] and not by_key[r["world"], "gain_shift", "same_probes_unit_bayes", 4]["success"]]
    failures = [r for r in corrected if not r["success"]]
    if recovered:
        saved.append(("First recovered placement", recovered[0]))
    if failures:
        saved.append(("First remaining gain-aware failure", failures[0]))
    if not saved:
        saved.append(("First shifted world (no recovered placement or remaining failure)", corrected[0]))
    for label, selected in saved:
        parts.append(f'<h2>{label}: world {selected["world"]}</h2><p>Deterministic first-by-world selection for illustration. All worlds contribute to the scores above.</p><div class="grid">')
        for method in ("same_probes_unit_bayes", "gain_bayes", "gain_mle"):
            guard.check()
            r = by_key[selected["world"], "gain_shift", method, 4]
            parts.append(f'<div class="card"><b>{method}: {"HIT" if r["success"] else "MISS"}</b>{replay_svg(r)}'
                         f'<p>Probe owner: {r["probe_owner"]}. μ truth={r["mu_true"]:.4f}; estimate={r["mu_estimate"]:.4f}; gain truth={r["gain_true"]:.3f}.</p>'
                         f'<p>Predicted success={r["predicted_success"]:.3f}; stop={r["stop"]:.3f} m; target={r["target"]:.3f} m.</p>'
                         f'<details><summary>Saved evidence</summary><pre>{escape(json.dumps(r, indent=2))}</pre></details></div>')
        parts.append('</div>')
    parts.append(f'<h2>Provenance</h2><pre>{escape(json.dumps(state["config"], indent=2))}</pre></html>')
    atomic_json(summary_path, summary)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(parts))
    print(json.dumps(dict(status="complete", demo=output, summary=summary_path, worlds=n)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--max-minutes", type=float, required=True)
    args = p.parse_args()
    render(args.checkpoint, args.out, args.summary, args.max_minutes)
