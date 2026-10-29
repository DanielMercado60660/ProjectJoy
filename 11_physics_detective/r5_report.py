"""Render a complete R5 checkpoint into a standalone HTML exhibit (no MLX/network)."""
import argparse
from html import escape
import json
from pathlib import Path

import numpy as np

from physics import MU_RATIO, positions
from r5_core import Guard, ZONE_WIDTH
from run_r5 import ROOT, atomic_json, read_json, summarize


COLORS = dict(eig="#4269d0", task_voi="#00876c", lsq="#b07800", tobit="#9254b7")


def chart(groups, family, metric, label):
    width, height = 570, 230
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{label}: {family}">',
             '<path d="M45 15V190H545" fill="none" stroke="#889"/>']
    for v in (0, .5, 1):
        y = 190 - 170 * v
        parts.append(f'<path d="M45 {y}H545" stroke="#ddd"/><text x="12" y="{y+4}" font-size="12">{v:g}</text>')
    for b in (0, 1, 2, 4):
        x = 45 + 120 * b
        parts.append(f'<text x="{x}" y="209" text-anchor="middle" font-size="12">{b}</text>')
    if metric == "posterior_coverage90":
        parts.append('<path d="M45 37H545" stroke="#666" stroke-dasharray="5 4"/>')
    for method, color in COLORS.items():
        data = sorted([r for r in groups if r["family"] == family and r["method"] == method], key=lambda r: r["budget"])
        points = " ".join(f'{45+120*r["budget"]},{190-170*r[metric]}' for r in data)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for r in data:
            x, y = 45 + 120 * r["budget"], 190 - 170 * r[metric]
            interval = r["success_wilson95"] if metric == "success_rate" else r[metric + "_wilson95"]
            lo, hi = (190 - 170 * interval[1], 190 - 170 * interval[0])
            parts.append(f'<path d="M{x} {lo}V{hi}" stroke="{color}" opacity=".35"/>')
            parts.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
    parts.append('<text x="295" y="228" text-anchor="middle" font-size="12">preliminary probes (total attempts = probes + 1)</text></svg>')
    return "".join(parts)


def replay_svg(row):
    # True trajectory for display only, reconstructed from the saved command/world.
    f, duration = row["action"]
    mu, gain = row["mu_true"], row["gain_true"]
    t_stop = duration + max(gain * f - mu * 9.81, 0) * duration / (mu * 9.81)
    times = np.linspace(0, max(t_stop + .25, 1), 50)
    x = positions(f, duration, mu, MU_RATIO * mu, k=gain, times=times)
    extent = max(1.0, row["stop"] * 1.05)
    px = lambda v: 30 + 510 * v / extent
    target_left = px(row["target"] - ZONE_WIDTH / 2)
    target_width = 510 * ZONE_WIDTH / extent
    values = ";".join(f"{px(v):.2f}" for v in x)
    color = "#00876c" if row["success"] else "#c0392b"
    return (f'<svg viewBox="0 0 570 86" role="img" aria-label="Final push: {escape(row["method"])}">'
            f'<rect x="{target_left:.2f}" y="8" width="{target_width:.2f}" height="47" fill="#8bd9ad"/>'
            '<path d="M30 55H540" stroke="#889"/>'
            f'<circle cx="{px(row["stop"]):.2f}" cy="37" r="6" fill="{color}" opacity=".35"/>'
            f'<circle cx="30" cy="37" r="6" fill="{color}"><animate attributeName="cx" values="{values}" '
            'dur="4s" repeatCount="indefinite"/></circle>'
            f'<text x="30" y="78" font-size="12">0 m</text><text x="540" y="78" text-anchor="end" font-size="12">{extent:.2f} m</text></svg>')


def render(checkpoint, output, summary_path, max_minutes):
    guard = Guard(max_minutes)
    state = read_json(checkpoint)
    summary = summarize(state)
    summary["checkpoint"] = str(Path(checkpoint).resolve())
    atomic_json(summary_path, summary)
    rows = state["rows"]
    n = state["config"]["worlds"]
    small = "TINY PIPELINE CHECK — not a headline result. " if n < 32 else ""
    html = ['<!doctype html><html lang="en"><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>Physics Detective · Probe to place</title><style>',
            'body{font:16px system-ui,sans-serif;background:#f7f5ef;color:#20282d;max-width:1200px;margin:36px auto;padding:0 24px}',
            'h1{font-size:38px}h2{margin-top:34px}small{color:#59636b}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}',
            '.card{background:white;padding:18px;border:1px solid #e1dfd8;border-radius:10px}svg{width:100%;height:auto}',
            'table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:right;padding:7px;border-bottom:1px solid #ddd}',
            'td:first-child,th:first-child{text-align:left}pre{white-space:pre-wrap;font-size:12px}.tag{font-weight:650;margin-right:20px}',
            '</style><h1>The Physics Detective: probe to place</h1>',
            f'<p>{small}{n} paired worlds per family. Target band: 5 cm; one scored push; every probe costs one reset and one extra attempt.</p>',
            '<p>Blue: information gain. Green: task value of information. Gold: least squares. Purple: censoring-aware Tobit fit. All inference assumes unit actuator gain.</p>',
            '<p>The gain-shift family has a hidden ±6–12% actuator error. It was excluded from tuning. No neural network is used in this round.</p>']
    for family in ("matched", "gain_shift"):
        guard.check()
        html.append(f'<h2>{family}</h2><div class="grid">')
        for metric, label in (("success_rate", "Placement success"), ("posterior_coverage90", "90% posterior coverage")):
            html.append(f'<div class="card"><b>{label}</b>{chart(summary["groups"], family, metric, label)}</div>')
        html.append('</div>')
    html.append('<p><small>Vertical lines: Wilson 95% intervals across worlds. Budgets are correlated. LSQ/Tobit coverage is from a diagnostic Bayesian posterior on their probes; their controllers use point fits.</small></p>')
    html.append('<h2>Attempts, error and uncertainty</h2><table><tr><th>Family / method / probes</th><th>Success</th><th>Attempts</th><th>MAE cm</th><th>μ RMSE</th><th>90% width</th></tr>')
    for g in summary["groups"]:
        width = f'{g["posterior_mean_width90"]:.4f}' if "posterior_mean_width90" in g else "—"
        html.append(f'<tr><td>{g["family"]} / {g["method"]} / {g["budget"]}</td><td>{g["success_rate"]:.3f}</td><td>{g["mean_attempts"]:.0f}</td><td>{g["placement_mae_cm"]:.2f}</td><td>{g["mu_rmse"]:.4f}</td><td>{width}</td></tr>')
    html.append('</table>')
    failures = [r for r in rows if r["method"] == "task_voi" and r["family"] == "gain_shift" and r["budget"] == 4 and not r["success"]]
    example = failures[0] if failures else next(r for r in rows if r["family"] == "gain_shift" and r["method"] == "task_voi" and r["budget"] == 4)
    html.append(f'<h2>Saved consequence: world {example["world"]}</h2><p>Selection rule: first task-VOI failure at four probes in gain_shift; if none, the first such world. This is an illustration, not a representative success estimate.</p><div class="grid">')
    for r in rows:
        if r["world"] != example["world"] or r["family"] != "gain_shift" or r["budget"] != 4:
            continue
        guard.check()
        html.append(f'<div class="card"><b>{r["method"]}: {"HIT" if r["success"] else "MISS"}</b>{replay_svg(r)}'
                    f'<p>True μ={r["mu_true"]:.4f}; gain={r["gain_true"]:.3f}; target={r["target"]:.3f} m; stop={r["stop"]:.3f} m.</p>'
                    f'<p>Predicted success={r["predicted_success"]:.3f}; 90% interval=[{r["ci90"][0]:.4f}, {r["ci90"][1]:.4f}].</p>'
                    f'<details><summary>Saved probes and camera readings</summary><pre>{escape(json.dumps(r["history"], indent=2))}</pre></details></div>')
    html.append('</div><h2>Provenance</h2><p>No external assets or network calls. Rebuilt from the saved checkpoint, including the failure trajectory.</p>')
    html.append(f'<pre>{escape(json.dumps(dict(checkpoint=str(Path(checkpoint).resolve()), config=state["config"]), indent=2))}</pre></html>')
    path = Path(output)
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError("Demo must stay inside project")
    if path.exists():
        raise ValueError("Refusing to overwrite an existing demo; choose a fresh filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(html))
    print(json.dumps(dict(status="complete", demo=str(path), summary=str(summary_path), worlds=n)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument("--max-minutes", type=float, required=True)
    a = p.parse_args()
    render(a.checkpoint, a.out, a.summary, a.max_minutes)
