"""Render measured R5 curves and a checkpoint-reloaded dream/reality demo, headlessly."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sim
from r5_core import Budget, digest, read_json, write_json, physics_rollout
from r5_soft_contact import soft_rollout

LABELS = dict(fitted="exact-law oracle (legacy key)", oracle_exact="exact-law oracle",
              soft_contact="fitted soft contact", reference_mean="turn-1 mean", reference_risk="turn-1 risk")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--prefix", required=True, help="media/r5_<run> (new prefix)")
    ap.add_argument("--max-minutes", type=float, required=True)
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    root, prefix = Path(a.eval), Path(a.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if list(prefix.parent.glob(prefix.name + "_*")):
        raise FileExistsError("media prefix exists; choose a fresh r5_* prefix")
    summary, selection = read_json(root / "summary.json"), read_json(a.selection)
    metric = summary["metrics"]
    methods = list(dict.fromkeys(k.split("/")[0] for k in metric))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, search in enumerate(("random", "cem")):
        for method in methods:
            keys = sorted([k for k in metric if k.startswith(f"{method}/{search}/")],
                          key=lambda x: int(x.rsplit("/", 1)[1]))
            xx = [int(k.rsplit("/", 1)[1]) for k in keys]
            for col, name in enumerate(("dream_success", "real_success", "fooled_rate")):
                axes[row, col].plot(xx, [metric[k][name] for k in keys], "o-", label=LABELS.get(method, method))
                axes[row, col].set(xlabel="shots" if search == "random" else "CEM iterations",
                                   ylabel=name.replace("_", " "), ylim=(-.02, 1.02), xscale="log")
        axes[row, 0].legend(fontsize=8)
    state = summary["status"]
    fig.suptitle(f"{state['completed_tasks']}/{state['requested_tasks']} paired tasks; completion={state['complete']}")
    fig.savefig(str(prefix) + "_pressure.png", dpi=130)
    plt.close(fig)
    fm = read_json(root / "forecast.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for method, horizons in fm["metrics"].items():
        hs = sorted(int(h) for h in horizons)
        for col, name in enumerate(("mean", "median")):
            axes[col].plot(hs, [horizons[str(h)]["all"][name] for h in hs], "o-", label=LABELS.get(method, method))
            axes[col].set(xlabel="horizon (frames)", ylabel=f"{name} position error")
    raw = np.load(root / "forecast_raw.npz")
    axes[2].scatter(raw["disagreement"][:, -1], raw["error"][:, -1], s=9, alpha=.5)
    axes[2].set(xlabel="ensemble disagreement", ylabel="real forecast error", title="Unselected independent scenes")
    axes[0].legend(fontsize=8)
    fig.savefig(str(prefix) + "_forecast.png", dpi=130)
    plt.close(fig)
    # Cost-normalized view makes ensemble overhead explicit.
    tasks = [read_json(p) for p in sorted((root / "tasks").glob("*.json"))]
    first = tasks[0]
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for method, search in (("hybrid_mean", "random"), ("hybrid_risk", "random"),
                           ("reference_mean", "random"), ("reference_risk", "random"),
                           ("walls", "random_member_matched"), ("fitted", "random_member_matched"),
                           ("oracle_exact", "random_member_matched"), ("soft_contact", "random_member_matched")):
        keys = sorted([k for k in metric if k.startswith(f"{method}/{search}/")],
                      key=lambda k: first["rows"][k]["member_frames"])
        if keys:
            ax.plot([first["rows"][k]["member_frames"] for k in keys],
                    [metric[k]["real_success"] for k in keys], "o-", label=LABELS.get(method, method))
    ax.set(xscale="log", xlabel="member × candidate × frame evaluations (not equal FLOPs)",
           ylabel="real success", ylim=(-.02, 1.02))
    ax.legend(fontsize=8)
    fig.savefig(str(prefix) + "_cost.png", dpi=130)
    plt.close(fig)

    budget.check()
    from r5_model import load_checkpoint, Predictor
    source = Path(selection["selected"]["checkpoint"])
    if digest(source / "model.safetensors") != selection["weights_sha256"]:
        raise ValueError("checkpoint hash changed")
    model, _ = load_checkpoint(source)
    predictor = Predictor(model)
    # First mean-model fooled case in task/effort order; otherwise worst gap.
    choices = [(t, key, row) for t in tasks for key, row in t["rows"].items()
               if key.startswith("hybrid_mean/random/") and row["dream"] is not None]
    fooled = [item for item in choices if item[2]["dream"] < sim.R <= item[2]["real"]]
    chosen = fooled[0] if fooled else max(choices, key=lambda item: item[2]["real"] - item[2]["dream"])
    task, key, row = chosen
    scene = np.array(task["scene"])[None]
    scene[0, 2:4] = row["velocity"]
    horizon = read_json(root / "manifest.json")["horizon"]
    imagined = predictor.rollout(scene, horizon, budget=budget).mean(axis=0)[0]
    real = physics_rollout(scene, horizon, budget=budget)[0]
    soft = selection.get("soft_contact")
    fitted = (soft_rollout(scene, horizon, soft["parameters"], budget=budget)[0] if soft
              else physics_rollout(scene, horizon, selection["parameters"], budget=budget)[0])
    classical_label = "FITTED SOFT same launch" if soft else "ORACLE same launch"
    replay_delta = float(np.linalg.norm(imagined[-1, :2] - row["predicted_position"]))
    if replay_delta > .003:
        raise RuntimeError(f"demo replay endpoint differs by {replay_delta}; inspect batch-size numerical sensitivity")
    np.savez_compressed(str(prefix) + "_demo.npz", imagined=imagined, real=real, fitted=fitted,
                        target=task["target"], initial=scene)
    frames, colors = [], ["#e6503c", "#3c8ce6", "#f0b428"]
    P, TOP, PAD = 220, 35, 12
    width, height = P + 2 * PAD, P + TOP + PAD
    for t in range(horizon + 1):
        budget.check()
        canvas = Image.new("RGB", (width * 3, height + 28), "#fafaf7")
        for col, (label, trajectory) in enumerate((("REAL chosen launch", real), ("ENSEMBLE MEAN dream", imagined), (classical_label, fitted))):
            tile = Image.new("RGB", (width, height), "#fafaf7")
            draw = ImageDraw.Draw(tile)
            draw.text((PAD, 8), label, fill="black")
            draw.rectangle((PAD, TOP, PAD + P, TOP + P), outline="black")
            tx, ty = task["target"]
            cx, cy, radius = PAD + P * tx, TOP + P * (1 - ty), P * sim.R
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), outline="#aa3344", width=2)
            for b, color in enumerate(colors):
                x, y = trajectory[t, 4 * b:4 * b + 2]
                if np.isfinite([x, y]).all() and -.2 < x < 1.2 and -.2 < y < 1.2:
                    cx, cy = PAD + P * x, TOP + P * (1 - y)
                    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color)
            canvas.paste(tile, (width * col, 0))
        ImageDraw.Draw(canvas).text((10, height + 6), f"task {task['task_id']} {key} frame {t}; selected {'fooled case' if fooled else 'largest gap (no fooled case)'}", fill="black")
        frames.append(np.array(canvas))
    imageio.mimsave(str(prefix) + "_demo.gif", frames, duration=40, loop=0)
    Image.fromarray(np.concatenate([frames[t] for t in (0, horizon // 2, horizon)], axis=0)).save(str(prefix) + "_strip.png")
    write_json(str(prefix) + "_demo.json", dict(task_id=task["task_id"], key=key, chosen=row,
                                               checkpoint=str(source), weights_sha256=selection["weights_sha256"],
                                               selection_rule="first fooled case, otherwise largest real-minus-dream gap",
                                               fooled_case=bool(fooled), replay_endpoint_delta=replay_delta,
                                               classical_panel="soft_contact" if soft else "exact_law_oracle",
                                               seconds=budget.elapsed, evaluation=str(root.resolve())))
    print(f"RENDER_OK prefix={prefix} replay_delta={replay_delta:.6g}", flush=True)


if __name__ == "__main__":
    main()
