"""Bounded subprocess smoke; retain every log and artifact. Run only on Claude's GPU host."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from r5_core import Budget, read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-minutes", type=float, default=1.9)
    a = ap.parse_args()
    budget = Budget(a.max_minutes)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_{os.getpid()}"
    root = Path("results/r5") / f"smoke_{run_id}"
    logs = Path("logs/r5") / f"smoke_{run_id}"
    ckpt = Path("checkpoints") / f"r5_smoke_{run_id}"
    root.mkdir(parents=True)
    logs.mkdir(parents=True)
    commands = []
    py = sys.executable

    def run(label, args, kill_at_checkpoint=False):
        budget.check()
        command = [py] + [str(arg) for arg in args]
        commands.append(dict(label=label, argv=command))
        print(f"SMOKE {label}", flush=True)
        path = logs / f"{label}.log"
        remaining = budget.seconds - budget.elapsed
        with path.open("w") as f:
            process = subprocess.Popen(command, stdout=subprocess.PIPE if kill_at_checkpoint else f,
                                       stderr=subprocess.STDOUT, start_new_session=True, text=True, bufsize=1)
            try:
                if kill_at_checkpoint:
                    selector = selectors.DefaultSelector()
                    selector.register(process.stdout, selectors.EVENT_READ)
                    killed = False
                    while process.poll() is None:
                        budget.check()
                        if selector.select(timeout=min(.2, budget.seconds - budget.elapsed)):
                            line = process.stdout.readline()
                            f.write(line)
                            f.flush()
                            if line.startswith("CHECKPOINT step=2 "):
                                os.killpg(process.pid, signal.SIGKILL)
                                killed = True
                                break
                    process.wait(timeout=5)
                    selector.close()
                    if not killed:
                        raise RuntimeError(f"trainer never reached kill checkpoint; inspect {path}")
                else:
                    process.wait(timeout=remaining)
                    if process.returncode:
                        raise RuntimeError(f"{label} failed ({process.returncode}); inspect {path}")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
                raise
        budget.check()

    try:
        run("numpy", ["-m", "pytest", "-q", "test_r5_numpy.py", "test_r5_turn2_numpy.py", "test_r5_turn3_numpy.py", f"--basetemp={root / 'pytest'}",
                      "-o", f"cache_dir={root / 'pytest_cache'}"])
        run("prepare", ["experiment_r5.py", "prepare", "--out", root / "data", "--scenes", 64,
                        "--val-scenes", 24, "--frames", 8, "--max-minutes", .2])
        run("fit", ["experiment_r5.py", "fit", "--data", root / "data", "--out", root / "fit.json",
                    "--samples", 64, "--candidates", 20, "--refine", 1, "--horizon", 2, "--max-minutes", .2])
        run("soft_fit", ["r5_soft_contact.py", "--data", root / "data", "--out", root / "soft_fit.json",
                         "--proposals", 8, "--refine", 2, "--shortlist", 5, "--samples", 32,
                         "--val-scenes", 8, "--fit-horizon", 2, "--horizon", 3, "--max-minutes", .2])
        train = ["train_r5.py", "--data", root / "data", "--members", 5, "--hidden", 8,
                 "--batch", 8, "--unroll", 2, "--max-minutes", .3]
        run("raw_training", train + ["--out", ckpt / "raw", "--steps", 3])
        train += ["--features", "geometry", "--depth", 3, "--collision-weight", 3,
                  "--contact-fraction", .75, "--decay-steps", 10]
        run("killed", train + ["--out", ckpt / "resumed", "--steps", 100000, "--checkpoint-every", 2], True)
        generation = read_json(ckpt / "resumed/latest.json")["generation"]
        saved_step = read_json(ckpt / "resumed" / generation / "state.json")["step"]
        assert saved_step >= 2
        target = saved_step + 3
        run("resumed", train + ["--out", ckpt / "resumed", "--resume", "--steps", target, "--checkpoint-every", 100])
        run("reference", train + ["--out", ckpt / "reference", "--steps", target, "--checkpoint-every", 100])
        checks = ["check_mlx_r5.py", "--resumed", ckpt / "resumed", "--reference", ckpt / "reference"]
        if Path("results/r5/r5_20261024T152040Z/selection.json").exists():
            checks += ["--legacy-selection", "results/r5/r5_20261024T152040Z/selection.json",
                       "--legacy-demo", "media/r5_20261024T152040Z_demo.npz"]
        run("mlx_checks", checks)
        run("tune_reference", ["experiment_r5.py", "tune", "--data", root / "data", "--fit", root / "fit.json",
                               "--checkpoints", ckpt / "raw", "--out", root / "reference_selection.json",
                               "--tasks", 2, "--shots", 8, "--horizon", 3, "--betas", 0, 1, "--max-minutes", .3])
        run("tune", ["experiment_r5.py", "tune", "--data", root / "data", "--fit", root / "fit.json",
                     "--soft-fit", root / "soft_fit.json", "--reference-selection", root / "reference_selection.json",
                     "--checkpoints", ckpt / "resumed", ckpt / "reference", "--out", root / "selection.json",
                     "--tasks", 2, "--forecast-val-scenes", 4, "--tune-seed", 53003,
                     "--shots", 8, "--horizon", 3, "--betas", 0, .5, 1, "--max-minutes", .3])
        evaluation = ["experiment_r5.py", "eval", "--selection", root / "selection.json", "--out", root / "eval",
                      "--tasks", 2, "--forecast-scenes", 4, "--horizon", 3, "--shots", 4, 8,
                      "--cem-iters", 1, 2, "--population", 4, "--matched-cost", "--max-minutes", .3,
                      "--turn2", "--seed", 61031, "--action-seed", 61032]
        run("eval", evaluation)
        run("eval_resume", evaluation)
        run("classical_tune", ["r5_classical_risk.py", "tune", "--selection", root / "selection.json",
                               "--out", root / "classical_selection.json", "--shots", 8, "--max-minutes", .3])
        classical_eval = ["r5_classical_risk.py", "eval", "--selection", root / "selection.json",
                          "--classical-selection", root / "classical_selection.json", "--out", root / "classical_eval",
                          "--tasks", 2, "--shots", 4, 8, "--seed", 74031, "--action-seed", 74032,
                          "--target-seed", 74131, "--chunk", 8, "--max-minutes", .3]
        run("classical_eval", classical_eval)
        run("classical_eval_resume", classical_eval)
        run("legacy", [*[arg for arg in evaluation if arg != "--turn2"], "--out", root / "legacy", "--legacy-data", root / "data/val.npz",
                       "--label", "smoke_legacy_path"])
        run("render", ["render_r5.py", "--eval", root / "eval", "--selection", root / "selection.json",
                       "--prefix", f"media/r5_smoke_{run_id}", "--max-minutes", .3])
        summary = read_json(root / "eval/summary.json")
        assert summary["status"]["complete"] and len(summary["task_ids"]) == 2
        assert len(summary["metrics"]) == 34  # 7 methods x 4 cells + 3 classical x 2 cost controls
        assert read_json(root / "legacy/status.json")["complete"]
        classical_summary = read_json(root / "classical_eval/summary.json")
        assert classical_summary["status"]["complete"] and len(classical_summary["metrics"]) == 12
        assert classical_summary["holm_family_size"] == 30
        write_json(root / "smoke.json", dict(ok=True, seconds=budget.elapsed, commands=commands, logs=str(logs),
                                             interrupted_step=saved_step, resumed_step=target))
        print(f"SMOKE_R5_OK seconds={budget.elapsed:.1f} receipts={root} logs={logs}", flush=True)
    except BaseException as exc:
        write_json(root / "smoke.json", dict(ok=False, seconds=budget.elapsed, commands=commands,
                                             logs=str(logs), error=str(exc)))
        raise


if __name__ == "__main__":
    main()
