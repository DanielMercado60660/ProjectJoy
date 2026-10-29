"""GPU smoke component: save one episode, kill the next, resume and verify custody.

Run only via smoke_r5.sh / gpu_run. Kills only the process group it creates.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from r5_core import atomic_json, digest


def main():
    pilot = json.loads(Path("results/r5/pilot.json").read_text())
    name = f"smoke_{os.getpid()}"
    planpath = Path(f"logs/r5/{name}_plan.json")
    plan = dict(sources=pilot["sources"], checkpoint=pilot["checkpoint"], rounds=1,
                mode="smoke only, never headline", jobs=[
                    dict(event="tracker", engine="numpy_cpu", policy="queue", load="ambient",
                         delay_ms=0.0, duration=0.15, seed=850041, split="fresh", block=0),
                    dict(event="tracker", engine="mlx_gpu", policy="latest", load="loaded",
                         delay_ms=50.0, duration=0.6, seed=850042, split="fresh", block=0)])
    atomic_json(planpath, plan)
    cmd = [sys.executable, "r5_run.py", "run", "--plan", str(planpath), "--name", name,
           "--max-minutes", "0.2"]
    # Explicit own session so a forced kill includes the background load process.
    child = subprocess.Popen(cmd+["--child"], start_new_session=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True)
    def expire():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    timer = threading.Timer(12, expire)
    timer.start()
    lines = []
    killed = False
    try:
        for line in child.stdout:
            lines.append(line)
            print(line, end="", flush=True)
            if line.startswith("START episode_00001"):
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
                killed = True
                break
        if not killed:
            raise RuntimeError("Probe never reached interrupted episode")
    finally:
        timer.cancel()
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    receipt = Path(f"results/r5/{name}/episode_00000.json")
    before = digest(receipt)
    state = json.loads(Path(f"checkpoints/r5_{name}_state.json").read_text())
    assert state["step"] == 1, state
    subprocess.run(cmd+["--resume"], check=True, timeout=16)
    assert digest(receipt) == before, "Resume rewrote a completed receipt"
    state = json.loads(Path(f"checkpoints/r5_{name}_state.json").read_text())
    assert state["step"] == 2
    status = json.loads(Path(f"results/r5/{name}/run_status.json").read_text())
    assert status["complete"]
    atomic_json("results/r5/resume_probe.json", dict(ok=True, name=name, killed=True,
                                                   completed_receipt_unchanged=True, step=state["step"],
                                                   time=time.strftime("%Y-%m-%dT%H:%M:%S%z")))
    print("RESUME OK: save / kill / restart; completed receipt byte-identical", flush=True)
    subprocess.run([sys.executable, "r5_report.py", "--run", name,
                    "--out", f"media/r5_{name}.html"], check=True)


if __name__ == "__main__":
    main()
