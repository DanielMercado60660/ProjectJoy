"""Bounded smoke driver: actual save/SIGKILL/resume, then all evaluation paths."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    start = time.monotonic()
    run = "smoke_"+time.strftime("%Y%m%d_%H%M%S")
    out = Path("results/r5")/run
    out.mkdir(parents=True)
    common = ["--run", run, "--holdout", "6", "--min-train", "24", "--batch", "8",
              "--low", "2", "--fixed", "4", "--high", "12", "--steps-per-round", "4", "--smoke"]
    def execute(argv):
        subprocess.run([sys.executable]+argv, check=True, timeout=max(1, 100-(time.monotonic()-start)))
    execute(["-m", "pytest", "-q", "tests/test_r5.py"])
    execute(["r5_verify_resume.py", f"checkpoints/r5_{run}/resume_check"])
    cmd = [sys.executable, "r5_run.py", "train"]+common+["--max-minutes", ".75", "--data-minutes", ".4"]
    with open(out/"before_kill.log", "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            marker = out/"checkpoint_ready.json"
            while not marker.exists():
                if process.poll() is not None:
                    raise RuntimeError(f"Smoke train exited {process.returncode}; see {out}/before_kill.log")
                if time.monotonic()-start > 65:
                    raise TimeoutError("Smoke checkpoint not reached within 65s; inspect logs before main run")
                time.sleep(.05)
            before = json.loads(marker.read_text())["steps"]
            os.kill(process.pid, signal.SIGKILL)
            process.wait()
        finally:
            if process.poll() is None:
                process.kill(); process.wait()
    execute(["r5_run.py", "train"]+common+["--max-minutes", ".15"])
    after = json.loads((out/"checkpoint_ready.json").read_text())["steps"]
    assert all(b > a for a, b in zip(before, after)), (before, after)
    execute(["r5_run.py", "evaluate"]+common+["--max-minutes", ".25"])
    report = json.loads((out/"evaluation.json").read_text())
    assert report["complete"] and len(report["summary"]) == 14
    assert json.loads((out/"secondary.json").read_text())["complete"]
    assert len(json.loads((out/"games.json").read_text())["records"]) >= 2
    assert (Path("media")/f"r5_{run}.html").exists()
    result = dict(smoke_pass=True, run=run, seconds=time.monotonic()-start,
                  killed_after_steps=before, resumed_to_steps=after,
                  note="Timing equality is not asserted on tiny smoke blocks; main requires reported ±10% match.")
    (out/"smoke.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
