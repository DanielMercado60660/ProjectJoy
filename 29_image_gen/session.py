"""Work-queue driver run INSIDE one gpu_run slot: once a slot is held, keep it busy (sequentially) with the
next pending task until the session time cap (<=42 min) is used up. Several sessions can run at once;
per-task lock files stop two sessions from taking the same task. Training tasks are resumable.

    ../bin/gpu_run ../.venv/bin/python session.py --minutes 42
"""
import argparse, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PY = sys.executable
B = ["--budget_pflop", "2.5", "--budget_min", "25", "--batch", "64"]

# name, kind, command, deps, est_minutes (non-resumable tasks only start if this fits)
TASKS = [
    ("classifier", "once", [PY, "train_classifier.py", "--epochs", "20"], [], 8),
    ("flow", "train", ["flow", "runs/flow"] + B, [], 0),
    ("ddpm", "train", ["ddpm", "runs/ddpm"] + B, [], 0),
    ("gan", "train", ["gan", "runs/gan"] + B, [], 0),
    ("vae", "train", ["vae", "runs/vae"] + B, [], 0),
    ("eval_main", "once", [PY, "evaluate.py", "main", "runs/vae", "runs/gan", "runs/ddpm", "runs/flow"],
     ["classifier", "flow", "ddpm", "gan", "vae"], 12),
    ("flow_cond", "train", ["flow", "runs/flow_cond", "--cond", "--budget_pflop", "2.5", "--budget_min", "40", "--batch", "64"], [], 0),
    ("eval_sweep", "once", [PY, "evaluate.py", "sweep", "runs/flow", "runs/ddpm", "runs/vae", "runs/gan"],
     ["eval_main"], 15),
    ("eval_stab", "once", [PY, "evaluate.py", "stability", "runs/vae", "runs/gan", "runs/ddpm", "runs/flow"],
     ["eval_main"], 10),
    ("flow_x2", "train", ["flow", "runs/flow_x2"], [], 0),
    ("ddpm_x2", "train", ["ddpm", "runs/ddpm_x2"], [], 0),
    ("gan_x2", "train", ["gan", "runs/gan_x2"], [], 0),
    ("vae_x2", "train", ["vae", "runs/vae_x2"], [], 0),
    ("eval_x2", "once", [PY, "evaluate.py", "main_x2", "runs/vae_x2", "runs/gan_x2", "runs/ddpm_x2", "runs/flow_x2"],
     ["flow_x2", "ddpm_x2", "gan_x2", "vae_x2"], 12),
    ("eval_cond", "once", [PY, "eval_cond.py", "runs/flow_cond"], ["flow_cond", "classifier", "eval_main"], 10),
]


def done(name):
    t = [t for t in TASKS if t[0] == name][0]
    if t[1] == "train":
        p = os.path.join(t[2][1], "latest", "meta.json")
        return os.path.exists(p) and json.load(open(p))["done"]
    return os.path.exists(f"logs/.done_{name}")


def try_lock(name):
    try:
        fd = os.open(f"logs/.lock_{name}", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        # stale lock?
        try:
            pid = int(open(f"logs/.lock_{name}").read())
            os.kill(pid, 0)
            return False
        except (ProcessLookupError, ValueError):
            os.remove(f"logs/.lock_{name}")
            return try_lock(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=42)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    t0 = time.time()
    only = set(a.only.split(",")) if a.only else None
    failed = set()
    while True:
        left = a.minutes - (time.time() - t0) / 60
        if left < 3:
            break
        ran = False
        for name, kind, cmd, deps, est in TASKS:
            if (only and name not in only) or name in failed or done(name) or not all(done(d) for d in deps):
                continue
            if kind == "once" and est > left:
                continue
            if kind == "train" and time.strftime("%H%M") >= "1045":  # no new training after 10:45 (deadline 12:30)
                continue
            if not try_lock(name):
                continue
            try:
                log = open(f"logs/{name}.log", "a")
                print(f"[session {os.getpid()}] {time.strftime('%H:%M')} start {name} ({left:.0f} min left)", flush=True)
                if kind == "train":
                    model, run = cmd[0], cmd[1]
                    resume = os.path.exists(os.path.join(run, "latest", "meta.json"))
                    c = [PY, "train_gen.py", "--model", model, "--run", run, "--session_min", f"{left - 1.5:.1f}"]
                    c += ["--resume"] if resume else cmd[2:]
                else:
                    c = cmd
                log.write(f"\n=== {time.strftime('%H:%M:%S')} {' '.join(c)}\n"); log.flush()
                rc = subprocess.call(c, stdout=log, stderr=subprocess.STDOUT)
                if rc != 0:
                    failed.add(name)
                if kind == "once" and rc == 0:
                    open(f"logs/.done_{name}", "w").write(time.strftime("%H:%M"))
                print(f"[session {os.getpid()}] {time.strftime('%H:%M')} end {name} rc={rc} done={done(name)}", flush=True)
            finally:
                os.remove(f"logs/.lock_{name}")
            ran = True
            break
        if not ran:
            break
    print(f"[session {os.getpid()}] exit; pending: {[t[0] for t in TASKS if not done(t[0])]}", flush=True)


if __name__ == "__main__":
    main()
