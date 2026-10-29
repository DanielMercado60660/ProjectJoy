"""Headless deterministic replay from a selected checkpoint; retain failures."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from env_r5 import make_pool, GROUPS, DR, DC
from eval_r5 import run_budget
from io_r5 import Deadline, write_json, file_digest


def render(pool, cave, trajectory, episode, output, deadline):
    size = int(pool.sizes[cave]); cell = 46; margin = 30
    images = []
    # Every second step is shown; include the terminal frame exactly.
    indices = list(range(0, episode["steps"], 2))
    if episode["steps"]-1 not in indices:
        indices.append(episode["steps"]-1)
    trail = []
    for k in indices:
        deadline.check()
        r, c, hd, t, pulses, used, bump, action, finished = trajectory[k, cave]
        image = Image.new("RGB", (max(450, size*cell+2*margin), size*cell+160), "#101522")
        draw = ImageDraw.Draw(image)
        draw.text((20, 12), f"BAT SCHOOL R5 | {GROUPS[pool.group[cave]]} | cave {cave}", fill="#f3e8c7")
        for rr in range(size):
            for cc in range(size):
                x, y = margin+cc*cell, 45+rr*cell
                for d in (0, 1, 2, 3):
                    if d == 0: line = (x, y, x+cell, y)
                    elif d == 1: line = (x+cell, y, x+cell, y+cell)
                    elif d == 2: line = (x, y+cell, x+cell, y+cell)
                    else: line = (x, y, x, y+cell)
                    if not pool.open[cave, rr, cc, d]:
                        draw.line(line, fill="#64748b", width=3)
                    elif pool.closed[cave, rr, cc, d] > 0:
                        shut = ((t+pool.phase[cave, rr, cc, d]) % pool.period[cave, rr, cc, d]) < pool.closed[cave, rr, cc, d]
                        draw.line(line, fill="#fb7185" if shut else "#2b414c", width=5 if shut else 1)
        goal_r, goal_c = pool.goal[cave]
        gx, gy = margin+(goal_c+.5)*cell, 45+(goal_r+.5)*cell
        draw.ellipse((gx-7, gy-7, gx+7, gy+7), fill="#fde68a")
        x, y = margin+(c+.5)*cell, 45+(r+.5)*cell
        trail.append((x, y))
        if len(trail) > 1:
            draw.line(trail, fill="#365a72", width=2)
        if used:
            draw.ellipse((x-18, y-18, x+18, y+18), outline="#a7f3d0", width=2)
        draw.ellipse((x-8, y-8, x+8, y+8), fill="#fb7185" if bump else "#67e8f9")
        draw.line((x, y, x+15*DC[hd], y+15*DR[hd]), fill="white", width=2)
        label = ["forward", "turn left", "turn right", "wait", "backward"][action]
        draw.text((20, size*cell+60), f"step {t}/320  pulses {pulses}/{episode['budget']}  action: {label}", fill="white")
        draw.text((20, size*cell+80), "pink: closed shutter/bump   ring: charged echo", fill="#cbd5e1")
        draw.text((20, size*cell+100), f"Result: {'completed' if episode['success'] else 'timeout'}; collisions {episode['collisions']}", fill="#f3e8c7")
        draw.text((20, size*cell+120), "True map shown to viewer only. Agent receives echoes.", fill="#cbd5e1")
        images.append(image)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(output, save_all=True, append_images=images[1:], duration=140, loop=0, optimize=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-group", type=int, default=128)
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--prefix", default="media/r5_main")
    ap.add_argument("--max-minutes", type=float, default=.5)
    a = ap.parse_args()
    deadline = Deadline(a.max_minutes)
    if Path(a.prefix+".json").exists():
        raise FileExistsError(a.prefix)
    from policy_r5 import load_policy, LearnedAgent
    model, saved = load_policy(a.ckpt)
    pool = make_pool("test", a.per_group)
    episodes, frames = run_budget(pool, lambda n: LearnedAgent(model, n), a.budget, deadline, trace=True)
    picks = []
    # Predeclared: first hard-group failure and first hard-group success.
    # If one category is absent, say so; never search another seed for a win.
    absent = []
    for success, name in ((False, "failure"), (True, "success")):
        eligible = [e for e in episodes if e["group"] == "hard" and e["success"] == success]
        if not eligible:
            absent.append(name); continue
        ep = eligible[0]
        path = a.prefix+"_"+name+".gif"
        render(pool, ep["cave"], frames, ep, path, deadline)
        picks.append(dict(file=path, episode=ep))
    write_json(a.prefix+".json", dict(checkpoint=a.ckpt, checkpoint_sha256=file_digest(a.ckpt), selected_update=saved["update"],
                                    manifest=pool.manifest(), clips=picks, absent=absent, seconds=deadline.elapsed))
    print(f"DEMO OK: {len(picks)} clips; absent categories {absent}")


if __name__ == "__main__":
    main()
