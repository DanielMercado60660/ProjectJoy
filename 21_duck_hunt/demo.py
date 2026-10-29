"""Demo: the learned flyer (PPO checkpoint) flies a HELD-OUT course; the flight is rendered into
three scenes (training-like, held-out background, held-out everything) and both trackers run on
the frames. Overlay: green dot = true duck, red ring = CNN+Kalman, blue square = classical.
  ../.venv/bin/python demo.py  -> media/duck_hunt_demo.gif, media/duck_hunt_demo_strip.png
"""
import json, sys
import numpy as np
from PIL import Image, ImageDraw
import flight as F
import scene as S
import train_flight as TF
import track_classical as TC
import eval_tracker as ET

UP = 3


def main(course_idx=3, out="media/duck_hunt_demo.gif"):
    pol = TF.policy_fn(TF.load("checkpoints/flight_ppo"))
    C = F.heldout_courses(500)[course_idx:course_idx + 1]
    r = F.run_episodes(C, pol, seed=99, record=True)
    tr = r["traj"][0]; tr = tr[~np.isnan(tr[:, 0])]
    nf = min(len(tr) // S.FRAME_EVERY, 60)
    pts = S.world_to_img(tr[np.minimum(np.arange(nf + 1) * S.FRAME_EVERY, len(tr) - 1)])
    print(f"held-out course {course_idx}: complete={r['complete'][0]} t={r['t'][0]}s, {nf} frames")
    net = ET.load_net()
    cprm = json.load(open("checkpoints/classical_tracker_params.json"))
    kprm = json.load(open("checkpoints/cnn_kf_params.json"))
    panels = []
    for cond, seed, label in [("iid", 91_001, "train-like scene"), ("bg", 91_002, "held-out background"),
                              ("all", 91_003, "held-out bg+light+distractors")]:
        fr, gt, vis = S.render_scene(seed, cond, nf, pts=pts)
        est_c = TC.track_scene(TC.precompute(fr), cprm)
        est_k = ET.cnn_kf_track(ET.net_peaks(net, fr), kprm)
        ims = []
        for f in range(nf):
            im = Image.fromarray(fr[f]).resize((S.W * UP, S.H * UP), Image.NEAREST)
            d = ImageDraw.Draw(im)
            g = (gt[f] + 0.5) * UP; k = (est_k[f] + 0.5) * UP; c = (est_c[f] + 0.5) * UP
            d.rectangle([c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8], outline=(40, 90, 255), width=2)
            d.ellipse([k[0] - 10, k[1] - 10, k[0] + 10, k[1] + 10], outline=(255, 30, 30), width=2)
            d.ellipse([g[0] - 2, g[1] - 2, g[0] + 2, g[1] + 2], fill=(0, 255, 0))
            ek, ec = np.linalg.norm(est_k[f] - gt[f]), np.linalg.norm(est_c[f] - gt[f])
            d.rectangle([0, 0, S.W * UP, 24], fill=(0, 0, 0))
            d.text((4, 2), label, fill=(255, 255, 255))
            d.text((4, 12), f"CNN+KF err {ek:4.1f}px  classical {ec:4.1f}px", fill=(255, 255, 255))
            ims.append(np.array(im))
        panels.append(np.stack(ims))
    frames = np.concatenate(panels, axis=2)
    frames = [Image.fromarray(f) for f in frames]
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=120, loop=0)
    strip = np.concatenate([np.concatenate([p[i] for p in panels], 0) for i in range(0, nf, max(nf // 5, 1))][:5], 1)
    Image.fromarray(strip).save("media/duck_hunt_demo_strip.png")
    print("wrote", out)


def failure_strips(tag=""):
    """Failure clips from saved eval tracks (results/tracks_<cond>.npz): for each held-out
    condition, the scene where the CNN's lost rate is highest, 6 frames, both trackers overlaid."""
    sfx = f"_{tag}" if tag else ""
    rows = []
    for cond in ["bg", "light", "distract"]:
        T = np.load(f"results/tracks{sfx}_{cond}.npz")
        err = np.linalg.norm(T["cnn_raw"] - T["gt"], axis=-1)
        i = int(np.argmax((err > TC.LOST_PX).mean(1)))
        fr, gt, vis = S.render_scene(S.COND_SEED[cond] + i, cond)
        ims = []
        for f in range(0, 36, 6):
            im = Image.fromarray(fr[f]).resize((S.W * UP, S.H * UP), Image.NEAREST); d = ImageDraw.Draw(im)
            g = (gt[f] + 0.5) * UP; k = (T["cnn_raw"][i, f] + 0.5) * UP; c = (T["classical"][i, f] + 0.5) * UP
            d.rectangle([c[0] - 8, c[1] - 8, c[0] + 8, c[1] + 8], outline=(40, 90, 255), width=2)
            d.ellipse([k[0] - 10, k[1] - 10, k[0] + 10, k[1] + 10], outline=(255, 30, 30), width=2)
            d.ellipse([g[0] - 2, g[1] - 2, g[0] + 2, g[1] + 2], fill=(0, 255, 0))
            d.rectangle([0, 0, S.W * UP, 12], fill=(0, 0, 0)); d.text((3, 1), f"{cond} scene {i} frame {f}", fill=(255, 255, 255))
            ims.append(np.array(im))
        rows.append(np.concatenate(ims, 1))
    Image.fromarray(np.concatenate(rows, 0)).save(f"media/failures{sfx}.png")
    print("wrote", f"media/failures{sfx}.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--failures":
        failure_strips(sys.argv[2] if len(sys.argv) > 2 else "")
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
