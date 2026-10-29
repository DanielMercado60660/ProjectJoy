"""Render demos from saved checkpoints.

  media/grow_damage_regrow.gif   side by side: growth-only baseline vs regeneration-trained, same seed and same damage
  media/filmstrip_<damage>.png   undamaged | damaged | +25 | +50 | +100 | +200 | +400 steps, both models
  media/failures/*.gif           automatically picked worst cases (explosions, ghost limbs, oscillation)
  media/demo.html                interactive erase-and-regrow page (weights inlined)

Usage: ../.venv/bin/python make_demo.py [--models lizard_persist lizard_regen] [--target lizard]
"""
import argparse, json, os
import numpy as np
import mlx.core as mx
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

import nca
import damage

SCALE = 4
FONT = None


def font(size=14):
    for p in ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def tile(x, scale=SCALE):
    """One state (H,W,C) -> uint8 RGB image upscaled with nearest neighbour."""
    rgb = nca.to_rgb(np.nan_to_num(np.clip(x, -10, 10)))
    im = Image.fromarray((rgb * 255).astype(np.uint8))
    return im.resize((im.width * scale, im.height * scale), Image.NEAREST)


def labelled_row(tiles, labels, banner=None, bh=22):
    w, h = tiles[0].width, tiles[0].height
    top = bh * (2 if banner else 1)
    canvas = Image.new("RGB", (w * len(tiles), h + top), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    f = font(14)
    if banner:
        d.rectangle([0, 0, canvas.width, bh], fill=(30, 30, 30))
        d.text((6, 3), banner, fill=(255, 255, 255), font=f)
    for i, (t, l) in enumerate(zip(tiles, labels)):
        canvas.paste(t, (i * w, top))
        d.text((i * w + 6, top - bh + 3), l, fill=(0, 0, 0), font=f)
    return canvas


def load(names):
    return {n: nca.load_model(os.path.join("checkpoints", n, "model.safetensors")) for n in names}


def pretty(n):
    return {"persist": "growth-only (baseline)", "regen": "regeneration-trained", "grow": "no pool (grow only)"}.get(n.split("_")[-1], n)


def side_by_side_gif(models, target, path, seed=7):
    """Grow 200 -> half cut (held-out) -> 250 -> blobs (held-out) -> 250 -> circle (train) -> 250."""
    H, W = target.shape[:2]
    names = list(models)
    rng = np.random.default_rng(seed)
    left = np.zeros((H, W), bool)
    left[:, : W // 2] = True  # remove the left half of the grid
    schedule = {200: ("left half (held-out)", left),
                450: ("random blobs (held-out)", damage.blobs(H, W, rng)),
                700: ("circle (training-type)", damage.circle(H, W, np.random.default_rng(3)))}
    total = 950
    xs = {n: mx.array(nca.make_seed(H, W)) for n in names}
    frames, banner = [], "growing from a single seed cell"
    for t in range(total + 1):
        if t in schedule:
            label, mask = schedule[t]
            banner = f"step {t}: damage = {label}"
            for n in names:
                xs[n] = mx.array(damage.apply(np.array(xs[n]), mask))
            for _ in range(8):  # hold the damaged frame briefly
                frames.append(labelled_row([tile(np.array(xs[n])[0]) for n in names], [pretty(n) for n in names],
                                           banner + "   (step %d)" % t))
        if t % 3 == 0:
            frames.append(labelled_row([tile(np.array(xs[n])[0]) for n in names], [pretty(n) for n in names],
                                       banner + "   (step %d)" % t))
        for n in names:
            mx.random.seed(seed * 100000 + t)  # identical fire masks for both models at each step
            xs[n] = models[n](xs[n])
        mx.eval(*xs.values())
    imageio.mimsave(path, [np.asarray(f) for f in frames], duration=0.04, loop=0)
    try:
        imageio.mimsave(path.replace(".gif", ".mp4"), [np.asarray(f) for f in frames], fps=25, macro_block_size=1)
    except Exception as e:
        print("mp4 skipped:", e)
    print("wrote", path, len(frames), "frames")


def repeated_gif(models, target, path, seed=77, rounds=10, every=150):
    """Grow 200 steps, then a random held-out wound every 150 steps (same wounds for both models)."""
    H, W = target.shape[:2]
    names = list(models)
    rng = np.random.default_rng(seed)
    fns = list(damage.HELDOUT_DAMAGE.items())
    wounds = {}
    for r in range(rounds):
        name, f = fns[rng.integers(len(fns))]
        wounds[200 + r * every] = (name, f(H, W, rng))
    xs = {n: mx.array(nca.make_seed(H, W)) for n in names}
    frames, banner = [], "growing"
    for t in range(200 + rounds * every + 1):
        if t in wounds:
            banner = f"wound {(t - 200) // every + 1}/{rounds}: {wounds[t][0]}"
            for n in names:
                xs[n] = mx.array(damage.apply(np.array(xs[n]), wounds[t][1]))
        if t % 5 == 0 or t in wounds:
            row = []
            for n in names:
                x = np.array(xs[n])
                row.append((tile(x[0]), f"{pretty(n)}  mse {float(((np.clip(x[0, ..., :4], -1e3, 1e3) - target) ** 2).mean()):.3f}"))
            frames.append(np.asarray(labelled_row([r[0] for r in row], [r[1] for r in row], f"{banner}   (step {t})")))
        for n in names:
            mx.random.seed(seed * 100000 + t)
            xs[n] = models[n](xs[n])
        mx.eval(*xs.values())
    imageio.mimsave(path, frames, duration=0.04, loop=0)
    print("wrote", path, len(frames), "frames")


def birth_grid(models, target, path, K=16, seed=5):
    """16 independent seeds per model after 200 steps: shows how often growth from a single cell fails."""
    H, W = target.shape[:2]
    blank = float((target ** 2).mean())
    panels, labels = [], []
    for n, model in models.items():
        mx.random.seed(seed)
        x = np.array(model.rollout(mx.array(nca.make_seed(H, W, K)), 200))
        err = ((x[..., :4] - target) ** 2).mean(axis=(1, 2, 3))
        tiles = [np.asarray(tile(x[k], 2)) for k in range(K)]
        grid = np.concatenate([np.concatenate(tiles[r * 4:(r + 1) * 4], 1) for r in range(K // 4)], 0)
        panels.append(Image.fromarray(grid))
        labels.append(f"{n}: {int((err < 0.5 * blank).sum())}/{K} grew")
    img = labelled_row(panels, labels, banner="16 seeds each, step 200")
    img.save(path)
    print("wrote", path, labels)


def filmstrips(models, target, seed=11):
    H, W = target.shape[:2]
    marks = [25, 50, 100, 200, 400]
    for i, d in enumerate(damage.ALL_DAMAGE):
        rows = []
        mask = damage.ALL_DAMAGE[d](H, W, np.random.default_rng(seed + i))
        for n, model in models.items():
            mx.random.seed(seed)
            x = model.rollout(mx.array(nca.make_seed(H, W)), 200)
            pre = np.array(x)[0]
            x = mx.array(damage.apply(np.array(x), mask))
            tiles, labels = [tile(pre), tile(np.array(x)[0])], ["grown", "damaged"]
            t = 0
            for mk in marks:
                x = model.rollout(x, mk - t); t = mk
                mx.eval(x)
                tiles.append(tile(np.array(x)[0])); labels.append(f"+{mk}")
            rows.append(labelled_row(tiles, labels, banner=f"{pretty(n)} - damage: {d}"))
        tgt = labelled_row([tile(target)], ["target"])
        canvas = Image.new("RGB", (rows[0].width, sum(r.height for r in rows) + tgt.height), (255, 255, 255))
        canvas.paste(tgt, (0, 0))
        y = tgt.height
        for r in rows:
            canvas.paste(r, (0, y)); y += r.height
        canvas.save(f"media/filmstrip_{d}.png")
    print("wrote filmstrips")


def failure_clips(models, target, K=16, seed=21):
    """Pick the worst-ending trial per (model, situation) and save it as a clip, if it is actually bad."""
    H, W = target.shape[:2]
    os.makedirs("media/failures", exist_ok=True)
    found = []
    for n, model in models.items():
        # (a) long rollout from seed, 2000 steps
        mx.random.seed(seed)
        x = mx.array(nca.make_seed(H, W, K))
        traj = []
        for t in range(2000):
            x = model(x)
            if t % 10 == 9:
                mx.eval(x)
                traj.append(np.array(x))
        traj = np.stack(traj)  # (200, K, H, W, C)
        err = ((np.clip(np.nan_to_num(traj[..., :4]), -1e3, 1e3) - target) ** 2).mean(axis=(2, 3, 4))
        k = int(np.argmax(err[-1] / err[19]))  # biggest drift from its own step-200 state
        osc = float(np.abs(np.diff(err[100:, k])).mean())
        if err[-1, k] > max(2 * err[19, k], 0.005) or osc > 2e-3:
            p = f"media/failures/{n}_long_rollout_trial{k}.gif"
            fr = [labelled_row([tile(traj[i, k])], [f"step {10*(i+1)}  mse {err[i,k]:.3f}"]) for i in range(0, 200, 2)]
            imageio.mimsave(p, [np.asarray(f) for f in fr], duration=0.05, loop=0)
            found.append((p, float(err[-1, k]), f"long rollout, no damage; mse at step 200 {err[19, k]:.4f}"))
        # (b) held-out damage: worst final error
        for i, d in enumerate(damage.HELDOUT_DAMAGE):
            rng = np.random.default_rng(seed + 10 + i)
            mx.random.seed(seed + i)
            x = np.array(model.rollout(mx.array(nca.make_seed(H, W, K)), 200))
            pre = ((np.clip(np.nan_to_num(x[..., :4]), -1e3, 1e3) - target) ** 2).mean(axis=(1, 2, 3))
            masks = [damage.HELDOUT_DAMAGE[d](H, W, rng) for _ in range(K)]
            for j in range(K):
                x[j] = damage.apply(x[j], masks[j])
            x = mx.array(x)
            traj = [np.array(x)]
            for t in range(600):
                x = model(x)
                if t % 4 == 3:
                    mx.eval(x); traj.append(np.array(x))
            traj = np.stack(traj)
            err = ((np.clip(np.nan_to_num(traj[..., :4]), -1e3, 1e3) - target) ** 2).mean(axis=(2, 3, 4))
            k = int(np.argmax(err[-1] / pre))  # worst final error relative to its own undamaged state
            if err[-1, k] > max(2 * pre[k], 0.005):  # did not heal: ended > 2x its pre-damage error
                p = f"media/failures/{n}_{d}_trial{k}.gif"
                fr = [labelled_row([tile(traj[i, k])], [f"+{4*i} mse {err[i,k]:.3f}"]) for i in range(len(traj))]
                imageio.mimsave(p, [np.asarray(f) for f in fr], duration=0.04, loop=0)
                found.append((p, float(err[-1, k]), f"after {d} damage, 600 steps; pre-damage mse {pre[k]:.4f}"))
    json.dump(found, open("media/failures/index.json", "w"), indent=1)
    for f in found:
        print("failure clip:", f)


def demo_html(model_name, target, extra=None):
    model = nca.load_model(os.path.join("checkpoints", model_name, "model.safetensors"))
    data = {"main": nca.export_json(model, target)}
    data["main"]["name"] = pretty(model_name)
    for n in (extra or []):
        if os.path.exists(os.path.join("checkpoints", n, "model.safetensors")):
            m = nca.load_model(os.path.join("checkpoints", n, "model.safetensors"))
            d = nca.export_json(m, target)
            d["name"] = pretty(n)
            data[n] = d
    tpl = open("demo_template.html").read()
    html = tpl.replace("/*__NCA_DATA__*/null", json.dumps(data, separators=(",", ":")))
    open("media/demo.html", "w").write(html)
    print("wrote media/demo.html", len(html) // 1024, "KB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["lizard_persist", "lizard_regen"])
    p.add_argument("--target", default="lizard")
    p.add_argument("--only", nargs="*", default=["gif", "repeat", "film", "fail", "html"])
    a = p.parse_args()
    target = nca.load_target(a.target)
    models = load(a.models)
    os.makedirs("media", exist_ok=True)
    if "gif" in a.only:
        side_by_side_gif(models, target, "media/grow_damage_regrow.gif")
    if "repeat" in a.only:
        repeated_gif(models, target, "media/repeated_damage.gif")
    if "birth" in a.only:
        birth_grid(models, target, "media/failures/stillbirth_grid.png")
    if "film" in a.only:
        filmstrips(models, target)
    if "fail" in a.only:
        failure_clips(models, target)
    if "html" in a.only:
        demo_html(a.models[-1], target, extra=a.models[:-1])


if __name__ == "__main__":
    main()
