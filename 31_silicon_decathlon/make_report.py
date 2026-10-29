"""Build the exhibit from results/*.json: results/summary.md, media/leaderboard.png, media/crossover.png,
media/demo.html (self-contained offline leaderboard).

  ../.venv/bin/python make_report.py
  ../.venv/bin/python make_report.py --results logs/smoke --out logs/smoke    # smoke: keep the exhibit untouched

For each event it prefers results/<event>.json (the scored --final pass) over
results/<event>_provisional.json (dev runs on a busy machine) and labels which one it used.
Missing events are simply left out. Medals go to qualifying athletes only (the event's
qualifies() verdict), and a Core ML athlete whose compute plan put no op on the unit it asked for
(e.g. coreml_ane with every op on the CPU) is marked "fell back to CPU" and gets no medal: it is the
coreml_cpu athlete under another name. TF32-pass athletes (`*_tf32`, from results/<event>_tf32.json) are
shown as exhibition rows and get no medal: they were timed in a separate process (not interleaved with the
others) and judged by an error tolerance only, with no task check (reviewer decision, REVIEW.md).
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
MEDIA = os.path.join(HERE, "media")
sys.path.insert(0, HERE)

ATHLETE_ORDER = ["numpy_cpu", "mlx_cpu", "mlx_gpu", "mlx_gpu_tf32", "mlx_gpu_fp16", "mlx_gpu_bf16", "mlx_gpu_compiled",
                 "mlx_gpu_compiled_tf32", "mlx_gpu_q8", "mlx_gpu_q4", "coreml_cpu", "coreml_gpu", "coreml_ane"]
# where an athlete asks to run; colours follow this, in the reference palette's fixed slot order
FAMILIES = ["NumPy CPU", "MLX CPU", "MLX GPU", "Core ML CPU", "Core ML GPU", "Core ML ANE"]
COLORS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
COLORS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
CPU_CLASS = {"NumPy CPU", "MLX CPU", "Core ML CPU"}
WANT_DEVICE = {"coreml_gpu": "GPU", "coreml_ane": "NeuralEngine"}
MEDALS = ["gold", "silver", "bronze"]


def family(name):
    if name.startswith("numpy"):
        return "NumPy CPU"
    if name == "mlx_cpu":
        return "MLX CPU"
    if name.startswith("mlx"):
        return "MLX GPU"
    return {"coreml_cpu": "Core ML CPU", "coreml_gpu": "Core ML GPU", "coreml_ane": "Core ML ANE"}.get(name, "other")


def order_key(n):
    return ATHLETE_ORDER.index(n) if n in ATHLETE_ORDER else len(ATHLETE_ORDER)


# ---------------------------------------------------------------- loading
def load_results(res_dir=RES):
    """{event_module: dict(kind='final'|'provisional', path, data)}; final wins over provisional."""
    found, tf32 = {}, {}
    for p in sorted(glob.glob(os.path.join(res_dir, "*.json"))):
        base = os.path.basename(p)[:-5]
        if base.endswith("_tf32"):
            tf32[base[:-len("_tf32")]] = p
            continue
        kind = "provisional" if base.endswith("_provisional") else "final"
        ev = base[:-len("_provisional")] if kind == "provisional" else base
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as e:
            print(f"[make_report] skipping unreadable {base}.json: {e!r}")
            continue
        if not isinstance(data, dict) or "batches" not in data or "event" not in data:
            continue
        if ev in found and found[ev]["kind"] == "final":
            continue
        found[ev] = dict(kind=kind, path=os.path.relpath(p, HERE), data=data)
    for ev, p in tf32.items():   # TF32-pass athletes (mlx_gpu_tf32, ...) join their event's tables
        if ev in found:
            try:
                with open(p) as f:
                    found[ev]["tf32"] = json.load(f)
                found[ev]["tf32_path"] = os.path.relpath(p, HERE)
            except Exception as e:
                print(f"[make_report] skipping unreadable {os.path.basename(p)}: {e!r}")
    return dict(sorted(found.items()))


def placement_info(entry, name):
    """(placement dict or None, fell_back bool or None)."""
    c = (entry.get("coreml") or {}).get(name)
    if not c:
        return None, None
    pl = c.get("placement")
    if not isinstance(pl, dict) or "error" in pl or not pl:
        return pl, None
    want = WANT_DEVICE.get(name)
    if want is None:
        return pl, False
    return pl, pl.get(want, 0) == 0


def placement_text(pl):
    if not pl:
        return ""
    if "error" in pl:
        return "plan unreadable"
    tot = sum(v for v in pl.values() if isinstance(v, (int, float)))
    return ", ".join(f"{k} {v}/{tot}" for k, v in sorted(pl.items(), key=lambda kv: -kv[1]))


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        a = abs(v)
        if a >= 1000 or a < 1e-3:
            return f"{v:.2e}"
        return f"{v:.3g}"
    return str(v)


def build_rows(results):
    """Flatten into the structure every output uses."""
    events = []
    for ev, r in results.items():
        d = r["data"]
        meta = d["event"]
        e_out = dict(module=ev, id=meta.get("id", ev), project=meta.get("project", ""), model=meta.get("model", ""),
                     params=meta.get("params"), unit=meta.get("unit", "items"), notes=meta.get("notes", ""),
                     checkpoint=meta.get("checkpoint", ""), kind=r["kind"], source=r["path"],
                     tf32_source=r.get("tf32_path"),
                     mlx_precision=("not recorded (harness default then: TF32)" if "mlx_tf32" not in d else
                                    f"{'TF32' if d['mlx_tf32'] else 'true fp32'} (matmul rel err "
                                    f"{d.get('mlx_gpu_matmul_rel_err', float('nan')):.1g})"),
                     skips=d.get("skips", {}), task_checks=d.get("task_checks"), batches=[],
                     env=dict(time=d.get("env_before", {}).get("time"),
                              loadavg_before=d.get("env_before", {}).get("loadavg"),
                              loadavg_after=d.get("env_after", {}).get("loadavg"),
                              power=d.get("env_before", {}).get("power"),
                              thermal=d.get("env_after", {}).get("thermal"),
                              noise_before=d.get("noise_before"), noise_after=d.get("noise_after")))
        extra = (r.get("tf32") or {}).get("batches", {})
        for b, entry in sorted(d["batches"].items(), key=lambda kv: int(kv[0])):
            batch = int(b)
            entry = dict(entry)
            if b in extra:   # merge the TF32-pass athletes into this batch
                for key in ("timing", "agreement", "memory_bytes", "qualifies", "task"):
                    entry[key] = {**(entry.get(key) or {}), **(extra[b].get(key) or {})}
            timing = entry.get("timing", {})
            np_ms = timing.get("numpy_cpu", {}).get("median_ms")
            rows = []
            for name in sorted(timing, key=order_key):
                t = timing[name]
                q = (entry.get("qualifies") or {}).get(name)
                pl, fell = placement_info(entry, name)
                fam = family(name)
                rows.append(dict(
                    athlete=name, family=fam, median_ms=t["median_ms"], p90_ms=t["p90_ms"], min_ms=t.get("min_ms"),
                    calls=t.get("calls"), items_per_s=batch / (t["median_ms"] / 1e3),
                    x_numpy=(np_ms / t["median_ms"]) if np_ms else None,
                    agreement=(entry.get("agreement") or {}).get(name, {}),
                    task=(entry.get("task") or {}).get(name),
                    qualifies=None if q is None else q.get("ok"),
                    qualifies_known=q is not None,
                    reason="no verdict recorded (older harness)" if q is None else q.get("reason", ""),
                    placement=pl, placement_text=placement_text(pl), fell_back=bool(fell),
                    effective=("Core ML CPU" if fell else fam),
                    exhibition=name.endswith("_tf32"),
                    memory_bytes=(entry.get("memory_bytes") or {}).get(name),
                    package_bytes=((entry.get("coreml") or {}).get(name) or {}).get("size_bytes"),
                    medal=None))
            # a row is medal-eligible if it qualifies (missing verdict = harness default "qualifies")
            # TF32-pass rows are exhibition only (separate process, no task check): no medal
            elig = [r_ for r_ in rows if (r_["qualifies"] is True or not r_["qualifies_known"]) and not r_["fell_back"]
                    and not r_["exhibition"]]
            for i, r_ in enumerate(sorted(elig, key=lambda x: x["median_ms"])[:3]):
                r_["medal"] = MEDALS[i]
            e_out["batches"].append(dict(batch=batch, rows=rows, numpy_ms=np_ms))
        events.append(e_out)
    return events


def medal_table(events):
    tally = {}
    for e in events:
        for b in e["batches"]:
            for r in b["rows"]:
                if r["medal"]:
                    t = tally.setdefault(r["athlete"], dict(gold=0, silver=0, bronze=0))
                    t[r["medal"]] += 1
    out = [dict(athlete=a, **t, total=sum(t.values())) for a, t in tally.items()]
    return sorted(out, key=lambda x: (-x["gold"], -x["silver"], -x["bronze"], order_key(x["athlete"])))


def crossover_points(events):
    """Per event and batch: per-call work and which class of engine wins."""
    pts = []
    for e in events:
        for b in e["batches"]:
            # fallen-back Core ML rows are the coreml_cpu run under another name: leave them out
            elig = [r for r in b["rows"] if (r["qualifies"] is True or not r["qualifies_known"]) and not r["fell_back"]
                    and not r["exhibition"]]
            cpu = [r for r in elig if r["effective"] in CPU_CLASS]
            acc = [r for r in elig if r["effective"] not in CPU_CLASS]
            if not cpu or not acc:
                continue
            bc, ba = min(cpu, key=lambda r: r["median_ms"]), min(acc, key=lambda r: r["median_ms"])
            win = bc if bc["median_ms"] <= ba["median_ms"] else ba
            pts.append(dict(event=e["id"], batch=b["batch"], params=e["params"],
                            work=(e["params"] * b["batch"]) if e["params"] else None,
                            numpy_ms=b["numpy_ms"], best_cpu=bc["athlete"], best_cpu_ms=bc["median_ms"],
                            best_acc=ba["athlete"], best_acc_ms=ba["median_ms"],
                            ratio=ba["median_ms"] / bc["median_ms"], winner=win["athlete"],
                            winner_family=win["effective"]))
    return pts


def load_probe():
    p = os.path.join(RES, "probes", "coreml_batch1.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- summary.md
def write_summary(events, medals, pts, path):
    L = [f"# Silicon Decathlon: results summary", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} by `make_report.py` from `results/*.json`.", ""]
    kinds = {e["kind"] for e in events}
    if not events:
        L += ["No results yet.", ""]
    if "provisional" in kinds:
        L += ["**Provisional numbers are marked.** They come from dev runs (`--repeats 3 --budget 0.1`) on a "
              "busy shared machine and are noisy; the scored pass (`run_event.py all --final`) replaces them.", ""]
    L += ["| event | source | model | params | batches | unit |", "|---|---|---|---|---|---|"]
    for e in events:
        L.append(f"| {e['id']} | {e['kind'].upper()} (`{e['source']}`) | {e['model']} | {fmt(e['params'])} | "
                 f"{', '.join(str(b['batch']) for b in e['batches'])} | {e['unit']} |")
    L += ["", "## Medal table", "",
          "Per event and batch size, the three fastest qualifying athletes by median latency (= highest "
          "throughput at that batch). Core ML athletes whose compute plan placed no op on the requested unit are "
          "excluded (they ran on the CPU; see RESULTS.md, Core ML at batch 1). TF32-pass athletes (`mlx_gpu_tf32`, "
          "`mlx_gpu_compiled_tf32`) are exhibition only: they ran in a separate process, not interleaved with the "
          "others, and were judged by an error tolerance without the task check, so they get no medal (see the TF32 "
          "table below).", "",
          "| athlete | gold | silver | bronze | total |", "|---|---|---|---|---|"]
    for m in medals:
        L.append(f"| {m['athlete']} | {m['gold']} | {m['silver']} | {m['bronze']} | {m['total']} |")
    L += ["", "Podium per event:", "", "| event | batch | gold | silver | bronze |", "|---|---|---|---|---|"]
    for e in events:
        for b in e["batches"]:
            pod = {r["medal"]: f"{r['athlete']} ({r['median_ms']:.3g} ms)" for r in b["rows"] if r["medal"]}
            L.append(f"| {e['id']} | {b['batch']} | " + " | ".join(pod.get(m, "–") for m in MEDALS) + " |")
    tf = []
    for e in events:
        for b in e["batches"]:
            by = {r["athlete"]: r for r in b["rows"]}
            for base in ("mlx_gpu", "mlx_gpu_compiled"):
                if base in by and base + "_tf32" in by:
                    a, t = by[base], by[base + "_tf32"]
                    tf.append(f"| {e['id']} | {b['batch']} | {base} | {a['median_ms']:.4g} | {t['median_ms']:.4g} | "
                              f"{a['median_ms'] / t['median_ms']:.2f} | {'yes' if t['qualifies'] else 'no'} |")
    if tf:
        L += ["", "## TF32 exhibition (speed cost of true fp32)", "",
              "MLX GPU with its default TF32 matmuls, timed in a separate process (`run_event.py all --tf32-pass`) "
              "against the true-fp32 athletes of the scored pass. Cross-process differences of a few percent are "
              "noise (e.g. e15 b=50 TF32 is slower). Tolerance check only, no task check; no medals.", "",
              "| event | batch | athlete | true fp32 median ms | TF32 median ms | fp32 / TF32 | TF32 within tolerance |",
              "|---|---|---|---|---|---|---|"] + tf
    if pts:
        L += ["", "## Crossover (best accelerator vs best CPU engine, qualifying athletes; TF32 exhibition rows excluded)", "",
              "ratio = best accelerator median / best CPU-class median; < 1 means the GPU or Neural Engine wins.", "",
              "| event | batch | params x batch | numpy ms | best CPU | best accelerator | ratio | winner |",
              "|---|---|---|---|---|---|---|---|"]
        for p in sorted(pts, key=lambda p: p["work"] or 0):
            L.append(f"| {p['event']} | {p['batch']} | {fmt(p['work'])} | {fmt(p['numpy_ms'])} | "
                     f"{p['best_cpu']} {p['best_cpu_ms']:.3g} ms | {p['best_acc']} {p['best_acc_ms']:.3g} ms | "
                     f"{p['ratio']:.2f} | {p['winner']} |")
    for e in events:
        L += ["", f"## {e['id']}  ({e['kind'].upper()})", "",
              f"{e['model']}. Params {fmt(e['params'])}. One item = {e['unit']}. Checkpoint `{e['checkpoint']}`.",
              ""]
        if e["notes"]:
            note = e["notes"]
            if "TF32 by default here" in note and e["mlx_precision"].startswith("true fp32"):
                note += (" *(Reviewer: this note predates the true-fp32 reference; this run used true fp32, and "
                         "the CPU fp32 engines agree with it to ~1e-6, see the tables. The event module's note is fixed.)*")
            L += [note, ""]
        env = e["env"]
        L += [f"MLX GPU fp32 reference: {e['mlx_precision']}."
              + (f" `*_tf32` athletes come from `{e['tf32_source']}` (TF32 on, timed in a separate process, "
                 "not interleaved with the others; agreement vs the saved true-fp32 reference)." if e["tf32_source"] else ""),
              ""]
        L += [f"Run {env['time']}; load avg before {fmt_load(env['loadavg_before'])}, after "
              f"{fmt_load(env['loadavg_after'])}; {env['power'] or ''}; noise probe before/after "
              f"{fmt_noise(env['noise_before'])} / {fmt_noise(env['noise_after'])}."
              + ("" if e["task_checks"] in (None, True) else " **Task checks skipped (--no-task).**"), ""]
        for b in e["batches"]:
            L += [f"### batch {b['batch']}", "",
                  "| athlete | median ms | p90 ms | items/s | x numpy | agreement vs MLX fp32 | qualifies | "
                  "Core ML placement | peak MLX mem | medal |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in b["rows"]:
                ag = ", ".join(f"{k} {fmt(v)}" for k, v in r["agreement"].items())
                q = ("yes" if r["qualifies"] else ("**no**" if r["qualifies"] is False else "?")) + \
                    (f": {r['reason']}" if r["reason"] else "")
                pl = r["placement_text"] + (" **(fell back to CPU)**" if r["fell_back"] else "")
                mem = f"{r['memory_bytes'] / 2**20:.1f} MB" if r["memory_bytes"] else ""
                L.append(f"| {r['athlete']} | {r['median_ms']:.4g} | {r['p90_ms']:.4g} | {r['items_per_s']:,.0f} | "
                         f"{fmt(r['x_numpy'])} | {ag} | {q} | {pl} | {mem} | "
                         f"{r['medal'] or ('exhibition (TF32 pass)' if r['exhibition'] else '')} |")
            L.append("")
        if e["skips"]:
            L += ["Skipped athletes: " + "; ".join(f"`{k}`: {v}" for k, v in e["skips"].items()), ""]
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


def fmt_load(v):
    return "–" if not v else "/".join(f"{x:.1f}" for x in v)


def fmt_noise(v):
    if not v:
        return "–"
    return f"CPU {v.get('numpy_512_matmul_ms', 0):.2f} ms, GPU {v.get('mlx_gpu_2048_matmul_ms', 0):.2f} ms"


# ---------------------------------------------------------------- charts
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                         "ytick.color": "#52514e", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    return plt


def leaderboard_png(events, path):
    """Heatmap: speed relative to NumPy (log, diverging blue = faster, red = slower), one row per event x batch."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    plt = _plt()
    rows = [(e, b) for e in events for b in e["batches"]]
    names = [n for n in ATHLETE_ORDER if any(r["athlete"] == n for _, b in rows for r in b["rows"])]
    fig_h = 1.4 + 0.42 * max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.82 * max(len(names), 1) + 2.2, fig_h))
    if not rows:
        ax.text(0.5, 0.5, "no results yet", ha="center"); fig.savefig(path, dpi=150); plt.close(fig); return
    M = np.full((len(rows), len(names)), np.nan)
    for i, (e, b) in enumerate(rows):
        for r in b["rows"]:
            if r["x_numpy"]:
                M[i, names.index(r["athlete"])] = r["x_numpy"]
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f3b9b5", "#f0efec", "#9ec5f4", "#1c5cab"])
    finite = M[np.isfinite(M)]
    span = float(np.max(np.abs(np.log10(finite)))) if finite.size else 1.0
    lim = 10 ** float(np.clip(np.ceil(span), 1, 3))
    norm = LogNorm(vmin=1 / lim, vmax=lim)
    ax.imshow(np.where(np.isfinite(M), M, 1.0), cmap=cmap, norm=norm, aspect="auto")
    for i, (e, b) in enumerate(rows):
        for r in b["rows"]:
            j = names.index(r["athlete"])
            if not np.isfinite(M[i, j]):
                continue
            v = M[i, j]
            txt = f"{v:.2g}x" if v < 100 else f"{v:.0f}x"
            mark = {"gold": " ●", "silver": " ◐", "bronze": " ○"}.get(r["medal"], "")
            dq = r["qualifies"] is False
            fb = r["fell_back"]
            label = txt + mark + ("\nDQ" if dq else "") + ("\nCPU!" if fb else "")
            dark = abs(np.log10(v)) > 0.6 * np.log10(lim)
            ax.text(j, i, label, ha="center", va="center", fontsize=7, color="#ffffff" if dark else "#0b0b0b",
                    fontweight="bold" if r["medal"] == "gold" else "normal")
            if dq:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#8a8984", lw=0, alpha=0.6))
        for j, n in enumerate(names):
            if not np.isfinite(M[i, j]):
                ax.text(j, i, "–", ha="center", va="center", fontsize=7, color="#8a8984")
    ax.set_xticks(range(len(names)), [n.replace("_", "\n", 1) for n in names], fontsize=7.5)
    ax.set_yticks(range(len(rows)), [f"{e['id']}  b={b['batch']}" + ("" if e["kind"] == "final" else " (prov.)")
                                     for e, b in rows], fontsize=8)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(names)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which="minor", color="#fcfcfb", lw=2)
    ax.tick_params(which="minor", length=0)
    ax.xaxis.tick_top()
    ax.set_title("Speed vs NumPy fp32 at matched batch (NumPy median / athlete median; blue = faster, red = slower)\n"
                 "● gold  ◐ silver  ○ bronze among qualifying athletes   DQ (hatched) = fails the event's qualifies()   "
                 "CPU! = Core ML put every op on the CPU   *_tf32 = exhibition (separate process, no task check, no medal)", fontsize=8.5, color="#0b0b0b", pad=34, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def crossover_png(pts, probe, path):
    import numpy as np
    plt = _plt()
    ncols = 3 if probe else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.0 * ncols, 4.4))
    fam_color = dict(zip(FAMILIES, COLORS_LIGHT))
    fam_marker = dict(zip(FAMILIES, ["o", "s", "D", "^", "v", "P"]))
    for ax, key, xlabel in ((axes[0], "work", "per-call work: params x batch"),
                            (axes[1], "numpy_ms", "per-call work: NumPy fp32 time at that batch (ms)")):
        ps = [p for p in pts if p[key]]
        for fam in FAMILIES:
            sel = [p for p in ps if p["winner_family"] == fam]
            if sel:
                ax.scatter([p[key] for p in sel], [p["ratio"] for p in sel], s=56, c=fam_color[fam],
                           marker=fam_marker[fam], edgecolors="#fcfcfb", linewidths=1.5, label=f"winner: {fam}", zorder=3)
        for p in ps:
            ax.annotate(f"{p['event'].split('_')[0]} b{p['batch']}", (p[key], p["ratio"]), xytext=(4, 3),
                        textcoords="offset points", fontsize=6.5, color="#52514e")
        ax.axhline(1.0, color="#8a8984", lw=1, ls="--", zorder=1)
        ax.text(0.99, 0.97, "above the line: a CPU engine wins\n(dispatch overhead dominates)", transform=ax.transAxes,
                fontsize=7, va="top", ha="right", color="#52514e")
        ax.text(0.01, 0.03, "below: GPU / Neural Engine wins\n(throughput dominates)", transform=ax.transAxes, fontsize=7,
                va="bottom", ha="left", color="#52514e")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(xlabel); ax.set_ylabel("best accelerator time / best CPU time")
        ax.grid(True, which="major", color="#e4e3df", lw=0.6)
        if not ps:
            ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=len(h), bbox_to_anchor=(0.36, 0.0))
    axes[0].set_title("Where the crossover sits, by model size", loc="left", fontsize=9)
    axes[1].set_title("...and by measured work", loc="left", fontsize=9)
    if probe:
        ax = axes[2]
        rows = [r for r in probe["rows"] if r["input_shape"][0] == 1]
        units = [("coreml_cpu", "Core ML CPU"), ("coreml_gpu", "Core ML GPU"), ("coreml_ane", "Core ML ANE")]
        for u, fam in units:
            xs = [r["params"] for r in rows]
            ys = [r["timing"][u]["median_ms"] for r in rows]
            ax.plot(xs, ys, color=fam_color[fam], lw=2, marker=fam_marker[fam], ms=6, label=fam,
                    markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            for r in rows:
                pref = r["plan"][u]["preferred"]
                if u != "coreml_cpu" and pref.get("CPU", 0) == sum(pref.values()):
                    ax.annotate("CPU", (r["params"], r["timing"][u]["median_ms"]), xytext=(4, -9 if u == "coreml_gpu" else 4),
                                textcoords="offset points", fontsize=6.5, color=fam_color[fam])
        for r in rows:
            ax.annotate(r["model"].replace("_random", "").replace("_real", " (e09)"),
                        (r["params"], r["timing"]["coreml_cpu"]["median_ms"]), xytext=(-8, 10),
                        textcoords="offset points", fontsize=6.5, color="#52514e")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("params (batch 1)"); ax.set_ylabel("median latency (ms)")
        ax.grid(True, which="major", color="#e4e3df", lw=0.6)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_title("Core ML batch-1 probe: 'CPU' = planner put every op on the CPU", loc="left", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- demo.html
def demo_html(events, medals, path):
    data = dict(generated=time.strftime("%Y-%m-%d %H:%M"), families=FAMILIES,
                colors_light=COLORS_LIGHT, colors_dark=COLORS_DARK, medals=medals,
                events=[dict(id=e["id"], model=e["model"], params=e["params"], unit=e["unit"], notes=e["notes"],
                             kind=e["kind"], skips=e["skips"], checkpoint=e["checkpoint"],
                             batches=[dict(batch=b["batch"], rows=[
                                 {k: r[k] for k in ("athlete", "family", "effective", "median_ms", "p90_ms",
                                                    "items_per_s", "x_numpy", "qualifies", "reason", "placement_text",
                                                    "fell_back", "medal", "agreement", "memory_bytes")}
                                 for r in b["rows"]]) for b in e["batches"]]) for e in events])
    blob = json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("</", "<\\/")
    with open(os.path.join(HERE, "demo_template.html")) as f:
        tpl = f.read()
    with open(path, "w") as f:
        f.write(tpl.replace("/*__DATA__*/null", blob))


def main(res_dir=RES, out_dir=None):
    """out_dir=None writes the exhibit (results/summary.md + media/); otherwise everything goes to out_dir."""
    media = out_dir or MEDIA
    summary = os.path.join(out_dir or res_dir, "summary.md")
    os.makedirs(media, exist_ok=True)
    results = load_results(res_dir)
    events = build_rows(results)
    medals = medal_table(events)
    pts = crossover_points(events)
    probe = load_probe()
    write_summary(events, medals, pts, summary)
    leaderboard_png(events, os.path.join(media, "leaderboard.png"))
    crossover_png(pts, probe, os.path.join(media, "crossover.png"))
    demo_html(events, medals, os.path.join(media, "demo.html"))
    print(f"[make_report] {len(events)} events ({', '.join(e['id'] + ':' + e['kind'][0].upper() for e in events)}); "
          f"wrote {os.path.relpath(summary, HERE)} and leaderboard.png, crossover.png, demo.html in "
          f"{os.path.relpath(media, HERE)}/")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=RES, help="directory of <event>[_provisional].json (default results/)")
    ap.add_argument("--out", help="write everything here instead of results/summary.md + media/ (smoke runs)")
    a = ap.parse_args()
    main(a.results, a.out)
