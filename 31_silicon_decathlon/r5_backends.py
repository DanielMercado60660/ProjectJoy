"""Lazy engine adapters. Importing this module never imports MLX/Core ML.

No conversion, downloads, torch, quantization or fitting. Existing Core ML
packages are loaded as-is; failure is a recorded capability gap, never fallback.
"""
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time

import numpy as np
from r5_core import atomic_json, digest
from r5_numpy import NumpyCar, NumpyTracker

SELECTED = Path("checkpoints/r5_selected")
PACKAGES = {"car": "checkpoints/coreml/e08_grand_prix_b1_fp16.mlpackage",
            "tracker": "checkpoints/coreml/e15_tracker_b1_fp16.mlpackage"}
ENGINES = ("numpy_cpu", "mlx_cpu", "mlx_gpu", "mlx_gpu_compiled",
           "coreml_cpu", "coreml_gpu", "coreml_ane")
DEFAULT_ENGINES = ("numpy_cpu", "mlx_gpu", "coreml_cpu", "coreml_ane")


def selected_manifest():
    """Copy the two frozen weight sets once; optimizer is explicitly inapplicable.

    Only these selected weights and small run cursors need committing. Existing
    exhibit checkpoints are read-only and no intermediate snapshots are made.
    """
    dest = SELECTED / "manifest.json"
    if not dest.exists():
        SELECTED.mkdir(parents=True, exist_ok=True)
        files = {"car.npz": "../08_tiny_grand_prix/checkpoints/es_best.npz",
                 "tracker.weights.npz": "../15_object_permanence/checkpoints/base/final.weights.npz"}
        provenance = {}
        for target, source in files.items():
            output = SELECTED / target
            shutil.copyfile(source, str(output)+".tmp")
            os.replace(str(output)+".tmp", output)
            provenance[target] = dict(source=source, sha256=digest(output))
        # Retain only architecture config, not the original optimizer/history.
        with open("../15_object_permanence/checkpoints/base/final.state.pkl", "rb") as f:
            cfg = pickle.load(f)["cfg"]
        atomic_json(SELECTED / "tracker_config.json", cfg)
        provenance["tracker_config.json"] = dict(sha256=digest(SELECTED / "tracker_config.json"))
        atomic_json(dest, dict(files=provenance, training="none; frozen round-3 models",
                               optimizer_state=None, step=0))
    manifest = json.loads(dest.read_text())
    for name, info in manifest["files"].items():
        if digest(SELECTED / name) != info["sha256"]:
            raise ValueError(f"Selected checkpoint changed: {name}")
    return manifest


def sh(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=3)
        return dict(returncode=p.returncode, stdout=p.stdout.strip(), stderr=p.stderr.strip())
    except Exception as e:
        return dict(error=str(e))


def environment():
    versions = {}
    for name in ("numpy", "mlx", "coremltools"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return dict(time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), uptime=sh(["uptime"]),
                power=sh(["pmset", "-g", "batt"]), thermal=sh(["pmset", "-g", "therm"]),
                loadavg=list(os.getloadavg()), versions=versions,
                threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                                        "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
                mlx_enable_tf32=os.environ.get("MLX_ENABLE_TF32"))


class Backend:
    def __init__(self, event, engine):
        self.event, self.engine, self.model = event, engine, None
        self.meta = dict(event=event, engine=engine)
        self.hid = 256
        if engine == "numpy_cpu":
            self.model = NumpyCar() if event == "car" else NumpyTracker()
            if event == "tracker":
                self.hid = self.model.hid
        elif engine.startswith("mlx_"):
            import mlx.core as mx
            self.mx = mx
            self.device = mx.cpu if engine == "mlx_cpu" else mx.gpu
            with mx.stream(self.device):
                if event == "car":
                    from models.grand_prix import load_mlx
                    self.model = load_mlx(str(SELECTED / "car.npz"))
                    self.call = self.model
                else:
                    from models.tracker import Tracker
                    cfg = json.loads((SELECTED / "tracker_config.json").read_text())
                    assert cfg["layers"] == 1
                    self.hid = cfg["hid"]
                    self.model = Tracker(enc_dim=cfg["enc_dim"], hid=self.hid, layers=1)
                    self.model.load_weights(str(SELECTED / "tracker.weights.npz"))
                    self.call = self.model.step
                self.model.eval()
                mx.eval(self.model.parameters())
                if engine.endswith("compiled"):
                    self.call = mx.compile(self.call, inputs=[self.model.state])
            self.meta["precision"] = "float32, MLX_ENABLE_TF32="+str(os.environ.get("MLX_ENABLE_TF32"))
        elif engine.startswith("coreml_"):
            import coremltools as ct
            path = Path(PACKAGES[event])
            if not path.is_dir():
                raise FileNotFoundError(f"Existing package missing: {path}; no automatic conversion")
            unit = {"coreml_cpu": "CPU_ONLY", "coreml_gpu": "CPU_AND_GPU", "coreml_ane": "CPU_AND_NE"}[engine]
            self.model = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, unit))
            self.outputs = [o.name for o in self.model.get_spec().description.output]
            assert len(self.outputs) == (1 if event == "car" else 2)
            self.meta.update(package=str(path), requested_compute_units=unit, precision="fp16 package",
                             actual_placement="not measured in R5; requested units may fall back to CPU",
                             package_hashes={str(p.relative_to(path)): digest(p)
                                             for p in sorted(path.rglob("*")) if p.is_file()})
        else:
            raise ValueError(engine)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, self.hid), np.float32)

    def infer(self, inputs):
        start = time.perf_counter()
        inp = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in inputs.items()}
        if self.event == "tracker":
            assert inp["frame"].shape == (1, 64, 64, 3)
            inp["h"] = self.h
        else:
            assert inp["obs"].shape == (1, 13)
        if self.engine.startswith("mlx_"):
            mx = self.mx
            with mx.stream(self.device):
                arrays = {k: mx.array(v) for k, v in inp.items()}
                mx.eval(*arrays.values())
                dispatched = time.perf_counter()
                if self.event == "car":
                    raw = (self.call(arrays["obs"]),)
                else:
                    raw = self.call(arrays["frame"], arrays["h"])
                mx.eval(*raw)
                computed = time.perf_counter()
                vals = [np.array(v, dtype=np.float32) for v in raw]
        else:
            dispatched = time.perf_counter()
            if self.engine.startswith("coreml_"):
                result = self.model.predict(inp)
                vals = [result[k] for k in self.outputs]
            elif self.event == "car":
                vals = [self.model(inp["obs"])]
            else:
                vals = self.model.step(inp["frame"], inp["h"])
            computed = time.perf_counter()
        if self.event == "car":
            out = {"action": np.asarray(vals[0], dtype=np.float32).reshape(1, 2)}
        else:
            self.h = np.asarray(vals[1], dtype=np.float32).reshape(1, self.hid).copy()
            out = {"out": np.asarray(vals[0], dtype=np.float32).reshape(1, 3, 5)}
        if not all(np.isfinite(v).all() for v in [*out.values(), self.h]):
            raise ValueError("Nonfinite network output/state")
        finished = time.perf_counter()
        return out, dict(dispatch_ms=(dispatched-start)*1000, compute_ms=(computed-dispatched)*1000,
                         postprocess_ms=(finished-computed)*1000)


def _background(active, ready, quiet, stop, count, failure):
    try:
        import mlx.core as mx
        with mx.stream(mx.gpu):
            a = mx.random.normal((1024, 1024), key=mx.random.key(5031))
            b = mx.random.normal((1024, 1024), key=mx.random.key(5032))
            mx.eval(a, b, a @ b)
            ready.set()
            while not stop.is_set():
                quiet.set()
                if not active.wait(0.02):
                    continue
                quiet.clear()
                start = time.perf_counter()
                mx.eval(a @ b)
                with count.get_lock():
                    count.value += 1
                # About 70% duty at uncontended speed, bounded allocation and one stream.
                time.sleep(max(0.0001, (time.perf_counter()-start)*3/7))
    except Exception as e:
        failure.send(repr(e))
        ready.set()


class Background:
    def __init__(self):
        ctx = mp.get_context("spawn")
        self.active, self.ready, self.quiet, self.stop = ctx.Event(), ctx.Event(), ctx.Event(), ctx.Event()
        self.count = ctx.Value("Q", 0)
        self.parent, child = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=_background, args=(self.active, self.ready, self.quiet, self.stop,
                                                            self.count, child), daemon=True)
        self.process.start()
        if not self.ready.wait(20) or not self.process.is_alive() or self.parent.poll():
            reason = self.parent.recv() if self.parent.poll() else "background startup timeout/exit"
            self.close()
            raise RuntimeError(reason)

    def set_load(self, loaded):
        if not self.process.is_alive() or self.parent.poll():
            raise RuntimeError("Background worker failed")
        if loaded:
            self.active.set()
            before = self.count.value
            deadline = time.perf_counter()+3
            while self.count.value == before:
                if time.perf_counter() > deadline or not self.process.is_alive():
                    raise RuntimeError("Background not advancing")
                time.sleep(0.005)
        else:
            self.active.clear()
            # The worker acknowledges its idle wait after any in-flight matmul.
            time.sleep(0.03)
            if not self.quiet.wait(3):
                raise RuntimeError("Background failed to pause")
        return self.count.value

    def close(self):
        self.stop.set()
        self.active.set()
        self.process.join(2)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(2)
