"""Core ML helpers: convert once (cached .mlpackage), load per compute unit, report op placement.

torch is used here only as a conversion frontend for coremltools (no training).
Shapes are static: export one package per batch size.
"""
import os
import warnings
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, "checkpoints", "coreml")

UNITS = {
    "coreml_cpu": "CPU_ONLY",
    "coreml_gpu": "CPU_AND_GPU",
    "coreml_ane": "CPU_AND_NE",
}


def package_path(name):
    os.makedirs(PKG_DIR, exist_ok=True)
    return os.path.join(PKG_DIR, name + ".mlpackage")


def export_torch(torch_module, example_inputs, input_names, name, precision="fp16", rebuild=False):
    """Trace a torch module and save <name>.mlpackage. example_inputs: tuple of numpy arrays.
    precision: 'fp16' (Core ML default, needed for the Neural Engine) or 'fp32'."""
    import coremltools as ct
    import torch
    path = package_path(name)
    if os.path.exists(path) and not rebuild:
        return path
    torch_module = torch_module.eval()
    ex = tuple(torch.from_numpy(np.ascontiguousarray(a)) for a in example_inputs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        traced = torch.jit.trace(torch_module, ex)
        ml = ct.convert(
            traced,
            inputs=[ct.TensorType(name=n, shape=a.shape, dtype=a.dtype) for n, a in zip(input_names, example_inputs)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16 if precision == "fp16" else ct.precision.FLOAT32,
            minimum_deployment_target=ct.target.macOS15,
        )
    ml.save(path)
    return path


def load(path, athlete):
    """Load a saved package for one athlete name in UNITS."""
    import coremltools as ct
    return ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, UNITS[athlete]))


def output_names(path):
    import coremltools as ct
    spec = ct.utils.load_spec(path)
    return [o.name for o in spec.description.output]


def make_athlete(path, athlete, rename=None):
    """Returns fn(inputs dict of numpy) -> outputs dict of numpy.
    rename maps Core ML output names to the event's output keys (in order if a list)."""
    model = load(path, athlete)
    names = output_names(path)
    if isinstance(rename, (list, tuple)):
        rename = dict(zip(names, rename))
    rename = rename or {}

    def fn(inputs):
        out = model.predict(inputs)
        return {rename.get(k, k): np.asarray(v) for k, v in out.items()}
    return fn


def placement(path, athlete="coreml_ane"):
    """Count ops by preferred device from Core ML's compute plan, e.g. {'NeuralEngine': 42}.
    Returns None if the plan can't be read."""
    import coremltools as ct
    try:
        from coremltools.models.compute_plan import MLComputePlan
        compiled = ct.utils.compile_model(path)
        plan = MLComputePlan.load_from_path(path=compiled, compute_units=getattr(ct.ComputeUnit, UNITS[athlete]))
        fn = plan.model_structure.program.functions["main"]
        counts = Counter()
        for op in fn.block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if usage is not None:
                dev = type(usage.preferred_compute_device).__name__
                counts[dev.replace("ML", "").replace("ComputeDevice", "")] += 1
        return dict(counts)
    except Exception as e:
        return {"error": repr(e)[:200]}
