"""Read the published ViNT / NoMaD checkpoints (visualnav-transformer, MIT) into plain numpy, once.

    ../.venv/bin/python -I -m r8_nav.weights      (run from 32_microduck; writes data/pretrained/{vint,nomad}/*.npz)

The checkpoints are untrusted data. They are opened ONLY with torch.load(..., weights_only=True); nothing from the
download is executed and no vint_train / efficientnet_pytorch code is imported (neither is installed).
- nomad.pth is a plain state dict (EMA weights, archive name "ema_29"): weights_only=True reads it directly.
- vint.pth is a training checkpoint that pickles whole objects ({"epoch", "model": ViNT module, "optimizer": AdamW,
  "scheduler": GradualWarmupScheduler, "avg_total_test_loss": numpy float}). weights_only=True refuses unknown
  classes, so every class it names is allow-listed as an inert stub of ours (_Stub below: it only stores the pickled
  state; no method of the original class runs). The stub tree is then walked like nn.Module.state_dict() to collect
  parameters and buffers, plus the few constructor facts we verify against (padding, heads, norm_first, ...).
torch is used here and in the parity test only; the runtime path (mlx_models.py) reads the .npz files.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
PRETRAINED = HERE / "data" / "pretrained"
VINT_PTH = PRETRAINED / "vint" / "vint.pth"
NOMAD_PTH = PRETRAINED / "nomad" / "nomad.pth"
VINT_NPZ = PRETRAINED / "vint" / "vint_weights.npz"
NOMAD_NPZ = PRETRAINED / "nomad" / "nomad_weights.npz"
SOURCES = {
    "folder": "https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg (README 'Pre-Trained Models')",
    "vint": "https://drive.google.com/file/d/1ckrceGb5m_uUtq3pD8KHwnqtJgPl6kF5 (vint.pth)",
    "nomad": "https://drive.google.com/file/d/1YJhkkMJAYOiKNyCaelbS_alpUpAJsOUb (nomad.pth)",
}

# Every global vint.pth names (torch.serialization.get_unsafe_globals_in_checkpoint), mapped to an inert stub.
VINT_GLOBALS = [
    "torch.nn.modules.container.Sequential", "torch.optim.lr_scheduler.CosineAnnealingLR", "torch.optim.adamw.AdamW",
    "torch.nn.modules.activation.MultiheadAttention", "numpy.core.multiarray.scalar",
    "torch.nn.modules.transformer.TransformerEncoderLayer", "torch.nn.modules.transformer.TransformerEncoder",
    "vint_train.models.vint.vint.ViNT", "efficientnet_pytorch.utils.BlockArgs", "torch.nn.modules.container.ModuleList",
    "efficientnet_pytorch.utils.Conv2dStaticSamePadding", "torch.nn.modules.batchnorm.BatchNorm2d",
    "torch.nn.modules.normalization.LayerNorm", "torch.nn.modules.padding.ZeroPad2d",
    "torch.nn.modules.linear.NonDynamicallyQuantizableLinear", "vint_train.models.vint.self_attention.PositionalEncoding",
    "efficientnet_pytorch.utils.MemoryEfficientSwish", "warmup_scheduler.scheduler.GradualWarmupScheduler",
    "torch.nn.modules.dropout.Dropout", "torch.nn.modules.linear.Linear",
    "vint_train.models.vint.self_attention.MultiLayerDecoder", "efficientnet_pytorch.utils.GlobalParams",
    "torch._C._nn.gelu", "efficientnet_pytorch.model.EfficientNet", "numpy.dtype", "torch.nn.modules.linear.Identity",
    "efficientnet_pytorch.model.MBConvBlock", "torch.nn.modules.pooling.AdaptiveAvgPool2d",
]


class _Stub:
    """Stand-in for a pickled class: keeps constructor args and the BUILD state, runs nothing else."""
    _path = "?"

    def __new__(cls, *args, **kwargs):
        obj = object.__new__(cls)
        obj.__dict__["_args"] = args
        obj.__dict__["_state"] = None
        return obj

    def __init__(self, *args, **kwargs):   # REDUCE-style construction (cls(*args)) lands here too
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state

    def __repr__(self):
        return f"<stub {self._path}>"


def _make_stub(path):
    module, name = path.rsplit(".", 1)
    return type(name, (_Stub,), {"__module__": module, "__qualname__": name, "_path": path})


def _stub_function(path):
    def f(*args, **kwargs):
        return ("stub-call", path, args)
    f.__module__, f.__qualname__ = path.rsplit(".", 1)
    return f


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_nomad_state(path=NOMAD_PTH):
    import torch
    sd = torch.load(str(path), weights_only=True, map_location="cpu")
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in sd.items()}


def _plain_dict(*args):
    """collections.defaultdict(dict) in the optimizer state, read back as a plain dict (the weights_only unpickler
    only SETITEMS into dict / OrderedDict / Counter)."""
    return {}


def _vint_safe_globals():
    out = [(dict, "builtins.dict"), (_plain_dict, "collections.defaultdict")]
    for path in VINT_GLOBALS:
        if path in ("torch._C._nn.gelu", "numpy.core.multiarray.scalar"):
            out.append((_stub_function(path), path))
        else:
            out.append((_make_stub(path), path))
    return out


def load_vint_checkpoint(path=VINT_PTH):
    """The pickled checkpoint dict, every non-tensor object replaced by a stub."""
    import torch
    with torch.serialization.safe_globals(_vint_safe_globals()):
        return torch.load(str(path), weights_only=True, map_location="cpu")


def _module_state(stub):
    st = stub._state
    if isinstance(st, tuple) and len(st) == 2 and isinstance(st[0], dict):   # (state, slotstate)
        st = st[0]
    return st if isinstance(st, dict) else {}


def walk_state_dict(stub, prefix=""):
    """nn.Module.state_dict() on the stub tree: parameters, persistent buffers, submodules (in order)."""
    import torch
    st = _module_state(stub)
    out = {}
    nonpersist = st.get("_non_persistent_buffers_set") or set()
    for name, p in (st.get("_parameters") or {}).items():
        if p is not None:
            out[prefix + name] = p.detach().cpu().numpy().astype(np.float32)
    for name, b in (st.get("_buffers") or {}).items():
        if b is not None and name not in nonpersist and isinstance(b, torch.Tensor):
            out[prefix + name] = b.detach().cpu().numpy()
    for name, m in (st.get("_modules") or {}).items():
        if m is not None:
            out.update(walk_state_dict(m, prefix + name + "."))
    return out


def find_modules(stub, prefix=""):
    """(dotted name, class path, state) for every module in the stub tree."""
    st = _module_state(stub)
    yield prefix.rstrip("."), getattr(stub, "_path", "?"), st
    for name, m in (st.get("_modules") or {}).items():
        if m is not None:
            yield from find_modules(m, prefix + name + ".")


def vint_facts(model_stub):
    """Constructor facts read from the pickled ViNT object (used to pin the port's configuration)."""
    facts = {"static_padding": {}, "transformer": {}, "top": {}}
    top = _module_state(model_stub)
    for k in ("context_size", "learn_angle", "len_trajectory_pred", "num_action_params", "obs_encoding_size",
              "goal_encoding_size", "late_fusion", "num_obs_features", "num_goal_features"):
        if k in top:
            facts["top"][k] = top[k]
    for name, cls, st in find_modules(model_stub):
        if cls.endswith("ZeroPad2d"):
            facts["static_padding"][name] = list(st.get("padding"))
        elif cls.endswith("Conv2dStaticSamePadding"):
            facts.setdefault("conv_stride", {})[name] = list(st.get("stride"))
        elif cls.endswith("TransformerEncoderLayer"):
            facts["transformer"][name] = {"norm_first": st.get("norm_first"),
                                          "activation": repr(st.get("activation")),
                                          "activation_relu_or_gelu": st.get("activation_relu_or_gelu")}
        elif cls.endswith("MultiheadAttention"):
            facts["transformer"][name] = {"num_heads": st.get("num_heads"), "embed_dim": st.get("embed_dim"),
                                          "batch_first": st.get("batch_first")}
        elif cls.endswith("LayerNorm"):
            facts.setdefault("layernorm_eps", {})[name] = st.get("eps")
        elif cls.endswith("BatchNorm2d"):
            facts.setdefault("batchnorm_eps", set()).add(st.get("eps"))
        elif cls.endswith("EfficientNet"):
            gp = st.get("_global_params")
            if gp is not None:
                facts.setdefault("global_params", {})[name] = [repr(a) for a in gp._args]
    if "batchnorm_eps" in facts:
        facts["batchnorm_eps"] = sorted(facts["batchnorm_eps"])
    return facts


def convert():
    """Write both .npz files (+ a provenance json) next to the downloads."""
    import torch
    ck = load_vint_checkpoint()
    model = ck["model"]
    vsd = walk_state_dict(model)
    facts = vint_facts(model)
    np.savez(VINT_NPZ, **vsd)
    nsd = load_nomad_state()
    np.savez(NOMAD_NPZ, **nsd)
    prov = {
        "sources": SOURCES,
        "download": "curl, non-interactive: the drive.usercontent.google.com 'Virus scan warning' page's own GET form "
                    "(id, export=download, confirm=t, uuid) re-requested with curl; no browser, no login",
        "vint.pth": {"bytes": VINT_PTH.stat().st_size, "sha256": sha256(VINT_PTH), "epoch": ck.get("epoch"),
                     "n_params": int(sum(v.size for k, v in vsd.items() if v.dtype == np.float32)),
                     "n_tensors": len(vsd)},
        "nomad.pth": {"bytes": NOMAD_PTH.stat().st_size, "sha256": sha256(NOMAD_PTH),
                      "n_params": int(sum(v.size for v in nsd.values())), "n_tensors": len(nsd)},
        "torch": torch.__version__,
        "loader": "torch.load(weights_only=True); vint.pth's classes allow-listed as inert stubs (r8_nav/weights.py)",
        "vint_facts": facts,
    }
    (PRETRAINED / "vint" / "provenance.json").write_text(json.dumps(prov, indent=1, default=str))
    (PRETRAINED / "nomad" / "provenance.json").write_text(json.dumps(
        {k: prov[k] for k in ("sources", "download", "nomad.pth", "torch", "loader")}, indent=1, default=str))
    return prov


if __name__ == "__main__":
    p = convert()
    json.dump({k: v for k, v in p.items() if k != "vint_facts"}, sys.stdout, indent=1, default=str)
    print()
    print(json.dumps(p["vint_facts"], indent=1, default=str)[:4000])
