"""Bounded CPU diagnostic of saved GPT weights, never a training/eval method.

FP32 NumPy reference without a KV cache. It can expose a learned wrong prediction
independently of the rollout decoder; it cannot certify Metal/BF16 numerical parity.
No MLX import. Keep probes <1 minute and threads <=4.
"""
import argparse
import json
import struct
import time
from pathlib import Path
import numpy as np
from r5_data import digest, json_write
from r5_world import Planner, prompt


class Reference:
    def __init__(self, checkpoint):
        path = Path(checkpoint)
        self.cfg = json.loads((path / "meta.json").read_text())["model_config"]
        assert self.cfg["n_head"] == self.cfg["n_kv_head"], "diagnostic supports this pilot's MHA only"
        with (path / "model.safetensors").open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            header, raw = json.loads(f.read(size)), f.read()
        self.w = {}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            assert spec["dtype"] == "F32"
            self.w[name] = np.frombuffer(raw, dtype="<f4", count=np.prod(spec["shape"]),
                                         offset=spec["data_offsets"][0]).reshape(spec["shape"])

    def logits(self, ids, positions):
        w, cfg = self.w, self.cfg
        norm = lambda x: x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(1e-6))
        x = norm(w["wte.weight"][ids])
        x0 = x.copy()
        T, H = len(ids), cfg["n_head"]
        D = cfg["n_embd"] // H
        theta = np.arange(T, dtype=np.float32)[:, None] * np.power(
            np.float32(100000), -np.arange(0, D, 2, dtype=np.float32) / D)[None, :]
        c, s = np.cos(theta)[None], np.sin(theta)[None]
        def rope(z):
            return np.concatenate((z[..., :D//2] * c - z[..., D//2:] * s,
                                   z[..., :D//2] * s + z[..., D//2:] * c), axis=-1)
        causal = np.tril(np.ones((T, T), bool))
        for i in range(cfg["n_layer"]):
            x = w["resid_lambdas"][i] * x + w["x0_lambdas"][i] * x0
            z, p = norm(x), f"h.{i}."
            q, k, v = [(z @ w[p + "attn.c_" + part + ".weight"].T).reshape(T, H, D).transpose(1, 0, 2)
                       for part in ("q", "k", "v")]
            q, k = norm(rope(q)) * np.float32(1.2), norm(rope(k)) * np.float32(1.2)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(np.float32(D))
            att = np.where(causal, att, np.float32(-np.inf))
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            x += (att @ v).transpose(1, 0, 2).reshape(T, -1) @ w[p + "attn.c_proj.weight"].T
            a = np.maximum(norm(x) @ w[p + "mlp.c_fc.weight"].T, 0)
            x += (a * a) @ w[p + "mlp.c_proj.weight"].T
        logits = norm(x)[positions] @ w["lm_head.weight"].T
        return np.float32(15) * np.tanh(logits / np.float32(15))


def score(logits, targets):
    loss = np.log(np.exp(logits - logits.max(-1, keepdims=True)).sum(-1)) + logits.max(-1)
    loss -= logits[np.arange(len(targets)), targets]
    return dict(nll_sum=float(loss.sum()), tokens=len(targets),
                exact=bool(np.array_equal(logits.argmax(-1), targets)),
                argmax=logits.argmax(-1).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="battery_s51")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-minutes", type=float, default=.9)
    a = ap.parse_args()
    assert 0 < a.max_minutes <= 1
    if Path(a.out).exists():
        raise FileExistsError(a.out)
    started = time.monotonic()
    deadline = started + 60 * a.max_minutes
    root = Path("results/r5") / a.run
    checkpoint = Path(json.loads((root / "selected.json").read_text())["methods"]["pretrained"])
    ref = Reference(checkpoint)
    from tokenizer import ChatTokenizer
    tok = ChatTokenizer.load()
    examples = []
    with np.load(root / "data/val.npz") as data:
        for i, (tokens, mask) in enumerate(zip(data["tokens"], data["masks"])):
            if time.monotonic() >= deadline:
                break
            positions = np.flatnonzero(mask)
            text = tok.decode(tokens[:positions[0] + 1])
            orders = text.split("ordered requests ")[1].split("\n")[0]
            lg = ref.logits(tokens[:positions[-1] + 1], positions)
            examples.append(dict(row=i, deliveries=len(orders.split(";")),
                                 **score(lg, tokens[positions + 1])))
        expected = len(data["targets"])
    groups = {}
    for name, rows in (("first32", examples[:32]), ("all", examples),
                       ("one_delivery", [r for r in examples if r["deliveries"] == 1]),
                       ("two_deliveries", [r for r in examples if r["deliveries"] == 2])):
        groups[name] = dict(examples=len(rows), nll=sum(r["nll_sum"] for r in rows) / max(1, sum(r["tokens"] for r in rows)),
                            teacher_forced_action_exact=sum(r["exact"] for r in rows) / max(1, len(rows)))
    # A preselected seen-task counterexample from the runner's first seed.
    ev = json.loads((root / "eval.json").read_text())
    ep = next(r for r in ev["episodes"] if r["method"] == "pretrained" and r["seed"] == 81_241_001)
    probes = []
    for k in (0, 6, 9, 10):
        if time.monotonic() >= deadline:
            break
        r = ep["trace"][k]
        teacher = Planner()(r["observation"])
        p = tok.render_for_completion([dict(role="user", content=prompt(r["observation"]))])
        target = tok.encode(teacher) + [tok.encode_special("<|assistant_end|>")]
        positions = np.arange(len(p) - 1, len(p) + len(target) - 1)
        sc = score(ref.logits(p + target[:-1], positions), np.array(target))
        probes.append(dict(step=k, rollout=r["action"], teacher=teacher,
                           reference_teacher_forced=tok.decode(sc["argmax"]), **sc))
    json_write(a.out, dict(kind="diagnostic_numpy_fp32_not_mlx_parity", run=a.run,
               checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint / "model.safetensors"),
               complete=len(examples) == expected, expected=expected, groups=groups,
               examples=examples, probes=probes, elapsed_s=time.monotonic() - started))
    print(json.dumps(dict(groups=groups, probes=probes, elapsed_s=time.monotonic() - started), indent=2))


if __name__ == "__main__":
    main()
