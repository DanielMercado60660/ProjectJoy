"""Uniform synthesis interface: name -> function(logmel (T,80) numpy) -> waveform numpy (16 kHz)."""
import os
import numpy as np
from griffin_lim import griffin_lim

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(ROOT, "checkpoints", "vocoder.safetensors")


def gl_fn(n_iter):
    return lambda mel: griffin_lim(mel, n_iter=n_iter, seed=0)


def neural_fn(ckpt=CKPT):
    import mlx.core as mx
    from model import load_vocoder
    model, cfg = load_vocoder(ckpt)

    def run(mel):
        y = model(mx.array(mel[None]))
        mx.eval(y)
        return np.array(y[0])
    run.model, run.cfg = model, cfg
    return run
