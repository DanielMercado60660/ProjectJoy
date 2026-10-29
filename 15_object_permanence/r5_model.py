"""MLX-only adapter around the existing CNN/GRU recipe; imported only by runner."""
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from model import Tracker, gaussian_nll, to_pixels


class CueTracker(Tracker):
    def __init__(self, components=1, enc_dim=512, hid=256, layers=1):
        super().__init__(K=3, enc_dim=enc_dim, hid=hid, layers=layers)
        self.components = components
        # Static cue: 3*(x,y,present), original wall geometry: 4, revealed wall: 2.
        self.cue_projection = nn.Linear(13, enc_dim)
        self.peek_projection = nn.Linear(2, enc_dim, bias=False)
        self.cue_projection.weight = mx.zeros_like(self.cue_projection.weight)
        self.cue_projection.bias = mx.zeros_like(self.cue_projection.bias)
        self.peek_projection.weight = mx.zeros_like(self.peek_projection.weight)
        self.head2 = nn.Linear(256, 3*components*6)

    def features(self, frames, cue, present, occ, flags):
        assert frames.ndim == 5 and frames.shape[-3:] == (64, 64, 3)
        assert cue.shape == (frames.shape[0], 3, 2)
        cue = (cue-32)/32 * present[..., None]
        geometry = mx.where(occ >= 0, (occ-32)/32, -2.)
        static = mx.concatenate([cue.reshape(len(cue), -1), present,
                                 geometry.reshape(len(cue), -1)], -1)
        return self.encode(frames) + self.cue_projection(static)[:, None, :] + self.peek_projection(flags)

    def __call__(self, frames, cue, present, occ, flags):
        e = self.features(frames, cue, present, occ, flags)
        h = e
        for g in self.grus:
            h = g(h)
        z = nn.relu(self.head1(mx.concatenate([h, e], -1)))
        return self.head2(z).reshape(frames.shape[0], frames.shape[1], 3, self.components, 6)

    def step(self, frame, cue, present, occ, flags, hidden=None):
        e = self.features(frame[:, None], cue, present, occ, flags[:, None])
        h = e
        next_hidden = []
        for i, g in enumerate(self.grus):
            h = g(h, None if hidden is None else hidden[i])
            next_hidden.append(h[:, -1])
        z = nn.relu(self.head1(mx.concatenate([h, e], -1)))
        return self.head2(z).reshape(frame.shape[0], 3, self.components, 6), next_hidden

    def warm_start(self, weights):
        # Both models get exactly the same trunk/head weights and zero cue additions.
        trunk = [(k, mx.array(v)) for k, v in weights.items() if not k.startswith('head2.')]
        self.load_weights(trunk, strict=False)
        w = weights['head2.weight'].reshape(3, 5, 256)
        b = weights['head2.bias'].reshape(3, 5)
        nw = np.zeros((3, self.components, 6, 256), np.float32)
        nb = np.zeros((3, self.components, 6), np.float32)
        nw[:, :, :5] = w[:, None]
        nb[:, :, :5] = b[:, None]
        # Small symmetric 0.5px offsets break component symmetry; all pretrained
        # means/covariances are otherwise identical. K=1 is exactly the old head.
        if self.components > 1:
            nb[:, :, 0] += np.linspace(-.5, .5, self.components)[None, :]/32
        self.head2.weight = mx.array(nw.reshape(-1, 256))
        self.head2.bias = mx.array(nb.reshape(-1))


def mixture_nll(raw, target):
    component = gaussian_nll(raw[..., :5], target[..., None, :])
    logits = raw[..., 5]
    logw = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return -mx.logsumexp(logw-component, axis=-1)


def training_loss(model, frames, cue, present, occ, flags, target, aux):
    raw = model(frames, cue, present, occ, flags)
    weights = mx.softmax(raw[..., 5], axis=-1)
    mean = (weights[..., None]*raw[..., :2]).sum(-2)
    var = (weights[..., None]*(mx.exp(2*mx.clip(raw[..., 2:4], -5.5, 1.))
                               + (raw[..., :2]-mean[..., None, :])**2)).sum(-2)
    # Existing beta=.5 recipe generalized to the mixture's marginal variances.
    beta_weight = mx.stop_gradient(mx.maximum(var[..., 0]*var[..., 1], 1e-12)**.25)
    loss = mixture_nll(raw, target)*beta_weight + aux*100*((mean-target)**2).sum(-1)
    mask = mx.broadcast_to(present[:, None], loss.shape)
    per_slot = (loss*mask).sum((0, 1)) / mx.maximum(mask.sum((0, 1)), 1.)
    return per_slot.mean()


def distribution(raw):
    raw = np.asarray(raw, np.float64)
    logits = raw[..., 5]-raw[..., 5].max(-1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(-1, keepdims=True)
    mean, cov = to_pixels(raw[..., :5])
    return weights, mean, cov
