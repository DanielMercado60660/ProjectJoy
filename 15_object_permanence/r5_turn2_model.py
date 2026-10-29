"""Per-query shared recurrent tracker: cue binding without an output-slot convention.

The CNN runs once per video. GRU/head weights are shared across the three queries.
A query's own cue enters every step, initial hidden state and readout. Means are
cue-relative displacements; frame-0 means are anchored to the observed noisy cue.
Permuting (cue,present) must permute outputs at every frame, by construction.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from model import Tracker


class BoundCueTracker(Tracker):
    def __init__(self, components=1, enc_dim=512, hid=256, layers=1):
        super().__init__(K=3, enc_dim=enc_dim, hid=hid, layers=layers)
        self.components = components
        self.cue_input = nn.Linear(8, enc_dim)
        self.cue_initial = [nn.Linear(8, hid) for _ in range(layers)]
        self.cue_head = nn.Linear(8, 256)
        self.peek_projection = nn.Linear(2, enc_dim, bias=False)
        self.head2 = nn.Linear(256, components*6)  # SAME readout for every query

    def query_features(self, cue, present, occ):
        own = (cue-32)/32 * present[..., None]
        geometry = mx.where(occ >= 0, (occ-32)/32, -2.).reshape(len(cue), 4)
        count = mx.broadcast_to(present.sum(1)[:, None, None]/3, (len(cue), 3, 1))
        return mx.concatenate([own, present[..., None], count,
                               mx.broadcast_to(geometry[:, None], (len(cue), 3, 4))], -1)

    def _forward(self, frames, cue, present, occ, flags, state=None):
        B, T = frames.shape[:2]
        assert frames.shape[-3:] == (64, 64, 3) and cue.shape == (B, 3, 2)
        assert present.shape == (B, 3) and flags.shape == (B, T, 2)
        q = self.query_features(cue, present, occ)
        # The visual encoder is shared, not re-run separately for each ball.
        visual = self.encode(frames) + self.peek_projection(flags)
        e = (visual[:, None] + self.cue_input(q)[:, :, None]).reshape(B*3, T, -1)
        h = e
        next_h = []
        for i, g in enumerate(self.grus):
            initial = (mx.tanh(self.cue_initial[i](q)).reshape(B*3, -1)
                       if state is None else state['layers'][i])
            h = g(h, initial)
            next_h.append(h[:, -1])
        cue_head = self.cue_head(q).reshape(B*3, 1, 256)
        z = nn.relu(self.head1(mx.concatenate([h, e], -1)) + cue_head)
        raw = self.head2(z).reshape(B, 3, T, self.components, 6).transpose(0, 2, 1, 3, 4)
        start = 0 if state is None else state['time']
        moving = mx.array((np.arange(start, start+T) > 0).astype(np.float32))[None, :, None, None, None]
        anchor = ((cue-32)/32 * present[..., None])[:, None, :, None, :]
        raw = mx.concatenate([anchor + moving*raw[..., :2], raw[..., 2:]], -1)
        return raw, dict(layers=next_h, time=start+T)

    def __call__(self, frames, cue, present, occ, flags):
        return self._forward(frames, cue, present, occ, flags)[0]

    def step(self, frame, cue, present, occ, flags, hidden=None):
        raw, state = self._forward(frame[:, None], cue, present, occ, flags[:, None], hidden)
        return raw[:, 0], state

    def warm_start(self, weights):
        # Reuse the identical original CNN, GRU and head1 in both models. The old
        # slot-specific absolute-position head is deliberately replaced by a new
        # shared displacement head with no inherited slot-specific parameters.
        self.load_weights([(k, mx.array(v)) for k, v in weights.items()
                           if not k.startswith('head2.')], strict=False)
        rng = np.random.default_rng(52024)
        for layer, scale in [(self.cue_input, .1), *[(v, .2) for v in self.cue_initial], (self.cue_head, .1)]:
            layer.weight = mx.array(rng.normal(0, scale, layer.weight.shape).astype(np.float32))
            layer.bias = mx.zeros_like(layer.bias)
        self.peek_projection.weight = mx.zeros_like(self.peek_projection.weight)
        self.head2.weight = mx.zeros_like(self.head2.weight)
        bias = np.zeros((self.components, 6), np.float32)
        bias[:, 2:4] = np.log(3/32)  # modest initial uncertainty, pixels
        if self.components > 1:
            bias[:, 0] = np.linspace(-.5, .5, self.components)/32
        self.head2.bias = mx.array(bias.reshape(-1))
