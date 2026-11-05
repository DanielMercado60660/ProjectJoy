"""Round 6 discriminators (MLX) and GAN losses.

MPD (HiFi-GAN): periods 2,3,5,7,11; waveform reshaped to (T/p, p); Conv2d (5,1) stride (3,1) stack + (3,1) post conv.
MRD (UnivNet / Vocos): log |STFT| at (512,128), (1024,256), (2048,512) treated as a 2-D image; Conv2d (3,9) stack with
frequency stride 2 + (3,3) post conv.
All convs are weight-normalised by hand (weight = g * v / ||v||; MLX has no weight_norm), as in HiFi-GAN.
Losses (Vocos recipe): hinge for D and G, L1 feature matching averaged over layers, + 45 x log-mel L1 for G.
"""
import numpy as np
import mlx.core as mx
import mlx.nn as nn

LRELU = 0.1


class WNConv2d(nn.Module):
    def __init__(self, cin, cout, k, stride=(1, 1), padding=(0, 0)):
        super().__init__()
        scale = 1.0 / np.sqrt(cin * k[0] * k[1])
        self.v = mx.random.uniform(-scale, scale, (cout, k[0], k[1], cin))
        self.g = mx.sqrt(mx.sum(self.v ** 2, axis=(1, 2, 3), keepdims=True))
        self.bias = mx.zeros((cout,))
        self.stride, self.padding = stride, padding

    def __call__(self, x):
        w = self.g * self.v / mx.sqrt(mx.sum(self.v ** 2, axis=(1, 2, 3), keepdims=True) + 1e-12)
        return mx.conv2d(x, w, stride=self.stride, padding=self.padding) + self.bias


class PeriodD(nn.Module):
    def __init__(self, p, ch):
        super().__init__()
        self.p = p
        cin, self.convs = 1, []
        for i, c in enumerate(ch):
            self.convs.append(WNConv2d(cin, c, (5, 1), stride=(3, 1) if i < len(ch) - 1 else (1, 1), padding=(2, 0)))
            cin = c
        self.post = WNConv2d(cin, 1, (3, 1), padding=(1, 0))

    def __call__(self, y):
        B, T = y.shape
        if T % self.p:
            y = mx.pad(y, [(0, 0), (0, self.p - T % self.p)])
        x = y.reshape(B, -1, self.p, 1)
        feats = []
        for c in self.convs:
            x = nn.leaky_relu(c(x), LRELU); feats.append(x)
        x = self.post(x); feats.append(x)
        return x.reshape(B, -1), feats


_IDX = {}


def stft_mag(x, n_fft, hop):
    B, L = x.shape
    xp = mx.pad(x, [(0, 0), (n_fft // 2, n_fft // 2)])
    n = 1 + L // hop
    key = (n_fft, hop, n)
    if key not in _IDX:
        idx = np.arange(n_fft)[None, :] + hop * np.arange(n)[:, None]
        _IDX[key] = (mx.array(idx.astype(np.int32)), mx.array(np.hanning(n_fft + 1)[:-1].astype(np.float32)))
    idx, win = _IDX[key]
    S = mx.fft.rfft(xp[:, idx] * win, axis=-1)
    return mx.sqrt(mx.real(S) ** 2 + mx.imag(S) ** 2 + 1e-9)


class ResD(nn.Module):
    def __init__(self, n_fft, hop, ch):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.convs = [WNConv2d(1, ch, (3, 9), padding=(1, 4))] + \
                     [WNConv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4)) for _ in range(3)] + \
                     [WNConv2d(ch, ch, (3, 3), padding=(1, 1))]
        self.post = WNConv2d(ch, 1, (3, 3), padding=(1, 1))

    def __call__(self, y):
        x = mx.log(stft_mag(y, self.n_fft, self.hop) + 1e-5)[..., None]  # (B, frames, bins, 1)
        feats = []
        for c in self.convs:
            x = nn.leaky_relu(c(x), LRELU); feats.append(x)
        x = self.post(x); feats.append(x)
        return x.reshape(x.shape[0], -1), feats


MPD_CH = {"full": (32, 128, 512, 1024, 1024), "mid": (32, 128, 256, 512, 512), "small": (32, 64, 128, 256, 256)}


class Discriminators(nn.Module):
    def __init__(self, mpd="mid", mrd_ch=32):
        super().__init__()
        self.cfg = {"mpd": mpd, "mrd_ch": mrd_ch}
        self.mpd = [PeriodD(p, MPD_CH[mpd]) for p in (2, 3, 5, 7, 11)]
        self.mrd = [ResD(n, h, mrd_ch) for n, h in ((512, 128), (1024, 256), (2048, 512))]

    def __call__(self, y):
        return [d(y) for d in self.mpd], [d(y) for d in self.mrd]


def d_hinge(real, fake):
    return sum(mx.mean(nn.relu(1 - r)) + mx.mean(nn.relu(1 + f)) for (r, _), (f, _) in zip(real, fake)) / len(real)


def g_hinge(fake):
    return sum(mx.mean(nn.relu(1 - f)) for f, _ in fake) / len(fake)


def feat_match(real, fake):
    tot, n = 0.0, 0
    for (_, rf), (_, ff) in zip(real, fake):
        for a, b in zip(rf, ff):
            tot = tot + mx.mean(mx.abs(mx.stop_gradient(a) - b)); n += 1
    return tot / len(real)  # Vocos: sum over layers, mean over sub-discriminators
