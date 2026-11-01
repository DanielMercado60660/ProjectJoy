"""SAC codec I/O (Soul-AILab, ACL 2026). Runs ONLY in ../.venv-sac (torch 2.5.1, SAC's pinned transformers 4.44.1).

SAC-16k-37_5Hz: 16 kHz audio -> semantic tokens (12.5 Hz, from the GLM-4-Voice Whisper-VQ tokenizer, 16,384 entries)
+ acoustic tokens (25 Hz, single codebook, 16,384 entries) = 37.5 tokens/s, ~525 bps.
SAC's code is cloned (read-only) at ../vendor/sac_src/SAC. Its checkpoint loader uses torch.load(weights_only=False);
here every torch.load is forced to weights_only=True, so a pickled checkpoint can't run code.

  ../.venv-sac/bin/python sac_io.py roundtrip --ids ids.txt --out dir          # LJ clip ids -> dir/<id>.wav (16 kHz)
  ../.venv-sac/bin/python sac_io.py encode --out data/sac_codes.npz            # all LJ clips (27's keep rule)
  ../.venv-sac/bin/python sac_io.py decode --codes x.npz --out dir             # names/lens + sem/aco streams -> wavs
"""
import argparse, glob, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAC_ROOT = os.path.join(HERE, "..", "vendor", "sac_src", "SAC")
LJ = os.path.join(HERE, "..", "data", "ljspeech", "LJSpeech-1.1")
HF = os.path.expanduser("~/.cache/huggingface/hub")
SR = 16000
HOP_SEM = 1280  # 12.5 Hz at 16 kHz

import torch  # noqa: E402
torch.set_num_threads(4)  # lab rule: <= 4 CPU threads per job

_orig_load = torch.load


def _safe_load(*args, **kw):
    kw["weights_only"] = True
    return _orig_load(*args, **kw)


torch.load = _safe_load
sys.path.insert(0, SAC_ROOT)
import types  # noqa: E402
for _name in ("wandb",):  # training-only imports in SAC's utils; empty stubs instead of installing them
    sys.modules.setdefault(_name, types.ModuleType(_name))


def snapshot(repo):
    return glob.glob(os.path.join(HF, "models--" + repo.replace("/", "--"), "snapshots", "*"))[0]


def derived_config():
    """SAC's config minus the ERes2Net speaker encoder (a training-loss module whose checkpoint isn't released with
    the codec). Weights load with strict=False, so nothing used at inference is dropped."""
    from omegaconf import OmegaConf
    src = os.path.join(SAC_ROOT, "configs", "sac_16k_37_5.yaml")
    dst = os.path.join(HERE, "runs", "sac_cfg", "sac_16k_37_5_infer.yaml")
    cfg = OmegaConf.load(src)
    gen = (cfg.config if "config" in cfg else cfg).model.generator
    if "speaker_encoder" in gen:
        del gen["speaker_encoder"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    txt = OmegaConf.to_yaml(cfg).replace("/path/to/glm-4-voice-tokenizer", snapshot("zai-org/glm-4-voice-tokenizer"))
    open(dst, "w").write(txt)
    return dst


def device():
    return torch.device("cpu")  # MPS left off: keeps the shared GPU free for MLX training; SAC is fast enough on CPU


class SAC:
    def __init__(self):
        from models.codec.sac.utils import inference_factory, load_whisper_tokenizer
        from utils.file import load_config
        cfg_path = derived_config()
        cfg_all = load_config(cfg_path)
        self.cfg = cfg_all.get("config", cfg_all)
        self.dev = device()
        self.tok, self.fe = load_whisper_tokenizer(snapshot("zai-org/glm-4-voice-tokenizer"), self.dev)
        self.model = inference_factory(self.cfg, {"ckpt": os.path.join(snapshot("Soul-AILab/SAC-16k-37_5Hz"), "SAC-16k-37_5Hz.pt"),
                                                  "ema_load": True, "device": self.dev, "config": cfg_path})

    @torch.no_grad()
    def encode(self, wav_path):
        from models.codec.sac.utils import process_audio
        wav = process_audio(wav_path, self.cfg, HOP_SEM)
        x = torch.from_numpy(wav).float()[None, None].to(self.dev)
        sem = self.semantic_tokens(wav_path)
        out = self.model.encode(wav=x, semantic_tokens=sem, return_zq=False)
        return out["semantic_tokens"][0].cpu().numpy().astype(np.int32).reshape(-1), \
            out["acoustic_tokens"][0].cpu().numpy().astype(np.int32).reshape(-1)

    @torch.no_grad()
    def semantic_tokens(self, wav_path):
        """Copy of SAC's tools/features/extract_tokens.extract_speech_token for one utterance, without its hardcoded
        .cuda() calls (same loading, 30 s chunking, feature extraction and masking)."""
        from utils.audio import load_audio
        audio = load_audio(wav_path, sampling_rate=SR, volume_normalize=True).astype(np.float32)
        segs = [audio[t * SR:(t + 30) * SR] for t in range(0, max(1, int(np.ceil(len(audio) / SR))), 30)]
        m = self.tok
        stride = m.conv1.stride[0] * m.conv2.stride[0] * (m.config.pooling_kernel_size or 1) * self.fe.hop_length
        feats = self.fe(segs, sampling_rate=SR, return_attention_mask=True, return_tensors="pt",
                        padding="longest", pad_to_multiple_of=stride).to(self.dev)
        toks = m(**feats).quantized_token_ids
        mask = feats.attention_mask[:, ::m.conv1.stride[0] * m.conv2.stride[0]][:, ::m.config.pooling_kernel_size]
        out = []
        for i in range(len(toks)):
            out.extend(toks[i][mask[i].bool()].tolist())
        return torch.tensor(out, dtype=torch.long)[None].to(self.dev)

    @torch.no_grad()
    def decode(self, sem, aco):
        s = torch.tensor(np.asarray(sem), dtype=torch.long)[None].to(self.dev)
        a = torch.tensor(np.asarray(aco), dtype=torch.long)[None].to(self.dev)
        if a.dim() == 2 and self._aco_dim() == 3:
            a = a[:, None]
        y = self.model.decode(semantic_tokens=s, acoustic_tokens=a)["recons"]
        return y.squeeze().cpu().numpy().astype(np.float32)

    def _aco_dim(self):
        return getattr(self, "_ad", 2)


def lj_wav(cid):
    return os.path.join(LJ, "wavs", cid + ".wav")


def cmd_roundtrip(a):
    import soundfile as sf
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    m = SAC()
    os.makedirs(a.out, exist_ok=True)
    te = td = 0.0
    shapes = None
    for cid in ids:
        t0 = time.perf_counter()
        sem, aco = m.encode(lj_wav(cid))
        t1 = time.perf_counter()
        y = m.decode(sem, aco)
        t2 = time.perf_counter()
        te += t1 - t0; td += t2 - t1
        shapes = shapes or (len(sem), len(aco))
        sf.write(os.path.join(a.out, cid + ".wav"), y, SR)
    print(json.dumps({"clips": len(ids), "first_shapes": shapes, "encode_s": round(te, 2), "decode_s": round(td, 2)}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("roundtrip"); r.add_argument("--ids", required=True); r.add_argument("--out", required=True)
    a = ap.parse_args()
    {"roundtrip": cmd_roundtrip}[a.cmd](a)


if __name__ == "__main__":
    main()
