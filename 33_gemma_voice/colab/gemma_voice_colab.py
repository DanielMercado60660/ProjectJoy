# %% [markdown]
# # Gemma Voice on Colab: Gemma 3 270M + a Ming-style flow head
#
# Project 33 taught Gemma 3 270M (a text-only model) to speak on a laptop: one character per token in, one audio frame
# out every 80 ms through a small flow-matching head, Mimi's 512-d latent as the target. It reached UTMOS 3.20 on test
# (3.12 on val) against a codec ceiling of 4.28, and the write-up concluded the gap was the model, not the codec.
#
# This notebook rebuilds it in PyTorch for a Colab A100 and takes the parts of **Ming-3.0-TTS 0.5B** that look like
# they matter (from its code, `inclusionAI/Ming-3.0-tts` @ a39b806):
# - **A 64-d continuous VAE latent** (`inclusionAI/Ming-omni-tts-tokenizer-12Hz`, Apache-2.0, 44.1 kHz, 12.5 frames/s)
#   instead of Mimi's 512-d. Their paper says flow matching scales poorly with latent width.
# - **A transformer flow head** (8 blocks, 1024 wide, about 0.1B) that attends over three tokens:
#   [time + backbone state, previous frame, noisy frame]. 33 used a 28M MLP.
# - **Their sampler:** 10 Euler steps bunched at the noise end, guidance by zeroing the backbone state (scale 3).
# - Kept from 33: character input, the stop head, per-dimension latent normalization.
#
# **Run order** (each step is one cell below; all state lives in Hugging Face repos, so a disconnect costs minutes):
# 1. `SMOKE = True` on any GPU (the cheap runtime is fine): codec gate on 4 clips, encode 80 LJ clips, train 30 steps in
#    two sessions (save, resume), generate and judge 4 clips. About 10 minutes.
# 2. `SMOKE = False`, **codec gate** (`run_gate()`): Ming VAE vs Mimi round trip on 33's LJ val-100, scored by Whisper
#    large-v3-turbo + UTMOS. Pick `CODEC` from it. About 10 minutes.
# 3. **Encode data** (`encode_all()`): LJ + LibriTTS-R + Hi-Fi TTS from Hugging Face, one parquet shard at a time
#    (download, encode, save to your HF dataset repo, delete). Skips shards that are already done.
# 4. **Train** on the A100 (`train()`): resumes from the last checkpoint on HF, saves every 30 minutes, stops itself
#    before the session limit. Rerun the cell after a disconnect.
# 5. **Evaluate** (`evaluate()`): 33's LJ val-100 through the same judges, next to f3's numbers.

# %%
# ---------------------------------------------------------------- CONFIG: the only cell you should need to edit
import os, sys, math, json, time, glob, shutil, subprocess, threading, queue, random, io
LOCAL = os.environ.get("GV_LOCAL") == "1"          # laptop test mode (MPS/CPU, Mimi codec, local LJ)
IN_COLAB = "google.colab" in sys.modules

SMOKE = True                           # tiny end-to-end run first; set False for the real run
CFG = dict(
    CODEC="ming",                      # "ming" (64-d VAE, 44.1 kHz) or "mimi" (512-d, 24 kHz). The gate decides.
    BACKBONE="google/gemma-3-270m",    # base (not -it). Needs the Gemma licence accepted on your HF account.
    RUN="g270m_v1",                    # checkpoint folder name in the HF checkpoint repo
    HF_USER=None,                      # None = the account you log in with
    PUSH=True,                         # write encoded data + checkpoints to private HF repos (survives disconnects)
    # data
    SETS=["lj", "libritts_clean100", "libritts_clean360", "libritts_other500", "hifi_clean", "hifi_other"],
    LJ_REPEAT=2, MIN_S=0.8, MAX_S=20.0, ENC_BATCH_S=400.0,
    # model
    HEAD_W=1024, HEAD_DEPTH=8, HEAD_HEADS=16,
    # training
    STEPS=30000, WARMUP=500, LR=5e-5, LR_NEW=2e-4, BUDGET=24000,   # BUDGET = padded tokens per batch
    K=1, CFG_DROP=0.1, STOP_W=1.0, POS_W=5.0, T_SAMPLING="uniform", # "uniform" (Ming) or "logit_normal" (33)
    LATENT="sample",                   # "sample": draw from the VAE posterior each batch (Ming); "mean": posterior mean
    IN_NOISE=None,                     # noise on fed-back latents; None = 0 for ming, 0.2 for mimi (33)
    VAL_N=300, VAL_EVERY=1000, SAVE_MIN=30, SESSION_HOURS=10.5, LOG_EVERY=50,
    # generation / evaluation
    GEN_CFG=3.0, GEN_STEPS=10, GEN_TEMP=1.0, GEN_SCHED="ming", EVAL_N=100, MAX_FRAMES=250,
    UNITS_PER_HOUR=8.88,               # A100 rate shown in Colab's Resources panel (for ETA in compute units)
    WHISPER="openai/whisper-large-v3-turbo",
)
if SMOKE:
    CFG.update(SETS=["lj"], STEPS=30, WARMUP=5, VAL_N=16, VAL_EVERY=15, SAVE_MIN=0, EVAL_N=4, BUDGET=6000,
               LOG_EVERY=5, RUN=CFG["RUN"] + "_smoke", MAX_FRAMES=150)
    CFG["LIMIT"] = 64                  # clips per set (LJ train); LJ val keeps 16 + the eval clips
if LOCAL:
    CFG.update(CODEC=os.environ.get("GV_CODEC", "mimi"), PUSH=False, HEAD_W=256, HEAD_DEPTH=2, HEAD_HEADS=4,
               WHISPER=os.environ.get("GV_WHISPER", "openai/whisper-tiny"), BUDGET=3000)
WORK = os.environ.get("GV_WORK", "/content/gv")
os.makedirs(WORK, exist_ok=True)
print("SMOKE", SMOKE, "LOCAL", LOCAL, "WORK", WORK, json.dumps(CFG))

# %%
# ---------------------------------------------------------------- setup: packages, Ming's tokenizer code, HF login
MING_COMMIT = "a39b80660bdc5a4394d5e1d41af6dbfd9c443a54"
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6", "num2words",
                    "soundfile", "pyarrow", "huggingface_hub>=0.34"], check=True)


def setup_ming():
    """Ming's audio VAE ships as source in its TTS repo, not on PyPI. Clone at a fixed commit, copy audio_tokenizer/,
    and replace its single diffusers import with the 10-line Gaussian it uses (mean + softplus std)."""
    pkg = os.path.join(WORK, "ming_pkg")
    if os.path.exists(os.path.join(pkg, "ming_audio_tokenizer", "modeling_audio_vae.py")):
        sys.path.insert(0, pkg); return
    src = os.path.join(WORK, "ming_src")
    if not os.path.exists(src):
        subprocess.run(["git", "clone", "-q", "https://github.com/inclusionAI/Ming-3.0-tts", src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "-q", MING_COMMIT], check=True)
    os.makedirs(pkg, exist_ok=True)
    dst = os.path.join(pkg, "ming_audio_tokenizer")
    shutil.copytree(os.path.join(src, "audio_tokenizer"), dst, dirs_exist_ok=True)
    p = os.path.join(dst, "modeling_audio_vae.py")
    s = open(p).read()
    old = "from diffusers.models.autoencoders.autoencoder_oobleck import OobleckDiagonalGaussianDistribution\n"
    assert old in s, "Ming's VAE file changed; check the import line"
    s = s.replace(old, "import torch\n\nclass OobleckDiagonalGaussianDistribution:\n"
                       "    def __init__(self, parameters):\n"
                       "        self.mean, self.scale = parameters.chunk(2, dim=1)\n"
                       "        self.std = torch.nn.functional.softplus(self.scale) + 1e-4\n"
                       "    def sample(self):\n"
                       "        return self.mean + self.std * torch.randn_like(self.mean)\n")
    open(p, "w").write(s)
    sys.path.insert(0, pkg)


def hf_login():
    from huggingface_hub import login, whoami
    try:
        return whoami()["name"]
    except Exception:
        pass
    tok = None
    if IN_COLAB:
        try:
            from google.colab import userdata
            tok = userdata.get("HF_TOKEN")             # add it under the key icon (Secrets) in Colab's left bar
        except Exception:
            tok = None
    if tok:
        login(token=tok)
    else:
        from huggingface_hub import notebook_login
        notebook_login()
    return whoami()["name"]


import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly

DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
AMP = torch.bfloat16 if DEV == "cuda" else None
if DEV == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(torch.cuda.get_device_name(), round(torch.cuda.get_device_properties(0).total_memory / 2**30), "GB")
HF_USER = CFG["HF_USER"] or (hf_login() if CFG["PUSH"] or IN_COLAB else None)
DATA_REPO = f"{HF_USER}/gemma-voice-data-{CFG['CODEC']}" + ("-smoke" if SMOKE else "")
CKPT_REPO = f"{HF_USER}/gemma-voice-ckpt"
if CFG["CODEC"] == "ming":
    setup_ming()
print("device", DEV, "| data repo", DATA_REPO, "| ckpt repo", CKPT_REPO)

# %%
# ---------------------------------------------------------------- codecs: wav <-> latent frames (12.5 per second)
def resample(x, sr_in, sr_out, dev=DEV):
    """1-D float32 numpy -> numpy at sr_out (torchaudio on the GPU; scipy fallback)."""
    if sr_in == sr_out:
        return x.astype(np.float32)
    try:
        import torchaudio.functional as AF
        y = AF.resample(torch.from_numpy(x.astype(np.float32)).to(dev), sr_in, sr_out)
        return y.float().cpu().numpy()
    except Exception:
        g = math.gcd(sr_in, sr_out)
        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


class MingCodec:
    """Ming-omni-tts-tokenizer-12Hz: 44.1 kHz, 882-sample frames at 50 Hz pooled 4:1 to 12.5 Hz, 64-d VAE latent.
    encode() returns [mean | std] (2 x 64) per frame so training can sample the posterior the way Ming does."""
    name, sr, D, hop = "ming", 44100, 64, 882 * 4

    def __init__(self, dev=DEV):
        from ming_audio_tokenizer.modeling_audio_vae import AudioVAE
        from ming_audio_tokenizer.configuration_audio_vae import AudioVAEconfig
        repo = "inclusionAI/Ming-omni-tts-tokenizer-12Hz"
        cfg = AudioVAEconfig.from_pretrained(repo)
        for kw in (cfg.enc_kwargs, cfg.dec_kwargs):     # flash-attn is optional; sdpa needs no extra install
            kw["backbone"]["_attn_implementation"] = "sdpa"
            kw["backbone"]["attn_implementation"] = "sdpa"
        dt = torch.bfloat16 if dev == "cuda" else torch.float32
        self.vae = AudioVAE.from_pretrained(repo, config=cfg, torch_dtype=dt).to(dev).eval()
        self.dev, self.dt = dev, dt

    def frames(self, n):
        return math.ceil(math.ceil(n / 882) / 4)

    @torch.no_grad()
    def encode(self, wavs):
        """wavs: list of 1-D float32 at 44.1 kHz -> list of (frames, 128) float32 [mean | std].
        Batched with right padding. The encoder is causal, and hidden states past each clip's last real 882-sample
        frame are zeroed before the 4:1 pooling, which is what a one-clip call does (it pads the last patch with zeros)."""
        enc = self.vae.encoder
        L = max(len(w) for w in wavs)
        x = torch.zeros(len(wavs), L, device=self.dev, dtype=self.dt)
        for b, w in enumerate(wavs):
            x[b, :len(w)] = torch.from_numpy(w).to(self.dev, self.dt)
        h = enc.fc2(enc.fc1(enc.get_frames(x)))
        h = enc.encoder(inputs_embeds=h).last_hidden_state
        for b, w in enumerate(wavs):
            h[b, math.ceil(len(w) / 882):] = 0
        h = enc.aggregator(inputs_embeds=enc.pad_patch_insert_cls(h)).last_hidden_state
        B, _, d = h.shape
        h = h.reshape(-1, enc.patch_size + 1, d)[:, -1:].reshape(B, -1, d)
        h = enc.fc3(h).float()
        mean, scale = h.chunk(2, dim=-1)
        std = F.softplus(scale) + 1e-4
        out = torch.cat([mean, std], -1).cpu().numpy()
        return [out[b, :self.frames(len(w))] for b, w in enumerate(wavs)]

    @torch.no_grad()
    def self_check(self, x):
        """encode() re-implements Ming's Encoder.forward (to batch). On one clip it must equal Ming's own forward."""
        h, _ = self.vae.encoder(torch.from_numpy(x)[None].to(self.dev, self.dt))
        ref = h[0, :, :self.D].float().cpu().numpy()
        mine = self.encode([x])[0][:, :self.D]
        n = min(len(ref), len(mine))
        err = float(np.linalg.norm(mine[:n] - ref[:n]) / (np.linalg.norm(ref[:n]) + 1e-9))
        print(f"ming encode vs Ming's own forward: frames {len(mine)} vs {len(ref)}, rel err {err:.5f}", flush=True)
        assert len(mine) == len(ref) and err < 0.02, "MingCodec.encode disagrees with Ming's Encoder.forward"

    @torch.no_grad()
    def decode(self, lat):
        """(frames, 64) float32 -> 1-D float32 wav at 44.1 kHz (whole utterance in one pass, as Ming does)."""
        z = torch.from_numpy(np.ascontiguousarray(lat, dtype=np.float32))[None].to(self.dev, self.dt)
        y = self.vae.decode(z, past_key_values=None, use_cache=False, last_chunk=True)[0]
        return y.reshape(-1).float().cpu().numpy()


class MimiCodec:
    """Kyutai Mimi through transformers: 24 kHz, 1920-sample frames, the 512-d latent at the quantizer input
    ('space c' in 33: semantic + acoustic input projections, codebook lookup skipped). Deterministic: std = 0."""
    name, sr, D, hop = "mimi", 24000, 512, 1920

    def __init__(self, dev=DEV):
        from transformers import MimiModel
        self.m = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()
        self.dev = dev

    def frames(self, n):
        return math.ceil(n / 1920)

    @torch.no_grad()
    def encode(self, wavs):
        m, q = self.m, self.m.quantizer
        L = max(len(w) for w in wavs)
        x = torch.zeros(len(wavs), 1, L, device=self.dev)
        for b, w in enumerate(wavs):
            x[b, 0, :len(w)] = torch.from_numpy(w).to(self.dev)
        e = m.encoder(x)
        e = m.encoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)
        z = m.downsample(e)
        c = torch.cat([q.semantic_residual_vector_quantizer.input_proj(z),
                       q.acoustic_residual_vector_quantizer.input_proj(z)], 1).transpose(1, 2).float()
        out = torch.cat([c, torch.zeros_like(c)], -1).cpu().numpy()
        return [out[b, :self.frames(len(w))] for b, w in enumerate(wavs)]

    @torch.no_grad()
    def decode(self, lat):
        m, q = self.m, self.m.quantizer
        x = torch.from_numpy(np.ascontiguousarray(lat.T, dtype=np.float32))[None].to(self.dev)
        e = q.semantic_residual_vector_quantizer.output_proj(x[:, :256]) + \
            q.acoustic_residual_vector_quantizer.output_proj(x[:, 256:])
        e = m.upsample(e)
        e = m.decoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)
        return m.decoder(e)[0, 0].float().cpu().numpy()


_CODECS = {}


def get_codec(name=None):
    name = name or CFG["CODEC"]
    if name not in _CODECS:
        _CODECS[name] = {"ming": MingCodec, "mimi": MimiCodec}[name]()
    return _CODECS[name]

# %%
# ---------------------------------------------------------------- judges: Whisper large-v3-turbo + UTMOS22 (as in 33)
import re
ABBR = {"mrs": "missus", "mr": "mister", "dr": "doctor", "st": "saint", "lt": "lieutenant", "sgt": "sergeant",
        "capt": "captain", "gov": "governor", "gen": "general", "jr": "junior", "sr": "senior"}


def normalize(t):
    """33's judge.py normalizer: lowercase, numbers spelled out, punctuation dropped except in-word apostrophes."""
    try:
        from num2words import num2words
    except ImportError:                                # laptop test env only; Colab installs it
        num2words = None
    t = t.lower().replace("&", " and ").replace("%", " percent")
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    if num2words:
        t = re.sub(r"\b(1[0-9]|20)\d\d\b", lambda m: " " + num2words(int(m.group()), to="year") + " ", t)
        t = re.sub(r"\d+(\.\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())) + " ", t)
    for k, v in ABBR.items():
        t = re.sub(rf"\b{k}\.", v + " ", t)
    t = t.replace("-", " ")
    t = re.sub(r"[^a-z' ]+", " ", t)
    t = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edits(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


class Judge:
    def __init__(self):
        from transformers import pipeline
        dt = torch.float16 if DEV == "cuda" else torch.float32
        self.asr = pipeline("automatic-speech-recognition", model=CFG["WHISPER"], torch_dtype=dt, device=DEV)
        try:
            self.mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
        except Exception as e:
            print("UTMOS unavailable:", e); self.mos = None

    def score(self, items, tag=""):
        """items: list of dicts {id, wav (1-D float32), sr, ref}. Returns corpus CER/WER %, mean UTMOS + 95% CI."""
        w16 = [resample(it["wav"], it["sr"], 16000, "cpu") if len(it["wav"]) else np.zeros(1600, np.float32) for it in items]
        hyps = self.asr([{"raw": w, "sampling_rate": 16000} for w in w16], batch_size=16,
                        generate_kwargs={"language": "en", "task": "transcribe"})
        ce = cn = we = wn = 0
        rows = []
        for it, h_raw, w in zip(items, hyps, w16):
            h, r = normalize(h_raw["text"]), normalize(it["ref"])
            c, wd = edits(h, r), edits(h.split(), r.split())
            ce += c; cn += len(r); we += wd; wn += len(r.split())
            mos = float(self.mos(torch.from_numpy(w)[None].to(DEV), 16000)[0]) if self.mos is not None else float("nan")
            rows.append({"id": it["id"], "hyp": h, "ref": r, "cer": c / max(1, len(r)), "utmos": round(mos, 4)})
        v = np.array([r["utmos"] for r in rows])
        rng = np.random.default_rng(1)
        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        out = {"tag": tag, "n": len(rows), "cer": round(100 * ce / max(cn, 1), 2), "wer": round(100 * we / max(wn, 1), 2),
               "utmos": round(float(v.mean()), 3), "utmos_ci95": [round(float(np.percentile(bs, 2.5)), 3),
                                                                 round(float(np.percentile(bs, 97.5)), 3)], "clips": rows}
        print(json.dumps({k: v for k, v in out.items() if k != "clips"}), flush=True)
        return out


_JUDGE = []


def judge():
    if not _JUDGE:
        _JUDGE.append(Judge())
    return _JUDGE[0]

# %%
# ---------------------------------------------------------------- data sources
# 33's LJ val-100 (RESULTS.md section 2 and every f1-f3 val number): the first 100 of its val list.
VAL100 = """LJ041-0003 LJ041-0005 LJ041-0007 LJ041-0013 LJ041-0018 LJ041-0023 LJ041-0027 LJ041-0032 LJ041-0042 LJ041-0059
LJ041-0062 LJ041-0065 LJ041-0072 LJ041-0073 LJ041-0101 LJ041-0113 LJ041-0136 LJ041-0140 LJ041-0198 LJ042-0007 LJ042-0009
LJ042-0015 LJ042-0017 LJ042-0026 LJ042-0038 LJ042-0041 LJ042-0061 LJ042-0075 LJ042-0083 LJ042-0091 LJ042-0105 LJ042-0114
LJ042-0116 LJ042-0125 LJ042-0128 LJ042-0137 LJ042-0143 LJ042-0157 LJ042-0180 LJ042-0189 LJ042-0194 LJ042-0195 LJ042-0210
LJ042-0229 LJ042-0230 LJ042-0233 LJ042-0237 LJ042-0242 LJ042-0244 LJ043-0002 LJ043-0026 LJ043-0035 LJ043-0038 LJ043-0042
LJ043-0048 LJ043-0055 LJ043-0057 LJ043-0059 LJ043-0080 LJ043-0082 LJ043-0090 LJ043-0097 LJ043-0119 LJ043-0124 LJ043-0135
LJ043-0137 LJ043-0155 LJ043-0158 LJ043-0162 LJ043-0166 LJ043-0169 LJ043-0179 LJ043-0186 LJ044-0011 LJ044-0018 LJ044-0020
LJ044-0029 LJ044-0042 LJ044-0048 LJ044-0059 LJ044-0064 LJ044-0075 LJ044-0079 LJ044-0090 LJ044-0096 LJ044-0104 LJ044-0119
LJ044-0129 LJ044-0134 LJ044-0154 LJ044-0160 LJ044-0178 LJ044-0189 LJ044-0194 LJ044-0199 LJ044-0207 LJ044-0208 LJ044-0220
LJ044-0235 LJ044-0238""".split()
# 33's reference numbers on val-100 (mlx-whisper large-v3-turbo; the gate re-measures the real recordings here)
REF_33 = {"real recording": (0.23, 1.06, 4.40), "Qwen3-12Hz 16 codebooks": (0.32, 1.18, 4.44),
          "Mimi latent (33's target)": (0.38, 1.23, 4.28), "f3 (33's model)": (0.61, 1.62, 3.12)}


def lj_root():
    if LOCAL:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ljspeech", "LJSpeech-1.1")
    root = os.path.join(WORK, "LJSpeech-1.1")
    if not os.path.exists(os.path.join(root, "metadata.csv")):
        tb = os.path.join(WORK, "LJSpeech-1.1.tar.bz2")
        subprocess.run(["wget", "-q", "-O", tb, "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"], check=True)
        subprocess.run(["tar", "-xjf", tb, "-C", WORK], check=True)
        os.remove(tb)
    return root


def lj_rows():
    """(id, split, normalized text). 33's split: LJ001-040 train, 041-044 val, 045-050 test."""
    out = []
    for line in open(os.path.join(lj_root(), "metadata.csv"), encoding="utf-8"):
        cid, raw, norm = line.rstrip("\n").split("|")
        ch = int(cid[2:5])
        out.append((cid, "train" if ch <= 40 else ("val" if ch <= 44 else "test"), norm))
    return out


def lj_wav(cid):
    x, sr = sf.read(os.path.join(lj_root(), "wavs", cid + ".wav"), dtype="float32")
    return x, sr


# HF parquet sources: (repo, file prefix, text column, speaker column, id column)
HF_SOURCES = {
    "libritts_clean100": ("mythicinfinity/libritts_r", "data/train.clean.100/", "text_normalized", "speaker_id", "id"),
    "libritts_clean360": ("mythicinfinity/libritts_r", "data/train.clean.360/", "text_normalized", "speaker_id", "id"),
    "libritts_other500": ("mythicinfinity/libritts_r", "data/train.other.500/", "text_normalized", "speaker_id", "id"),
    "hifi_clean": ("MikhailT/hifi-tts", "data/train.clean-", "text_normalized", "speaker", "file"),
    "hifi_other": ("MikhailT/hifi-tts", "data/train.other-", "text_normalized", "speaker", "file"),
}
SPK_PREFIX = {"libritts": "libritts", "hifi": "hifi", "lj": "lj"}


def source_shards(set_name):
    """Shards of one set: list of (shard name, loader) where loader() yields (id, wav, sr, text, speaker, split)."""
    if set_name == "lj":
        rows = lj_rows()
        lim = CFG.get("LIMIT")

        def load(split):
            sel = [r for r in rows if r[1] == split]
            if lim:
                keep = set(VAL100[:CFG["EVAL_N"]]) if split == "val" else set()
                sel = [r for r in sel if r[0] in keep] + [r for r in sel if r[0] not in keep][:lim if split == "train" else 16]
            for cid, sp, text in sel:
                x, sr = lj_wav(cid)
                yield cid, x, sr, text, "lj", sp
        return [("train", lambda: load("train")), ("val", lambda: load("val"))]
    from huggingface_hub import HfApi, hf_hub_download
    repo, prefix, tcol, scol, icol = HF_SOURCES[set_name]
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))
    if CFG.get("LIMIT"):
        files = files[:1]

    def make(f):
        def load():
            import pyarrow.parquet as pq
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(WORK, "raw"))
            t = pq.read_table(p, columns=["audio", tcol, scol, icol])
            n = t.num_rows if not CFG.get("LIMIT") else min(t.num_rows, CFG["LIMIT"])
            for i in range(n):
                a = t["audio"][i].as_py()
                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
                if x.ndim > 1:
                    x = x.mean(1)
                yield str(t[icol][i].as_py()), x, sr, str(t[tcol][i].as_py()), str(t[scol][i].as_py()), "train"
            del t
            os.remove(p)                                # the user's loop: pull, encode, save, delete
        return load
    return [(os.path.basename(f).replace(".parquet", ""), make(f)) for f in files]

# %%
# ---------------------------------------------------------------- codec gate: round trip on LJ val-100
def run_gate(variants=("ming", "mimi"), n=None):
    """Encode -> decode 33's val-100 with each codec and score it with this notebook's judges, next to the real
    recordings. 'ming' uses the posterior mean, 'ming_sample' a posterior sample (what training targets look like)."""
    n = n or (4 if SMOKE else 100)
    ids = VAL100[:n]
    text = {cid: t for cid, _, t in lj_rows()}
    real = [{"id": c, "wav": lj_wav(c)[0], "sr": lj_wav(c)[1], "ref": text[c]} for c in ids]
    res = {"real": judge().score(real, "real")}
    for v in variants:
        cod = get_codec(v.split("_")[0])
        if hasattr(cod, "self_check") and not getattr(cod, "checked", False):
            cod.self_check(resample(real[0]["wav"], real[0]["sr"], cod.sr)); cod.checked = True
        items = []
        for it in real:
            x = resample(it["wav"], it["sr"], cod.sr)
            ms = cod.encode([x])[0]
            lat = ms[:, :cod.D]
            if v.endswith("_sample"):
                lat = lat + ms[:, cod.D:] * np.random.default_rng(0).standard_normal(lat.shape).astype(np.float32)
            items.append({**it, "wav": cod.decode(lat), "sr": cod.sr})
        res[v] = judge().score(items, v)
        os.makedirs(os.path.join(WORK, "gate", v), exist_ok=True)
        for it in items[:5]:
            sf.write(os.path.join(WORK, "gate", v, it["id"] + ".wav"), it["wav"], it["sr"])
    print("\nround trip on LJ val-%d (this notebook's judges)" % n)
    print(f"{'':28s} {'CER%':>6s} {'WER%':>6s} {'UTMOS':>6s}")
    for k, r in res.items():
        print(f"{k:28s} {r['cer']:6.2f} {r['wer']:6.2f} {r['utmos']:6.3f}")
    print("33's table (mlx-whisper, same clips):")
    for k, (c, w, u) in REF_33.items():
        print(f"{k:28s} {c:6.2f} {w:6.2f} {u:6.3f}")
    json.dump({k: {kk: vv for kk, vv in r.items() if kk != "clips"} for k, r in res.items()},
              open(os.path.join(WORK, "gate.json"), "w"), indent=1)
    return res

# %%
# ---------------------------------------------------------------- encode: sources -> latent shards (+ HF dataset repo)
def shard_paths(set_name, shard):
    d = os.path.join(WORK, "data", CFG["CODEC"], set_name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, shard + ".f16"), os.path.join(d, shard + ".json")


def hf_files(repo, repo_type):
    from huggingface_hub import HfApi
    try:
        return set(HfApi().list_repo_files(repo, repo_type=repo_type))
    except Exception:
        return set()


def batching_ok(cod, wavs):
    """Batched encoding pads short clips on the right; that is exact only if the encoder never looks ahead.
    Compare batched vs one-at-a-time on a few clips (Mimi through transformers fails this: 11-13% error)."""
    if getattr(cod, "batch_ok", None) is None:
        ws = sorted(wavs, key=len)[:2] + sorted(wavs, key=len)[-1:]
        if hasattr(cod, "self_check") and not getattr(cod, "checked", False):
            cod.self_check(ws[-1]); cod.checked = True
        zb = cod.encode(ws)
        err = max(float(np.linalg.norm(a[:, :cod.D] - b[:, :cod.D]) / (np.linalg.norm(b[:, :cod.D]) + 1e-9))
                  for a, b in zip(zb, [cod.encode([w])[0] for w in ws]))
        cod.batch_ok = err < 0.03
        print(f"{cod.name}: batched vs single encode rel err {err:.4f} -> " + ("batched" if cod.batch_ok else "one clip at a time"), flush=True)
    return cod.batch_ok


def encode_set(set_name, remote=None):
    """Encode every shard of one set that isn't already done locally or in the HF data repo."""
    cod = get_codec()
    remote = hf_files(DATA_REPO, "dataset") if remote is None and CFG["PUSH"] else (remote or set())
    for shard, load in source_shards(set_name):
        fp, jp = shard_paths(set_name, shard)
        rel = f"{set_name}/{shard}"
        if os.path.exists(jp):
            continue
        if rel + ".json" in remote:
            from huggingface_hub import hf_hub_download
            for ext in (".f16", ".json"):
                hf_hub_download(DATA_REPO, rel + ext, repo_type="dataset", local_dir=os.path.join(WORK, "data", CFG["CODEC"]))
            continue
        t0 = time.time()
        clips = []
        for cid, x, sr, text, spk, split in load():
            dur = len(x) / sr
            if not (CFG["MIN_S"] <= dur <= CFG["MAX_S"]) or len(text.strip()) < 3:
                continue
            clips.append((cid, resample(x, sr, cod.sr), text.strip(), f"{set_name.split('_')[0]}:{spk}", split))
        clips.sort(key=lambda c: len(c[1]))
        lats = [None] * len(clips)
        batched = batching_ok(cod, [c[1] for c in clips[:: max(1, len(clips) // 8)]]) if clips else False
        i = 0
        while i < len(clips):                           # length-sorted batches under an audio-seconds budget
            j = i
            while batched and j < len(clips) and (j - i + 1) * len(clips[j][1]) <= CFG["ENC_BATCH_S"] * cod.sr:
                j += 1
            j = max(j, i + 1)
            for k, z in enumerate(cod.encode([c[1] for c in clips[i:j]])):
                lats[i + k] = z
            i = j
        offs = np.concatenate([[0], np.cumsum([len(z) for z in lats])]).astype(np.int64)
        allz = np.concatenate(lats).astype(np.float32) if lats else np.zeros((0, 2 * cod.D), np.float32)
        allz.astype(np.float16).tofile(fp)
        mean = allz[:, :cod.D]
        meta = {"codec": cod.name, "D": cod.D, "frames": int(len(allz)), "ids": [c[0] for c in clips],
                "texts": [c[2] for c in clips], "speakers": [c[3] for c in clips], "splits": [c[4] for c in clips],
                "offs": offs.tolist(), "sum": mean.sum(0).tolist(), "sumsq": (mean.astype(np.float64) ** 2).sum(0).tolist(),
                "sum_var": (allz[:, cod.D:].astype(np.float64) ** 2).sum(0).tolist(),
                "hours": round(sum(len(c[1]) for c in clips) / cod.sr / 3600, 3)}
        json.dump(meta, open(jp, "w"))
        if CFG["PUSH"]:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=fp, path_in_repo=rel + ".f16", repo_id=DATA_REPO, repo_type="dataset")
            api.upload_file(path_or_fileobj=jp, path_in_repo=rel + ".json", repo_id=DATA_REPO, repo_type="dataset")
        print(f"{rel}: {len(clips)} clips, {meta['hours']} h, {meta['frames']} frames, {time.time() - t0:.0f} s", flush=True)


def encode_all():
    remote = hf_files(DATA_REPO, "dataset") if CFG["PUSH"] else set()
    for s in CFG["SETS"]:
        encode_set(s, remote)

# %%
# ---------------------------------------------------------------- dataset: shards -> training items
class Store:
    """All shards of the run behind one global frame index; each row is [mean (D) | std (D)] in fp16."""
    def __init__(self):
        self.parts, self.base = [], [0]

    def add(self, fp, frames, D):
        self.parts.append(np.memmap(fp, dtype=np.float16, mode="r", shape=(frames, 2 * D)) if frames else None)
        self.base.append(self.base[-1] + frames)
        return self.base[-2]

    def get(self, off, n):
        k = int(np.searchsorted(self.base, off, side="right") - 1)
        return self.parts[k][off - self.base[k]: off - self.base[k] + n]


_CHAR = {}


def text_ids(tok, text):
    """BOS + one Gemma token per character (33: subwords were too sparse at this data size)."""
    ids = [tok.bos_token_id]
    for ch in text:
        if ch not in _CHAR:
            _CHAR[ch] = tok.encode(ch, add_special_tokens=False)
        ids.extend(_CHAR[ch])
    return ids


def load_data(tok):
    """Returns train items, val items, eval items (val-100 ids), the store, speaker map and latent stats."""
    cod_D = {"ming": 64, "mimi": 512}[CFG["CODEC"]]
    metas = []
    for s in CFG["SETS"]:
        for jp in sorted(glob.glob(os.path.join(WORK, "data", CFG["CODEC"], s, "*.json"))):
            metas.append((s, jp, json.load(open(jp))))
    assert metas, "no encoded shards found: run encode_all() first"
    spk = {"lj:lj": 0}
    for s in sorted({sp for _, _, m in metas for sp in m["speakers"]} - {"lj:lj"}):
        spk[s] = len(spk)
    n = s1 = s2 = sv = 0
    store = Store()
    train, val, ev = [], [], {}
    want = set(VAL100[:CFG["EVAL_N"]])
    for s, jp, m in metas:
        base = store.add(jp[:-5] + ".f16", m["frames"], m["D"])
        rep = CFG["LJ_REPEAT"] if s == "lj" else 1
        for i, cid in enumerate(m["ids"]):
            it = (tuple(text_ids(tok, m["texts"][i])), base + m["offs"][i], m["offs"][i + 1] - m["offs"][i], spk[m["speakers"][i]])
            if m["splits"][i] == "train":
                train += [it] * rep
            elif m["splits"][i] == "val":
                val.append(it)
                if cid in want:
                    ev[cid] = (it, m["texts"][i])
        if any(sp == "train" for sp in m["splits"]):
            n += m["frames"]; s1 += np.array(m["sum"]); s2 += np.array(m["sumsq"]); sv += np.array(m["sum_var"])
    mu = s1 / n
    var = s2 / n - mu ** 2 + (sv / n if CFG["LATENT"] == "sample" else 0)
    sd = np.sqrt(np.maximum(var, 1e-8))
    rng = random.Random(0)
    rng.shuffle(val)
    hours = sum(it[2] for it in train) / 12.5 / 3600
    print(f"train {len(train)} clips ({hours:.0f} h incl. LJ x{CFG['LJ_REPEAT']}), val {len(val)}, eval {len(ev)}, "
          f"speakers {len(spk)}, D {cod_D}", flush=True)
    return train, val[:CFG["VAL_N"]], ev, store, spk, mu.astype(np.float32), sd.astype(np.float32)


def batches_for_epoch(lens, budget, seed, epoch):
    """33's length-bucketed batches under a padded-token budget; order is a pure function of (seed, epoch)."""
    rng = np.random.default_rng(seed * 1000 + epoch)
    lens = np.asarray(lens)
    order = np.argsort(lens + rng.uniform(0, 8, len(lens)))
    out, cur, L = [], [], 0
    for i in order:
        L2 = max(L, lens[i])
        if cur and L2 * (len(cur) + 1) > budget:
            out.append(cur); cur, L2 = [], lens[i]
        cur.append(int(i)); L = L2
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def make_batch(items, store, D, mu, sd, rng):
    """-> dict of tensors: text (B,Lt), tlen, lat (B,La,D) normalized (posterior-sampled if LATENT='sample'), flen, spk."""
    B = len(items)
    Lt = max(len(it[0]) for it in items)
    La = max(it[2] for it in items)
    text = np.zeros((B, Lt), np.int64)
    lat = np.zeros((B, La, D), np.float32)
    tlen, flen, spk = (np.array([len(it[0]) for it in items]), np.array([it[2] for it in items]),
                       np.array([it[3] for it in items]))
    for b, (t, off, Fn, _) in enumerate(items):
        text[b, :len(t)] = t
        z = np.asarray(store.get(off, Fn), np.float32)
        x = z[:, :D]
        if CFG["LATENT"] == "sample":
            x = x + z[:, D:] * rng.standard_normal(x.shape).astype(np.float32)
        lat[b, :Fn] = (x - mu) / sd
    t = lambda a: torch.from_numpy(a)
    return {"text": t(text), "tlen": t(tlen), "lat": t(lat), "flen": t(flen), "spk": t(spk)}

# %%
# ---------------------------------------------------------------- model: backbone + input projection + flow head + stop
class Backbone(nn.Module):
    """Thin adapter so the backbone is a config string (Gemma 3 270M now; Gemma 4 E2B or another HF causal LM later).
    embed() returns the model's own scaled token embeddings; forward() takes inputs_embeds."""
    def __init__(self, name):
        super().__init__()
        from transformers import AutoModelForCausalLM
        lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa")
        self.m = lm.model
        self.d = lm.config.hidden_size
        self.m.embed_tokens.weight.requires_grad_(False)     # 168M-row text table stays frozen (as in 33)

    def embed(self, ids):
        return self.m.embed_tokens(ids)

    def forward(self, embeds, attention_mask=None, past=None, use_cache=False, position_ids=None):
        o = self.m(inputs_embeds=embeds, attention_mask=attention_mask, past_key_values=past, use_cache=use_cache,
                   position_ids=position_ids)
        return o.last_hidden_state, o.past_key_values


def rope3(x):
    """Rotary embedding over the head's 3 token positions (0, 1, 2). x (N, H, 3, hd)."""
    hd = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))
    f = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)[:, None] * inv[None]
    cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)
    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


class HeadBlock(nn.Module):
    def __init__(self, w, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.RMSNorm(w, eps=1e-6), nn.RMSNorm(w, eps=1e-6)
        self.qkv, self.o = nn.Linear(w, 3 * w), nn.Linear(w, w)
        self.f1, self.f2 = nn.Linear(w, 4 * w), nn.Linear(4 * w, w)

    def forward(self, x):
        N, T, w = x.shape
        q, k, v = self.qkv(self.n1(x)).reshape(N, T, 3, self.h, w // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rope3(q), rope3(k), v)          # bidirectional over the 3 tokens
        x = x + self.o(a.transpose(1, 2).reshape(N, T, w))
        return x + self.f2(F.gelu(self.f1(self.n2(x)), approximate="tanh"))


class FlowHead(nn.Module):
    """Ming's DiT head: tokens [t_emb(t) + W_c h, W_x prev_frame, W_x x_t], 8 pre-RMSNorm blocks, velocity read off the
    last token. Same W_x for the previous frame and the noisy frame (Ming's x_embedder)."""
    def __init__(self, D, d_cond, w=1024, depth=8, heads=16):
        super().__init__()
        self.t1, self.t2 = nn.Linear(256, w), nn.Linear(w, w)
        self.x_in, self.c_in = nn.Linear(D, w), nn.Linear(d_cond, w)
        self.blocks = nn.ModuleList([HeadBlock(w, heads) for _ in range(depth)])
        self.norm = nn.RMSNorm(w, eps=1e-6)
        self.out = nn.Linear(w, D)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def t_emb(self, t):
        half = 128
        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
        a = 1000.0 * t.float()[:, None] * f[None]
        return self.t2(F.silu(self.t1(torch.cat([a.sin(), a.cos()], -1).to(self.t1.weight.dtype))))

    def forward(self, x, t, h, prev):
        tok = torch.stack([self.t_emb(t) + self.c_in(h), self.x_in(prev), self.x_in(x)], 1)
        for b in self.blocks:
            tok = b(tok)
        return self.out(self.norm(tok[:, 2]))


class GemmaVoice(nn.Module):
    """Sequence per clip: [BOS text chars] [AUDIO_START + spk] [frame 0 + spk] ... ; the hidden state at AUDIO_START+s
    conditions frame s through the flow head and says 'stop' at s = F."""
    def __init__(self, backbone_name, D, n_spk):
        super().__init__()
        self.bb = Backbone(backbone_name)
        d = self.bb.d
        with torch.no_grad():
            target = float(self.bb.embed(torch.arange(20000)[None]).std())
        self.D = D
        self.in_proj = nn.Linear(D, d)
        nn.init.normal_(self.in_proj.weight, std=target / math.sqrt(D)); nn.init.zeros_(self.in_proj.bias)
        self.audio_start = nn.Parameter(torch.randn(d) * target)
        self.spk = nn.Embedding(max(1, n_spk), d)
        nn.init.zeros_(self.spk.weight)
        self.head = FlowHead(D, d, CFG["HEAD_W"], CFG["HEAD_DEPTH"], CFG["HEAD_HEADS"])
        self.stop = nn.Linear(d, 1)
        nn.init.zeros_(self.stop.weight); nn.init.constant_(self.stop.bias, -4.0)

    def new_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("bb.") and p.requires_grad]

    def bb_params(self):
        return [p for n, p in self.named_parameters() if n.startswith("bb.") and p.requires_grad]


def amp():
    return torch.autocast("cuda", dtype=AMP) if AMP is not None else torch.autocast("cpu", enabled=False)


def loss_fn(model, b, in_noise):
    """Flow-matching MSE on every real frame (K draws) + stop BCE. Returns (total, flow, stop)."""
    text, tlen, lat, flen, spk = (b[k].to(DEV, non_blocking=True) for k in ("text", "tlen", "lat", "flen", "spk"))
    B, La, D = lat.shape
    fed = lat + in_noise * torch.rand(B, 1, 1, device=DEV) * torch.randn_like(lat) if in_noise > 0 else lat
    with amp():
        te = model.bb.embed(text).float()
        ae = model.in_proj(fed).float()
        sv = model.spk(spk).float()
        seqs = [torch.cat([te[i, :tlen[i]], (model.audio_start.float() + sv[i])[None], ae[i, :flen[i]] + sv[i]], 0)
                for i in range(B)]
        lens = torch.tensor([len(s) for s in seqs], device=DEV)
        emb = nn.utils.rnn.pad_sequence(seqs, batch_first=True)
        T = emb.shape[1]
        mask = (torch.arange(T, device=DEV)[None] < lens[:, None]).long()
        h, _ = model.bb(emb, attention_mask=mask)
        s = torch.arange(La + 1, device=DEV)[None]
        idx = (tlen[:, None] + s).clamp(max=T - 1)
        hs = h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1])).float()          # (B, La+1, d)
        st_logit = model.stop(hs)[..., 0].float()
        st_tgt = (s == flen[:, None]).float()
        st_mask = (s <= flen[:, None]).float()
        bce = F.binary_cross_entropy_with_logits(st_logit, st_tgt, reduction="none") * (1 + (CFG["POS_W"] - 1) * st_tgt)
        stop = (bce * st_mask).sum() / st_mask.sum()
        valid = torch.arange(La, device=DEV)[None] < flen[:, None]
        hc = hs[:, :La][valid]
        x1 = lat[valid]
        prev = torch.cat([torch.zeros_like(lat[:, :1]), lat[:, :-1]], 1)[valid]        # Ming: zeros before frame 0
        if CFG["CFG_DROP"] > 0:
            hc = hc * (torch.rand(hc.shape[0], 1, device=DEV) >= CFG["CFG_DROP"]).float()
        K = CFG["K"]
        if K > 1:
            hc, x1, prev = hc.repeat(K, 1), x1.repeat(K, 1), prev.repeat(K, 1)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=DEV) if CFG["T_SAMPLING"] == "uniform" else torch.sigmoid(torch.randn(x1.shape[0], device=DEV))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        v = model.head(xt, t, hc, prev).float()
        flow = ((v - (x1 - x0)) ** 2).mean()
    return flow + CFG["STOP_W"] * stop, flow, stop

# %%
# ---------------------------------------------------------------- generation (batched, KV cache) and evaluation
def time_grid(n, sched, dev):
    if sched == "ming" and n == 10:                    # Ming's pruned grid + sway warp t' = 1 - cos(pi t / 2)
        t = torch.tensor([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], device=dev, dtype=torch.float32) / 32
    else:
        t = torch.linspace(0, 1, n + 1, device=dev)
    return 1 - torch.cos(math.pi / 2 * t) if sched == "ming" else t


@torch.no_grad()
def sample_frames(model, h, prev, cfg=3.0, steps=10, temp=1.0, sched="ming"):
    """h (B,d), prev (B,D) normalized -> (B,D). Euler ODE; guidance v = v_u + cfg (v_c - v_u), v_u with h zeroed."""
    x = torch.randn(h.shape[0], model.D, device=h.device) * temp
    tg = time_grid(steps, sched, h.device)
    h2 = torch.cat([h, torch.zeros_like(h)])
    p2 = torch.cat([prev, prev])
    for i in range(len(tg) - 1):
        tt = tg[i].expand(2 * x.shape[0])
        with amp():
            v = model.head(torch.cat([x, x]), tt, h2, p2).float()
        vc, vu = v.chunk(2)
        x = x + (tg[i + 1] - tg[i]) * (vu + cfg * (vc - vu))
    return x


@torch.no_grad()
def generate(model, texts, spks, max_frames=250, min_frames=3, cfg=3.0, steps=10, temp=1.0, sched="ming", seed=0):
    """texts: list of token-id lists, spks: list of ints -> list of (F, D) normalized latents, info.
    Left-padded batch with explicit position ids, so every clip sees the same positions as in training."""
    model.eval()
    torch.manual_seed(seed)
    B = len(texts)
    P = max(len(t) for t in texts) + 1
    emb = torch.zeros(B, P, model.bb.d, device=DEV)
    mask = torch.zeros(B, P, dtype=torch.long, device=DEV)
    sv = model.spk(torch.tensor(spks, device=DEV)).float()
    for b, t in enumerate(texts):
        e = torch.cat([model.bb.embed(torch.tensor(t, device=DEV)[None])[0].float(), (model.audio_start.float() + sv[b])[None]])
        emb[b, P - len(e):] = e
        mask[b, P - len(e):] = 1
    pos = (mask.cumsum(1) - 1).clamp(min=0)
    with amp():
        h, past = model.bb(emb, attention_mask=mask, use_cache=True, position_ids=pos)
    h = h[:, -1].float()
    nxt = pos[:, -1:] + 1
    prev = torch.zeros(B, model.D, device=DEV)
    frames = [[] for _ in range(B)]
    done = torch.zeros(B, dtype=torch.bool, device=DEV)
    hit_max = torch.zeros(B, dtype=torch.bool, device=DEV)
    for s in range(max_frames + 1):
        p = torch.sigmoid(model.stop(h)[:, 0].float())
        if s >= min_frames:
            done |= p > 0.5
        if s == max_frames:
            hit_max = ~done
            break
        if bool(done.all()):
            break
        x = sample_frames(model, h, prev, cfg, steps, temp, sched)
        for b in range(B):
            if not done[b]:
                frames[b].append(x[b].cpu().numpy())
        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=DEV)], 1)
        with amp():
            h, past = model.bb((model.in_proj(x).float() + sv)[:, None], attention_mask=mask, past=past,
                               use_cache=True, position_ids=nxt)
        h = h[:, -1].float()
        nxt = nxt + 1
        prev = x
    lats = [np.stack(f) if f else np.zeros((0, model.D), np.float32) for f in frames]
    return lats, {"hit_max": int(hit_max.sum())}


def evaluate(model=None, tag=None, cfg=None, temp=None, steps=None, sched=None, n=None, batch=16):
    """Generate 33's LJ val-100 in LJ's voice (speaker 0), decode, judge. Compare with f3's row of REF_33."""
    model = model or STATE["model"]
    cfg = cfg if cfg is not None else CFG["GEN_CFG"]
    temp = temp if temp is not None else CFG["GEN_TEMP"]
    steps, sched = steps or CFG["GEN_STEPS"], sched or CFG["GEN_SCHED"]
    ev = STATE["eval"]
    ids = [c for c in VAL100[: n or CFG["EVAL_N"]] if c in ev]
    ids.sort(key=lambda c: len(ev[c][0][0]))            # similar lengths per batch = little padding
    cod, mu, sd = get_codec(), STATE["mu"], STATE["sd"]
    tag = tag or f"{CFG['RUN']}_s{STATE['step']}_cfg{cfg}_t{temp}_n{steps}{sched}"
    out_dir = os.path.join(WORK, "eval", tag)
    os.makedirs(out_dir, exist_ok=True)
    items, t0, n_frames, hit = [], time.time(), 0, 0
    for i in range(0, len(ids), batch):
        chunk = ids[i:i + batch]
        lats, info = generate(model, [list(ev[c][0][0]) for c in chunk], [0] * len(chunk), CFG["MAX_FRAMES"],
                              cfg=cfg, steps=steps, temp=temp, sched=sched)
        hit += info["hit_max"]
        for c, z in zip(chunk, lats):
            n_frames += len(z)
            wav = cod.decode(z * sd + mu) if len(z) else np.zeros(0, np.float32)
            sf.write(os.path.join(out_dir, c + ".wav"), wav, cod.sr)
            items.append({"id": c, "wav": wav, "sr": cod.sr, "ref": ev[c][1]})
    gen_s = time.time() - t0
    r = judge().score(items, tag)
    real_frames = sum(ev[c][0][2] for c in ids)
    r.update(len_ratio=round(n_frames / max(real_frames, 1), 3), hit_max=hit, gen_s=round(gen_s, 1))
    json.dump(r, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    c3, w3, u3 = REF_33["f3 (33's model)"]
    print(f"{tag}: CER {r['cer']} / WER {r['wer']} / UTMOS {r['utmos']}   (f3 on val: {c3} / {w3} / {u3}); "
          f"length ratio {r['len_ratio']}, runaways {hit}", flush=True)
    if CFG["PUSH"]:
        from huggingface_hub import HfApi
        HfApi().upload_folder(folder_path=out_dir, path_in_repo=f"{CFG['RUN']}/eval/{tag}", repo_id=CKPT_REPO)
    return r

# %%
# ---------------------------------------------------------------- training with checkpoint / resume (local + HF)
STATE = {}


def ckpt_dir():
    d = os.path.join(WORK, "ckpt", CFG["RUN"])
    os.makedirs(d, exist_ok=True)
    return d


def save_ckpt(model, opt, step, meta, push):
    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("bb.m.embed_tokens")}
    p = os.path.join(ckpt_dir(), "last.pt")
    torch.save({"model": sd, "opt": opt.state_dict(), "step": step, "meta": meta,
                "rng": [random.getstate(), np.random.get_state(), torch.get_rng_state()]}, p + ".tmp")
    os.replace(p + ".tmp", p)
    json.dump(meta, open(os.path.join(ckpt_dir(), "meta.json"), "w"), indent=1)
    if push and CFG["PUSH"]:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(CKPT_REPO, private=True, exist_ok=True)
        api.upload_folder(folder_path=ckpt_dir(), path_in_repo=CFG["RUN"], repo_id=CKPT_REPO, allow_patterns=["last.pt", "meta.json", "log.jsonl"])
    print(f"saved step {step}" + (" (pushed to HF)" if push and CFG["PUSH"] else ""), flush=True)


def find_ckpt():
    p = os.path.join(ckpt_dir(), "last.pt")
    if os.path.exists(p):
        return p
    if CFG["PUSH"] and f"{CFG['RUN']}/last.pt" in hf_files(CKPT_REPO, "model"):
        from huggingface_hub import hf_hub_download
        for f in ("last.pt", "meta.json", "log.jsonl"):
            try:
                hf_hub_download(CKPT_REPO, f"{CFG['RUN']}/{f}", local_dir=os.path.join(WORK, "ckpt"))
            except Exception:
                pass
        return p
    return None


def lr_lambda(step):
    w = CFG["WARMUP"]
    if step < w:
        return 0.01 + 0.99 * step / max(1, w)
    prog = min(1.0, (step - w) / max(1, CFG["STEPS"] - w))
    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def val_loss(model, val, store, D, mu, sd, in_noise):
    model.eval()
    st = (torch.get_rng_state(), torch.cuda.get_rng_state() if DEV == "cuda" else None)
    torch.manual_seed(1234)
    rng = np.random.default_rng(1234)
    tot = fl = sp = 0.0
    lens = [len(it[0]) + it[2] + 1 for it in val]
    for bi in batches_for_epoch(lens, CFG["BUDGET"], 12345, 0):
        b = make_batch([val[i] for i in bi], store, D, mu, sd, rng)
        l, f, s = loss_fn(model, b, in_noise)
        tot += float(l) * len(bi); fl += float(f) * len(bi); sp += float(s) * len(bi)
    torch.set_rng_state(st[0])
    if st[1] is not None:
        torch.cuda.set_rng_state(st[1])
    model.train()
    n = len(val)
    return tot / n, fl / n, sp / n


def train(max_steps_this_session=None):
    """Train (or resume) CFG['RUN'] until STEPS, SESSION_HOURS, or max_steps_this_session. Rerun after a disconnect."""
    from transformers import AutoTokenizer
    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(CFG["BACKBONE"])
    train_items, val, ev, store, spk, mu, sd = load_data(tok)
    D = {"ming": 64, "mimi": 512}[CFG["CODEC"]]
    in_noise = CFG["IN_NOISE"] if CFG["IN_NOISE"] is not None else (0.0 if CFG["CODEC"] == "ming" else 0.2)
    model = GemmaVoice(CFG["BACKBONE"], D, len(spk)).to(DEV)
    opt = torch.optim.AdamW([{"params": model.bb_params(), "lr": CFG["LR"], "weight_decay": 0.01},
                             {"params": model.new_params(), "lr": CFG["LR_NEW"], "weight_decay": 0.0}],
                            betas=(0.9, 0.95), fused=(DEV == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    step = 0
    meta = {"cfg": CFG, "codec": CFG["CODEC"], "D": D, "n_spk": len(spk), "lat_mean": mu.tolist(), "lat_std": sd.tolist()}
    p = find_ckpt()
    if p:
        ck = torch.load(p, map_location=DEV, weights_only=False)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        assert not unexpected and all(k.startswith("bb.m.embed_tokens") for k in missing), (missing, unexpected)
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        random.setstate(ck["rng"][0]); np.random.set_state(ck["rng"][1]); torch.set_rng_state(ck["rng"][2].cpu())
        mu, sd = np.array(ck["meta"]["lat_mean"], np.float32), np.array(ck["meta"]["lat_std"], np.float32)
        meta.update(lat_mean=ck["meta"]["lat_mean"], lat_std=ck["meta"]["lat_std"])
        print(f"resumed {CFG['RUN']} at step {step}", flush=True)
    sched.last_epoch = step - 1
    sched.step()
    n_bb = sum(p.numel() for p in model.bb_params()); n_new = sum(p.numel() for p in model.new_params())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"trainable: backbone {n_bb / 1e6:.1f}M, new {n_new / 1e6:.1f}M (flow head {n_head / 1e6:.1f}M)", flush=True)
    STATE.update(model=model, eval=ev, mu=mu, sd=sd, step=step, spk=spk)
    log = open(os.path.join(ckpt_dir(), "log.jsonl"), "a")
    lens = [len(it[0]) + it[2] + 1 for it in train_items]
    nb = len(batches_for_epoch(lens, CFG["BUDGET"], 0, 0))
    rng = np.random.default_rng(step + 7)
    last_save = time.time()
    stop_at = CFG["STEPS"] if max_steps_this_session is None else min(CFG["STEPS"], step + max_steps_this_session)

    def producer(q, start_step):                       # one batch ahead on a CPU thread
        s = start_step
        while s < stop_at:
            ep, k = divmod(s, nb)
            order = batches_for_epoch(lens, CFG["BUDGET"], 0, ep)
            for bi in order[k:]:
                q.put(make_batch([train_items[i] for i in bi], store, D, mu, sd, rng))
                s += 1
                if s >= stop_at:
                    break
        q.put(None)
    q = queue.Queue(maxsize=3)
    threading.Thread(target=producer, args=(q, step), daemon=True).start()
    if step == 0:
        vl = val_loss(model, val, store, D, mu, sd, in_noise)
        print(json.dumps({"step": 0, "val": [round(x, 4) for x in vl]}), flush=True)
    model.train()
    tw, nt, t_log = time.time(), 0, time.time()
    while True:
        b = q.get()
        if b is None:
            break
        loss, flow, stop = loss_fn(model, b, in_noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        step += 1
        STATE["step"] = step
        nt += int(b["text"].shape[0] * (b["text"].shape[1] + b["lat"].shape[1] + 1))
        if step % CFG["LOG_EVERY"] == 0:
            dt = time.time() - tw
            sps = CFG["LOG_EVERY"] / max(time.time() - t_log, 1e-6)
            eta_h = (CFG["STEPS"] - step) / max(sps, 1e-6) / 3600
            rec = {"step": step, "loss": round(float(loss), 4), "flow": round(float(flow), 4), "stop": round(float(stop), 4),
                   "gnorm": round(float(gn), 3), "lr": round(sched.get_last_lr()[0], 7), "tok_s": round(nt / dt),
                   "eta_h": round(eta_h, 2), "eta_units": round(eta_h * CFG["UNITS_PER_HOUR"], 1),
                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if DEV == "cuda" else None}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
            tw, nt, t_log = time.time(), 0, time.time()
        if step % CFG["VAL_EVERY"] == 0:
            vl = val_loss(model, val, store, D, mu, sd, in_noise)
            rec = {"step": step, "val": [round(x, 4) for x in vl]}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush()
        out_of_time = (time.time() - t_start) / 3600 > CFG["SESSION_HOURS"]
        timed = CFG["SAVE_MIN"] > 0 and (time.time() - last_save) / 60 > CFG["SAVE_MIN"]   # 0 = only at the end
        if timed or step >= stop_at or out_of_time:
            save_ckpt(model, opt, step, meta, push=True)
            last_save = time.time()
        if step >= stop_at or out_of_time:
            break
    print(f"stopped at step {step} of {CFG['STEPS']} ({(time.time() - t_start) / 60:.1f} min this session)", flush=True)
    return model

# %% [markdown]
# ## Run
# Run the cells above once per session, then the steps you need below. `SMOKE = True` first.

# %%
# 1. codec gate (SMOKE: 4 clips). With SMOKE = False: 100 clips, ~10 min. Pick CFG["CODEC"] from the table.
gate = run_gate(("ming", "ming_sample", "mimi") if not LOCAL else ("mimi",))

# %%
# 2. encode the training data for CFG["CODEC"] (skips shards already in your HF data repo)
encode_all()

# %%
# 3. train. SMOKE runs 15 steps, "disconnects", resumes and finishes 30, to prove resume works.
if SMOKE:
    train(max_steps_this_session=15)
    model = train()
else:
    model = train()

# %%
# 4. evaluate on 33's LJ val-100 (SMOKE: 4 clips) and listen to a few
res = evaluate()
if IN_COLAB:
    from IPython.display import Audio, display
    for row in res["clips"][:3]:
        print(row["id"], "|", row["ref"]); print("heard:", row["hyp"])
        display(Audio(os.path.join(WORK, "eval", res["tag"], row["id"] + ".wav")))
