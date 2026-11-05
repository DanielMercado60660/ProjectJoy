# %% [markdown]
# # Gemma Voice: 2,500-hour first training milestone
# 
# This variant extends the A100 notebook with Emilia-YODAS English. It prepares
# at least 2,500 hours of filtered, unique training clips; validation, test and
# LJ repeats are excluded from the target. Source revision and filtering policy
# are pinned. Original LJ / LibriTTS-R / Hi-Fi data are retained.
# 
# The original smoke and full codec gate were completed on 2026-11-03. Full-model
# training on this larger corpus has **not** been completed. Run setup/definitions,
# then the preparation cell. It runs continuously until the data target and audit
# are complete, saving chunks to your private HF repo. Colab can still end the
# runtime; rerun preparation to resume. The user authorized removing the 40-minute
# limit. A later expansion to 10,000+ hours reuses the completed encoding.
# 
# Training uses a verified fresh-process checkpoint recovery path. At the current
# batch budget one epoch is 45,143 updates; the learning-rate horizon is three
# epochs (135,431 updates). The training cell restores or verifies/calibrates the
# run, then continues through the first full epoch with saved speech evaluations.
# Review its generated speech before extending to epochs two and three.

# %%
# CONFIG: full dataset; batch budget controls GPU memory, not total data.
import os, sys, math, json, time, glob, shutil, subprocess, threading, queue, random, io
LOCAL = os.environ.get("GV_LOCAL") == "1"
IN_COLAB = "google.colab" in sys.modules
SMOKE = False
CFG = dict(
    CODEC="ming", BACKBONE="google/gemma-3-270m", RUN="g270m_v1", HF_USER=None, PUSH=True,
    SETS=["lj", "libritts_clean100", "libritts_clean360", "libritts_other500", "hifi_clean", "hifi_other"],
    LJ_REPEAT=2, MIN_S=0.8, MAX_S=20.0, ENC_BATCH_S=400.0,
    HEAD_W=1024, HEAD_DEPTH=8, HEAD_HEADS=16,
    STEPS=30000, WARMUP=500, LR=5e-5, LR_NEW=2e-4, BUDGET=6000,
    K=1, CFG_DROP=0.1, STOP_W=1.0, POS_W=5.0, T_SAMPLING="uniform",
    LATENT="sample", IN_NOISE=None,
    VAL_N=300, VAL_EVERY=1000, SAVE_MIN=10, SESSION_HOURS=0.25, LOG_EVERY=50,
    GEN_CFG=3.0, GEN_STEPS=10, GEN_TEMP=1.0, GEN_SCHED="ming", EVAL_N=100, MAX_FRAMES=250,
    UNITS_PER_HOUR=6.73, WHISPER="openai/whisper-large-v3-turbo",
)
# STEPS is provisional; first full-data training is limited to 100 steps / 15 minutes.
if SMOKE:
    CFG.update(SETS=["lj"], STEPS=30, WARMUP=5, VAL_N=16, VAL_EVERY=15, SAVE_MIN=0,
               EVAL_N=4, BUDGET=6000, LOG_EVERY=5, RUN=CFG["RUN"] + "_smoke", MAX_FRAMES=150)
    CFG["LIMIT"] = 64
if LOCAL:
    CFG.update(CODEC=os.environ.get("GV_CODEC", "mimi"), PUSH=False, HEAD_W=256, HEAD_DEPTH=2,
               HEAD_HEADS=4, WHISPER=os.environ.get("GV_WHISPER", "openai/whisper-tiny"), BUDGET=3000)
WORK = os.environ.get("GV_WORK", "/content/gv_smoke" if SMOKE else "/content/gv_full_ming_v1")
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
# 
# The full codec gate already passed its technical checks (results in the run
# folder). On reconnect run configuration/setup/definitions, then rerun preparation.
# Completed chunks are restored from HF. Do not start model training on a partial
# dataset. The next cell checks the target and all original corpus shards first.

# %%
# Full-data pilot: preserve smoke files, reuse only raw LJ audio and installed codec code.
import gc
DATA_REPO = f"{HF_USER}/gemma-voice-data-{CFG['CODEC']}" + ("-smoke" if SMOKE else "")
CKPT_REPO = f"{HF_USER}/gemma-voice-ckpt"
for asset in ("LJSpeech-1.1", "ming_pkg"):
    old = os.path.join("/content/gv", asset)
    new = os.path.join(WORK, asset)
    if os.path.isdir(old) and not os.path.lexists(new):
        os.symlink(old, new, target_is_directory=True)
STATE.clear()
model = None
_ = gc.collect()
if DEV == "cuda":
    torch.cuda.empty_cache()
print("Full gate", WORK, "|", DATA_REPO, "|", CKPT_REPO, flush=True)
with torch.no_grad():
    gate = run_gate(("ming", "ming_sample", "mimi"), n=100)
os.makedirs(os.path.join(WORK, "gate"), exist_ok=True)
with open(os.path.join(WORK, "gate", "per_clip.json"), "w") as f:
    json.dump(gate, f, indent=1)
if CFG["PUSH"]:
    from huggingface_hub import HfApi
    HfApi().upload_folder(folder_path=os.path.join(WORK, "gate"),
                         path_in_repo=f"{CFG['RUN']}/gate", repo_id=CKPT_REPO)
print("Full 100-clip gate saved.", flush=True)

# %%
"""Paste/run after the base notebook definitions. Encoding only; never starts training.

Emilia-YODAS EN -> bounded Ming posterior chunks, private HF backups, measured
unique training hours. Runs to the agreed target; rerun after a Colab disconnect.
"""
import collections
import hashlib
import pathlib
import tarfile
import gc
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

EMILIA_SET = "emilia_yodas"
EMILIA_REPO = "amphion/Emilia-Dataset"
EMILIA_REV = "d7f2f7340a6385696f3766c8049fa920a4707c07"
BASE_SETS = ["lj", "libritts_clean100", "libritts_clean360", "libritts_other500", "hifi_clean", "hifi_other"]
EMILIA_POLICY = dict(version=1, repo=EMILIA_REPO, revision=EMILIA_REV,
    prefix="Emilia-YODAS/EN/", language="en", min_s=0.8, max_s=20.0,
    min_dnsmos=3.3, min_chars=3, max_chars=1000, chunk_clips=512,
    seed=20261103, split="sha256(source video): 1% test, 1% val, 98% train",
    codec="ming", codec_code=MING_COMMIT)
POLICY_ID = hashlib.sha256(json.dumps(EMILIA_POLICY, sort_keys=True).encode()).hexdigest()
# User chose an intermediate listening/training milestone on 2026-11-03.
# Increasing this later reuses all completed encoded chunks.
TARGET_TRAIN_HOURS = 2500.0
if "_source_shards_original" not in globals():
    _source_shards_original = source_shards


def atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "w") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(path + ".partial", path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def pair_ok(jp, check_hash=False):
    try:
        with open(jp) as f:
            m = json.load(f)
        fp = jp[:-5] + ".f16"
        assert os.path.getsize(fp) == m["frames"] * 2 * m["D"] * 2
        n = len(m["ids"])
        assert all(len(m[k]) == n for k in ("texts", "speakers", "splits"))
        assert len(m["offs"]) == n + 1 and m["offs"][0] == 0 and m["offs"][-1] == m["frames"]
        assert all(a <= b for a, b in zip(m["offs"], m["offs"][1:]))
        if check_hash and "sha256" in m:
            assert sha256_file(fp) == m["sha256"]
        return m
    except (OSError, ValueError, KeyError, AssertionError):
        return None


def emilia_split(cid):
    # Source video IDs can contain underscores, so split from the RIGHT.
    group = cid.rsplit("_W", 1)[0]
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10000
    return "test" if bucket < 100 else "val" if bucket < 200 else "train"


def metadata_rejection(m):
    try:
        dur, score = float(m["duration"]), float(m["dnsmos"])
        text = str(m["text"]).strip()
        if m["language"].lower() != "en":
            return "language"
        if not math.isfinite(dur) or not 0.8 <= dur <= 20:
            return "duration"
        if not math.isfinite(score) or score < 3.3:
            return "dnsmos"
        if not 3 <= len(text) <= 1000:
            return "text"
        if not m.get("speaker"):
            return "speaker"
    except (KeyError, TypeError, ValueError):
        return "metadata"
    return None


def encoded_inventory():
    summary = {}
    for s in CFG["SETS"]:
        by_split = collections.defaultdict(float)
        n = clips = 0
        for jp in sorted(glob.glob(os.path.join(WORK, "data", "ming", s, "*.json"))):
            m = pair_ok(jp)
            assert m is not None, f"Incomplete encoded pair: {jp}"
            splits = set(m["splits"])
            assert len(splits) <= 1, f"Mixed split stats are unsafe: {jp}"
            if s == EMILIA_SET:
                assert m.get("policy_id") == POLICY_ID, f"Encoding policy changed: {jp}"
            if splits:
                by_split[next(iter(splits))] += m["hours"]
            n += 1
            clips += len(m["ids"])
        summary[s] = dict(shards=n, clips=clips, hours=dict(by_split))
    return summary


def training_hours(summary):
    return sum(s["hours"].get("train", 0.0) for s in summary.values())


def commit_paths(api, rels, remote, message):
    root = os.path.join(WORK, "data", "ming")
    missing = [r for r in rels if r not in remote]
    if missing:
        api.create_commit(repo_id=DATA_REPO, repo_type="dataset", commit_message=message,
            operations=[CommitOperationAdd(path_in_repo=r, path_or_fileobj=os.path.join(root, r)) for r in missing])
        remote.update(missing)


def restore_and_reconcile(api, remote):
    """Remote JSON+audio pairs commit together; repair old base uploads interrupted between files."""
    root = os.path.join(WORK, "data", "ming")
    restored = 0
    for rel in sorted(remote):
        if not rel.endswith(".json") or rel.split("/")[0] not in CFG["SETS"]:
            continue
        fp = rel[:-5] + ".f16"
        assert fp in remote, f"Remote metadata has no audio: {rel}"
        jp = os.path.join(root, rel)
        if not pair_ok(jp):
            for r in (fp, rel):
                hf_hub_download(DATA_REPO, r, repo_type="dataset", local_dir=root)
            assert pair_ok(jp, check_hash=True), f"Restored data failed verification: {rel}"
            restored += 1
    pending = []
    for s in CFG["SETS"]:
        for jp in sorted(glob.glob(os.path.join(root, s, "*.json"))):
            if pair_ok(jp):
                rel = os.path.relpath(jp, root)
                pending.extend([rel[:-5] + ".f16", rel])
    for i in range(0, len(pending), 100):
        commit_paths(api, pending[i:i+100], remote, "Resume: persist complete encoded pairs")
    print("Restore verified:", restored, "pairs downloaded; complete local pairs backed up", flush=True)


def emilia_id_owners():
    owners = {}
    for jp in sorted(glob.glob(os.path.join(WORK,"data","ming",EMILIA_SET,"*.json"))):
        m = pair_ok(jp)
        assert m and m.get("policy_id") == POLICY_ID
        for cid in m["ids"]:
            assert cid not in owners, f"Duplicate encoded clip: {cid}"
            owners[cid] = m["source_archive"]
    return owners


def encode_emilia_chunk(tar, rows, shard, archive, split):
    fp, jp = shard_paths(EMILIA_SET, shard)
    prior = pair_ok(jp)
    if prior:
        assert prior.get("policy_id") == POLICY_ID
        return False
    cod = get_codec()
    clips, rejected = [], collections.Counter()
    for cid, meta, audio_member in rows:
        try:
            x, sr = sf.read(io.BytesIO(tar.extractfile(audio_member).read()), dtype="float32")
        except (RuntimeError, ValueError, OSError):
            rejected["decode"] += 1
            continue
        if x.ndim == 2:
            x = x.mean(axis=1)
        dur = len(x) / sr
        if not 0.8 <= dur <= 20.0 or abs(dur - float(meta["duration"])) > 0.25:
            rejected["decoded_duration"] += 1
            continue
        if not np.isfinite(x).all() or np.max(np.abs(x), initial=0) < 1e-5:
            rejected["invalid_audio"] += 1
            continue
        clips.append((cid, resample(x, sr, cod.sr), str(meta["text"]).strip(),
                      "emilia:" + str(meta["speaker"]), dur))
    clips.sort(key=lambda c: (len(c[1]), c[0]))
    lats = []
    batched = batching_ok(cod, [c[1] for c in clips[:2] + clips[-1:]]) if clips else False
    i = 0
    with torch.no_grad():
        while i < len(clips):
            j = i
            while batched and j < len(clips) and (j-i+1)*len(clips[j][1]) <= CFG["ENC_BATCH_S"]*cod.sr:
                j += 1
            j = max(j, i+1)
            lats.extend(cod.encode([c[1] for c in clips[i:j]]))
            i = j
    z = np.concatenate(lats).astype(np.float32) if lats else np.zeros((0, 2*cod.D), np.float32)
    assert np.isfinite(z).all() and (z[:, cod.D:] >= 0).all()
    z16 = z.astype(np.float16)
    assert np.isfinite(z16).all(), "Float16 posterior overflow"
    z16.tofile(fp + ".partial")
    os.replace(fp + ".partial", fp)
    mean = z[:, :cod.D].astype(np.float64)
    meta = dict(codec=cod.name, D=cod.D, frames=len(z), ids=[c[0] for c in clips],
        texts=[c[2] for c in clips], speakers=[c[3] for c in clips], splits=[split]*len(clips),
        offs=np.concatenate([[0], np.cumsum([len(x) for x in lats])]).tolist(),
        sum=mean.sum(0).tolist(), sumsq=(mean**2).sum(0).tolist(),
        sum_var=(z[:, cod.D:].astype(np.float64)**2).sum(0).tolist(),
        hours=sum(c[4] for c in clips)/3600, duration_s=[c[4] for c in clips],
        policy_id=POLICY_ID, source_repo=EMILIA_REPO, source_revision=EMILIA_REV,
        source_archive=archive, rejected=dict(rejected), sha256=sha256_file(fp))
    atomic_json(jp, meta)  # JSON is the completion marker, written last.
    print(shard, len(clips), "clips", round(meta["hours"],3), "hours", flush=True)
    return True


def preparation_deadline(minutes, start):
    """None means run to the data target, subject to Colab's own runtime limits."""
    if minutes is None:
        return math.inf
    if not math.isfinite(minutes) or minutes <= 0:
        raise ValueError("minutes must be None or a positive finite number")
    return start + minutes*60


def prepare_large_data(minutes=None, max_new_chunks=None):
    """Prepare the agreed dataset. Target counts unique, filtered TRAIN audio only."""
    global DATA_READY, source_shards
    assert not SMOKE and CFG["CODEC"] == "ming" and "LIMIT" not in CFG
    assert CFG["PUSH"], "The large run requires durable private backups"
    start = time.monotonic()
    deadline = preparation_deadline(minutes, start)
    DATA_READY = False
    CFG["SETS"] = BASE_SETS + [EMILIA_SET]
    CFG["TARGET_TRAIN_HOURS"] = TARGET_TRAIN_HOURS
    _JUDGE.clear(); _CODECS.pop("mimi", None); STATE.clear()
    gc.collect(); torch.set_num_threads(4)
    if DEV == "cuda":
        torch.cuda.empty_cache()
    api = HfApi()
    assert api.repo_info(DATA_REPO, repo_type="dataset").private, "Expected the private experiment repo"
    remote = set(api.list_repo_files(DATA_REPO, repo_type="dataset"))
    root = os.path.join(WORK, "data", "ming")
    policy_rel = "preparation/emilia-policy.json"
    if policy_rel in remote:
        p = hf_hub_download(DATA_REPO, policy_rel, repo_type="dataset", local_dir=root)
        assert json.load(open(p)) == EMILIA_POLICY, "Use a new dataset repo for a changed preparation policy"
    else:
        atomic_json(os.path.join(root, policy_rel), EMILIA_POLICY)
        commit_paths(api, [policy_rel], remote, "Pin Emilia-YODAS preparation policy")
    restore_and_reconcile(api, remote)
    owners = emilia_id_owners()
    new_chunks = 0
    print("PREPARATION CONFIG", json.dumps(dict(target_train_hours=TARGET_TRAIN_HOURS,
        timed_stop_minutes=minutes, backups="private HF after each base shard / <=16 Emilia chunks")), flush=True)
    print("Starting accepted training hours:", round(training_hours(encoded_inventory()),3), flush=True)
    # Include the original corpora in the target BEFORE adding Emilia. This gives
    # an actual 2,500-hour mixed-data milestone, not 2,500 plus another ~700 hours.
    base_complete = True
    for s in BASE_SETS:
        for shard, loader in _source_shards_original(s):
            if time.monotonic() >= deadline:
                base_complete = False
                break
            fp, jp = shard_paths(s, shard)
            if not pair_ok(jp):
                if os.path.exists(jp):
                    # Preserve invalid metadata for inspection; allow the base
                    # encoder to regenerate this disposable incomplete pair.
                    os.replace(jp, jp + ".incomplete")
                print("Encoding base corpus", s, shard, flush=True)
                old_push = CFG["PUSH"]
                try:
                    source_shards = lambda _: [(shard, loader)]
                    CFG["PUSH"] = False
                    encode_set(s)
                finally:
                    source_shards = _source_shards_original
                    CFG["PUSH"] = old_push
            assert pair_ok(jp), f"Incomplete base shard: {s}/{shard}"
            commit_paths(api, [f"{s}/{shard}.f16",f"{s}/{shard}.json"], remote, f"Encode {s}/{shard}")
        if not base_complete:
            break
        print("BASE SOURCE READY",s,"| total accepted train hours:",
              round(training_hours(encoded_inventory()),3),flush=True)
    files = sorted(f for f in api.list_repo_files(EMILIA_REPO, repo_type="dataset", revision=EMILIA_REV)
                   if f.startswith("Emilia-YODAS/EN/") and f.endswith(".tar"))
    assert files, "No English Emilia-YODAS archives found"
    random.Random(EMILIA_POLICY["seed"]).shuffle(files)
    # Start with the already inspected archive, then use the seeded permutation.
    probe = "Emilia-YODAS/EN/EN-B000000.tar"
    files = [probe] + [f for f in files if f != probe]
    print("English archives:", len(files), "| target:", TARGET_TRAIN_HOURS, "unique training hours", flush=True)
    for archive in files:
        summary = encoded_inventory()
        if training_hours(summary) >= TARGET_TRAIN_HOURS or time.monotonic() >= deadline:
            break
        name = pathlib.PurePosixPath(archive).stem
        receipt = f"preparation/emilia/{name}.json"
        if receipt in remote:
            continue
        if shutil.disk_usage(WORK).free < 8*2**30:
            raise RuntimeError("Less than 8 GiB free; stop safely before the next archive")
        path = hf_hub_download(EMILIA_REPO, archive, repo_type="dataset", revision=EMILIA_REV,
                               token=True, local_dir=os.path.join(WORK,"raw","emilia"))
        pending, complete = [], True
        with tarfile.open(path) as tar:
            members = {m.name:m for m in tar.getmembers() if m.isfile()}
            groups = {s:[] for s in ("train", "val", "test")}
            rejected, seen = collections.Counter(), set()
            for key in sorted(members):
                if not key.endswith(".json"):
                    continue
                meta = json.load(tar.extractfile(members[key]))
                cid = pathlib.PurePosixPath(key).stem
                assert meta.get("_id", meta.get("id", cid)) == cid, "Archive identity mismatch"
                assert cid not in seen, "Duplicate clip ID in archive"
                seen.add(cid)
                if cid in owners and owners[cid] != archive:
                    rejected["duplicate_id"] += 1
                    continue
                reason = metadata_rejection(meta)
                if reason:
                    rejected[reason] += 1
                    continue
                audio = key[:-5] + ".mp3"
                if audio not in members:
                    rejected["missing_audio"] += 1
                    continue
                groups[emilia_split(cid)].append((cid, meta, members[audio]))
            print(name, "metadata retained", {s:len(v) for s,v in groups.items()},
                  "rejected", dict(rejected), flush=True)
            stop = False
            for split, rows in groups.items():
                for offset in range(0, len(rows), EMILIA_POLICY["chunk_clips"]):
                    if time.monotonic() >= deadline or (max_new_chunks is not None and new_chunks >= max_new_chunks):
                        complete = False; stop = True; break
                    shard = f"{name}-{split}-{offset//EMILIA_POLICY['chunk_clips']:05d}"
                    new = encode_emilia_chunk(tar, rows[offset:offset+EMILIA_POLICY["chunk_clips"]], shard, archive, split)
                    new_chunks += int(new)
                    chunk_meta = pair_ok(shard_paths(EMILIA_SET,shard)[1])
                    for cid in chunk_meta["ids"]:
                        owners[cid] = archive
                    pending.extend([f"{EMILIA_SET}/{shard}.f16", f"{EMILIA_SET}/{shard}.json"])
                    if len(pending) >= 32:
                        commit_paths(api, pending, remote, f"Encode {name}: progress")
                        pending.clear()
                if stop:
                    break
            if complete:
                atomic_json(os.path.join(root,receipt), dict(archive=archive, revision=EMILIA_REV,
                    policy_id=POLICY_ID, metadata_clips=len(seen), rejected=dict(rejected)))
                pending.append(receipt)
        commit_paths(api, pending, remote, f"Encode {name}: {'complete' if complete else 'paused'}")
        if complete:
            os.remove(path)  # Disposable download; encoded pairs are verified and backed up.
        summary = encoded_inventory()
        print("PREPARATION", json.dumps(summary), "\nUnique training hours:", round(training_hours(summary),3), flush=True)
        if not complete:
            break
    summary = encoded_inventory()
    if base_complete and training_hours(summary) >= TARGET_TRAIN_HOURS:
        audit_large_data()
        DATA_READY = True
    report = dict(policy_id=POLICY_ID, target_train_hours=TARGET_TRAIN_HOURS,
                  unique_train_hours=training_hours(summary), sources=summary,
                  new_chunks=new_chunks, session_minutes=(time.monotonic()-start)/60,
                  data_ready=DATA_READY, timed_stop_minutes=minutes)
    atomic_json(os.path.join(WORK,"preparation_progress.json"),report)
    print("SESSION SAVED", json.dumps(report), flush=True)
    if DATA_READY:
        print("DATA READY: target reached, original corpora complete, backup/identity audit passed. Main training has not started.", flush=True)
    else:
        print("Preparation incomplete. Rerun prepare_large_data() to continue. Main training has not started.", flush=True)
    return report


def audit_large_data():
    """Fail closed until the requested dataset and backups are complete. No training here."""
    summary = encoded_inventory()
    assert training_hours(summary) >= TARGET_TRAIN_HOURS, f"Continue preparation: fewer than {TARGET_TRAIN_HOURS:,.0f} unique training hours"
    for s in BASE_SETS:
        for shard, _ in _source_shards_original(s):
            assert pair_ok(shard_paths(s,shard)[1]), f"Original corpus incomplete: {s}/{shard}"
    emilia_id_owners()  # Enforce unique clip IDs across archives/chunks.
    remote = set(HfApi().list_repo_files(DATA_REPO, repo_type="dataset"))
    root = os.path.join(WORK,"data","ming")
    manifest = []
    for s in CFG["SETS"]:
        for jp in sorted(glob.glob(os.path.join(root,s,"*.json"))):
            rel = os.path.relpath(jp,root)
            assert rel in remote and rel[:-5]+".f16" in remote, f"Backup incomplete: {rel}"
            manifest.append(dict(path=rel,metadata_sha256=sha256_file(jp)))
    result = dict(policy_id=POLICY_ID, unique_train_hours=training_hours(summary),
                  sources=summary, shards=manifest)
    atomic_json(os.path.join(WORK,"large_data_audit.json"),result)
    print("Data audit passed. Set training steps from measured epoch length and validate cold checkpoint resume before the long run.", flush=True)
    return result

# User-authorized continuous preparation to 2,500 usable training hours.
# Saves progress throughout; rerun after a Colab runtime interruption.
preparation = prepare_large_data(minutes=None)

# %%
# Full-corpus training: exact resume test and short timing calibration.
# All data are already prepared; this never reruns encoding.
assert not SMOKE and 'emilia_yodas' in CFG['SETS']
import signal, gc
STATE.clear()
model = None
_CODECS.clear()
_JUDGE.clear()
for _name in ('preflight_data','preflight_items','preflight_lens','preflight_spk','preflight_tok'):
    globals().pop(_name, None)
gc.collect(); torch.cuda.empty_cache()
CFG.update(RUN='g270m_2500_v1', TRAIN_EPOCHS=3, SEED=20261104,
           LOG_EVERY=25, VAL_EVERY=2000, SAVE_MIN=10, SESSION_HOURS=0.75, UNITS_PER_HOUR=6.73)
training_worker_path = os.path.join(WORK,'training_worker.py')
training_config_path = os.path.join(WORK,'training_launch.json')
with open(training_worker_path,"w") as f: f.write('import os, sys, math, json, time, glob, shutil, subprocess, threading, queue, random, io\nimport numpy as np\nimport soundfile as sf\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nfrom scipy.signal import resample_poly\nos.environ[\'TOKENIZERS_PARALLELISM\'] = \'false\'\ntorch.set_num_threads(4)\nwith open(sys.argv[1]) as f:\n    launch_config = json.load(f)\nCFG, WORK, DATA_REPO, CKPT_REPO = (launch_config[k] for k in (\'CFG\',\'WORK\',\'DATA_REPO\',\'CKPT_REPO\'))\nDEV, AMP, LOCAL, SMOKE, IN_COLAB = \'cuda\', torch.bfloat16, False, False, False\nsys.path.insert(0, os.path.join(WORK,\'ming_pkg\'))\ntorch.backends.cuda.matmul.allow_tf32 = True\ntorch.backends.cudnn.allow_tf32 = True\n\nTRAINING_SOURCE_HASH = \'a1e627f6e485a922f2a4f9314a237c5f2b46b301b094bbea182ea9d19b82e40a\'\n# ---------------------------------------------------------------- codecs: wav <-> latent frames (12.5 per second)\ndef resample(x, sr_in, sr_out, dev=DEV):\n    """1-D float32 numpy -> numpy at sr_out (torchaudio on the GPU; scipy fallback)."""\n    if sr_in == sr_out:\n        return x.astype(np.float32)\n    try:\n        import torchaudio.functional as AF\n        y = AF.resample(torch.from_numpy(x.astype(np.float32)).to(dev), sr_in, sr_out)\n        return y.float().cpu().numpy()\n    except Exception:\n        g = math.gcd(sr_in, sr_out)\n        return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)\n\n\nclass MingCodec:\n    """Ming-omni-tts-tokenizer-12Hz: 44.1 kHz, 882-sample frames at 50 Hz pooled 4:1 to 12.5 Hz, 64-d VAE latent.\n    encode() returns [mean | std] (2 x 64) per frame so training can sample the posterior the way Ming does."""\n    name, sr, D, hop = "ming", 44100, 64, 882 * 4\n\n    def __init__(self, dev=DEV):\n        from ming_audio_tokenizer.modeling_audio_vae import AudioVAE\n        from ming_audio_tokenizer.configuration_audio_vae import AudioVAEconfig\n        repo = "inclusionAI/Ming-omni-tts-tokenizer-12Hz"\n        cfg = AudioVAEconfig.from_pretrained(repo)\n        for kw in (cfg.enc_kwargs, cfg.dec_kwargs):     # flash-attn is optional; sdpa needs no extra install\n            kw["backbone"]["_attn_implementation"] = "sdpa"\n            kw["backbone"]["attn_implementation"] = "sdpa"\n        dt = torch.bfloat16 if dev == "cuda" else torch.float32\n        self.vae = AudioVAE.from_pretrained(repo, config=cfg, torch_dtype=dt).to(dev).eval()\n        self.dev, self.dt = dev, dt\n\n    def frames(self, n):\n        return math.ceil(math.ceil(n / 882) / 4)\n\n    @torch.no_grad()\n    def encode(self, wavs):\n        """wavs: list of 1-D float32 at 44.1 kHz -> list of (frames, 128) float32 [mean | std].\n        Batched with right padding. The encoder is causal, and hidden states past each clip\'s last real 882-sample\n        frame are zeroed before the 4:1 pooling, which is what a one-clip call does (it pads the last patch with zeros)."""\n        enc = self.vae.encoder\n        L = max(len(w) for w in wavs)\n        x = torch.zeros(len(wavs), L, device=self.dev, dtype=self.dt)\n        for b, w in enumerate(wavs):\n            x[b, :len(w)] = torch.from_numpy(w).to(self.dev, self.dt)\n        h = enc.fc2(enc.fc1(enc.get_frames(x)))\n        h = enc.encoder(inputs_embeds=h).last_hidden_state\n        for b, w in enumerate(wavs):\n            h[b, math.ceil(len(w) / 882):] = 0\n        h = enc.aggregator(inputs_embeds=enc.pad_patch_insert_cls(h)).last_hidden_state\n        B, _, d = h.shape\n        h = h.reshape(-1, enc.patch_size + 1, d)[:, -1:].reshape(B, -1, d)\n        h = enc.fc3(h).float()\n        mean, scale = h.chunk(2, dim=-1)\n        std = F.softplus(scale) + 1e-4\n        out = torch.cat([mean, std], -1).cpu().numpy()\n        return [out[b, :self.frames(len(w))] for b, w in enumerate(wavs)]\n\n    @torch.no_grad()\n    def self_check(self, x):\n        """encode() re-implements Ming\'s Encoder.forward (to batch). On one clip it must equal Ming\'s own forward."""\n        h, _ = self.vae.encoder(torch.from_numpy(x)[None].to(self.dev, self.dt))\n        ref = h[0, :, :self.D].float().cpu().numpy()\n        mine = self.encode([x])[0][:, :self.D]\n        n = min(len(ref), len(mine))\n        err = float(np.linalg.norm(mine[:n] - ref[:n]) / (np.linalg.norm(ref[:n]) + 1e-9))\n        print(f"ming encode vs Ming\'s own forward: frames {len(mine)} vs {len(ref)}, rel err {err:.5f}", flush=True)\n        assert len(mine) == len(ref) and err < 0.02, "MingCodec.encode disagrees with Ming\'s Encoder.forward"\n\n    @torch.no_grad()\n    def decode(self, lat):\n        """(frames, 64) float32 -> 1-D float32 wav at 44.1 kHz (whole utterance in one pass, as Ming does)."""\n        z = torch.from_numpy(np.ascontiguousarray(lat, dtype=np.float32))[None].to(self.dev, self.dt)\n        y = self.vae.decode(z, past_key_values=None, use_cache=False, last_chunk=True)[0]\n        return y.reshape(-1).float().cpu().numpy()\n\n\nclass MimiCodec:\n    """Kyutai Mimi through transformers: 24 kHz, 1920-sample frames, the 512-d latent at the quantizer input\n    (\'space c\' in 33: semantic + acoustic input projections, codebook lookup skipped). Deterministic: std = 0."""\n    name, sr, D, hop = "mimi", 24000, 512, 1920\n\n    def __init__(self, dev=DEV):\n        from transformers import MimiModel\n        self.m = MimiModel.from_pretrained("kyutai/mimi").to(dev).eval()\n        self.dev = dev\n\n    def frames(self, n):\n        return math.ceil(n / 1920)\n\n    @torch.no_grad()\n    def encode(self, wavs):\n        m, q = self.m, self.m.quantizer\n        L = max(len(w) for w in wavs)\n        x = torch.zeros(len(wavs), 1, L, device=self.dev)\n        for b, w in enumerate(wavs):\n            x[b, 0, :len(w)] = torch.from_numpy(w).to(self.dev)\n        e = m.encoder(x)\n        e = m.encoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)\n        z = m.downsample(e)\n        c = torch.cat([q.semantic_residual_vector_quantizer.input_proj(z),\n                       q.acoustic_residual_vector_quantizer.input_proj(z)], 1).transpose(1, 2).float()\n        out = torch.cat([c, torch.zeros_like(c)], -1).cpu().numpy()\n        return [out[b, :self.frames(len(w))] for b, w in enumerate(wavs)]\n\n    @torch.no_grad()\n    def decode(self, lat):\n        m, q = self.m, self.m.quantizer\n        x = torch.from_numpy(np.ascontiguousarray(lat.T, dtype=np.float32))[None].to(self.dev)\n        e = q.semantic_residual_vector_quantizer.output_proj(x[:, :256]) + \\\n            q.acoustic_residual_vector_quantizer.output_proj(x[:, 256:])\n        e = m.upsample(e)\n        e = m.decoder_transformer(e.transpose(1, 2))[0].transpose(1, 2)\n        return m.decoder(e)[0, 0].float().cpu().numpy()\n\n\n_CODECS = {}\n\n\ndef get_codec(name=None):\n    name = name or CFG["CODEC"]\n    if name not in _CODECS:\n        _CODECS[name] = {"ming": MingCodec, "mimi": MimiCodec}[name]()\n    return _CODECS[name]\n\n# ---------------------------------------------------------------- judges: Whisper large-v3-turbo + UTMOS22 (as in 33)\nimport re\nABBR = {"mrs": "missus", "mr": "mister", "dr": "doctor", "st": "saint", "lt": "lieutenant", "sgt": "sergeant",\n        "capt": "captain", "gov": "governor", "gen": "general", "jr": "junior", "sr": "senior"}\n\n\ndef normalize(t):\n    """33\'s judge.py normalizer: lowercase, numbers spelled out, punctuation dropped except in-word apostrophes."""\n    try:\n        from num2words import num2words\n    except ImportError:                                # laptop test env only; Colab installs it\n        num2words = None\n    t = t.lower().replace("&", " and ").replace("%", " percent")\n    t = re.sub(r"(\\d),(\\d)", r"\\1\\2", t)\n    if num2words:\n        t = re.sub(r"\\b(1[0-9]|20)\\d\\d\\b", lambda m: " " + num2words(int(m.group()), to="year") + " ", t)\n        t = re.sub(r"\\d+(\\.\\d+)?", lambda m: " " + num2words(float(m.group()) if "." in m.group() else int(m.group())) + " ", t)\n    for k, v in ABBR.items():\n        t = re.sub(rf"\\b{k}\\.", v + " ", t)\n    t = t.replace("-", " ")\n    t = re.sub(r"[^a-z\' ]+", " ", t)\n    t = re.sub(r"(?<![a-z])\'|\'(?![a-z])", " ", t)\n    return re.sub(r"\\s+", " ", t).strip()\n\n\ndef edits(a, b):\n    prev = list(range(len(b) + 1))\n    for i, x in enumerate(a, 1):\n        cur = [i] + [0] * len(b)\n        for j, y in enumerate(b, 1):\n            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))\n        prev = cur\n    return prev[-1]\n\n\nclass Judge:\n    def __init__(self):\n        from transformers import pipeline\n        dt = torch.float16 if DEV == "cuda" else torch.float32\n        self.asr = pipeline("automatic-speech-recognition", model=CFG["WHISPER"], torch_dtype=dt, device=DEV)\n        try:\n            self.mos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()\n        except Exception as e:\n            print("UTMOS unavailable:", e); self.mos = None\n\n    def score(self, items, tag=""):\n        """items: list of dicts {id, wav (1-D float32), sr, ref}. Returns corpus CER/WER %, mean UTMOS + 95% CI."""\n        w16 = [resample(it["wav"], it["sr"], 16000, "cpu") if len(it["wav"]) else np.zeros(1600, np.float32) for it in items]\n        hyps = self.asr([{"raw": w, "sampling_rate": 16000} for w in w16], batch_size=16,\n                        generate_kwargs={"language": "en", "task": "transcribe"})\n        ce = cn = we = wn = 0\n        rows = []\n        for it, h_raw, w in zip(items, hyps, w16):\n            h, r = normalize(h_raw["text"]), normalize(it["ref"])\n            c, wd = edits(h, r), edits(h.split(), r.split())\n            ce += c; cn += len(r); we += wd; wn += len(r.split())\n            mos = float(self.mos(torch.from_numpy(w)[None].to(DEV), 16000)[0]) if self.mos is not None else float("nan")\n            rows.append({"id": it["id"], "hyp": h, "ref": r, "cer": c / max(1, len(r)), "utmos": round(mos, 4)})\n        v = np.array([r["utmos"] for r in rows])\n        rng = np.random.default_rng(1)\n        bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]\n        out = {"tag": tag, "n": len(rows), "cer": round(100 * ce / max(cn, 1), 2), "wer": round(100 * we / max(wn, 1), 2),\n               "utmos": round(float(v.mean()), 3), "utmos_ci95": [round(float(np.percentile(bs, 2.5)), 3),\n                                                                 round(float(np.percentile(bs, 97.5)), 3)], "clips": rows}\n        print(json.dumps({k: v for k, v in out.items() if k != "clips"}), flush=True)\n        return out\n\n\n_JUDGE = []\n\n\ndef judge():\n    if not _JUDGE:\n        _JUDGE.append(Judge())\n    return _JUDGE[0]\n\n# ---------------------------------------------------------------- data sources\n# 33\'s LJ val-100 (RESULTS.md section 2 and every f1-f3 val number): the first 100 of its val list.\nVAL100 = """LJ041-0003 LJ041-0005 LJ041-0007 LJ041-0013 LJ041-0018 LJ041-0023 LJ041-0027 LJ041-0032 LJ041-0042 LJ041-0059\nLJ041-0062 LJ041-0065 LJ041-0072 LJ041-0073 LJ041-0101 LJ041-0113 LJ041-0136 LJ041-0140 LJ041-0198 LJ042-0007 LJ042-0009\nLJ042-0015 LJ042-0017 LJ042-0026 LJ042-0038 LJ042-0041 LJ042-0061 LJ042-0075 LJ042-0083 LJ042-0091 LJ042-0105 LJ042-0114\nLJ042-0116 LJ042-0125 LJ042-0128 LJ042-0137 LJ042-0143 LJ042-0157 LJ042-0180 LJ042-0189 LJ042-0194 LJ042-0195 LJ042-0210\nLJ042-0229 LJ042-0230 LJ042-0233 LJ042-0237 LJ042-0242 LJ042-0244 LJ043-0002 LJ043-0026 LJ043-0035 LJ043-0038 LJ043-0042\nLJ043-0048 LJ043-0055 LJ043-0057 LJ043-0059 LJ043-0080 LJ043-0082 LJ043-0090 LJ043-0097 LJ043-0119 LJ043-0124 LJ043-0135\nLJ043-0137 LJ043-0155 LJ043-0158 LJ043-0162 LJ043-0166 LJ043-0169 LJ043-0179 LJ043-0186 LJ044-0011 LJ044-0018 LJ044-0020\nLJ044-0029 LJ044-0042 LJ044-0048 LJ044-0059 LJ044-0064 LJ044-0075 LJ044-0079 LJ044-0090 LJ044-0096 LJ044-0104 LJ044-0119\nLJ044-0129 LJ044-0134 LJ044-0154 LJ044-0160 LJ044-0178 LJ044-0189 LJ044-0194 LJ044-0199 LJ044-0207 LJ044-0208 LJ044-0220\nLJ044-0235 LJ044-0238""".split()\n# 33\'s reference numbers on val-100 (mlx-whisper large-v3-turbo; the gate re-measures the real recordings here)\nREF_33 = {"real recording": (0.23, 1.06, 4.40), "Qwen3-12Hz 16 codebooks": (0.32, 1.18, 4.44),\n          "Mimi latent (33\'s target)": (0.38, 1.23, 4.28), "f3 (33\'s model)": (0.61, 1.62, 3.12)}\n\n\ndef lj_root():\n    if LOCAL:\n        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ljspeech", "LJSpeech-1.1")\n    root = os.path.join(WORK, "LJSpeech-1.1")\n    if not os.path.exists(os.path.join(root, "metadata.csv")):\n        tb = os.path.join(WORK, "LJSpeech-1.1.tar.bz2")\n        subprocess.run(["wget", "-q", "-O", tb, "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"], check=True)\n        subprocess.run(["tar", "-xjf", tb, "-C", WORK], check=True)\n        os.remove(tb)\n    return root\n\n\ndef lj_rows():\n    """(id, split, normalized text). 33\'s split: LJ001-040 train, 041-044 val, 045-050 test."""\n    out = []\n    for line in open(os.path.join(lj_root(), "metadata.csv"), encoding="utf-8"):\n        cid, raw, norm = line.rstrip("\\n").split("|")\n        ch = int(cid[2:5])\n        out.append((cid, "train" if ch <= 40 else ("val" if ch <= 44 else "test"), norm))\n    return out\n\n\ndef lj_wav(cid):\n    x, sr = sf.read(os.path.join(lj_root(), "wavs", cid + ".wav"), dtype="float32")\n    return x, sr\n\n\n# HF parquet sources: (repo, file prefix, text column, speaker column, id column)\nHF_SOURCES = {\n    "libritts_clean100": ("mythicinfinity/libritts_r", "data/train.clean.100/", "text_normalized", "speaker_id", "id"),\n    "libritts_clean360": ("mythicinfinity/libritts_r", "data/train.clean.360/", "text_normalized", "speaker_id", "id"),\n    "libritts_other500": ("mythicinfinity/libritts_r", "data/train.other.500/", "text_normalized", "speaker_id", "id"),\n    "hifi_clean": ("MikhailT/hifi-tts", "data/train.clean-", "text_normalized", "speaker", "file"),\n    "hifi_other": ("MikhailT/hifi-tts", "data/train.other-", "text_normalized", "speaker", "file"),\n}\nSPK_PREFIX = {"libritts": "libritts", "hifi": "hifi", "lj": "lj"}\n\n\ndef source_shards(set_name):\n    """Shards of one set: list of (shard name, loader) where loader() yields (id, wav, sr, text, speaker, split)."""\n    if set_name == "lj":\n        rows = lj_rows()\n        lim = CFG.get("LIMIT")\n\n        def load(split):\n            sel = [r for r in rows if r[1] == split]\n            if lim:\n                keep = set(VAL100[:CFG["EVAL_N"]]) if split == "val" else set()\n                sel = [r for r in sel if r[0] in keep] + [r for r in sel if r[0] not in keep][:lim if split == "train" else 16]\n            for cid, sp, text in sel:\n                x, sr = lj_wav(cid)\n                yield cid, x, sr, text, "lj", sp\n        return [("train", lambda: load("train")), ("val", lambda: load("val"))]\n    from huggingface_hub import HfApi, hf_hub_download\n    repo, prefix, tcol, scol, icol = HF_SOURCES[set_name]\n    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith(prefix) and f.endswith(".parquet"))\n    if CFG.get("LIMIT"):\n        files = files[:1]\n\n    def make(f):\n        def load():\n            import pyarrow.parquet as pq\n            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(WORK, "raw"))\n            t = pq.read_table(p, columns=["audio", tcol, scol, icol])\n            n = t.num_rows if not CFG.get("LIMIT") else min(t.num_rows, CFG["LIMIT"])\n            for i in range(n):\n                a = t["audio"][i].as_py()\n                x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")\n                if x.ndim > 1:\n                    x = x.mean(1)\n                yield str(t[icol][i].as_py()), x, sr, str(t[tcol][i].as_py()), str(t[scol][i].as_py()), "train"\n            del t\n            os.remove(p)                                # the user\'s loop: pull, encode, save, delete\n        return load\n    return [(os.path.basename(f).replace(".parquet", ""), make(f)) for f in files]\n\n# ---------------------------------------------------------------- encode: sources -> latent shards (+ HF dataset repo)\ndef shard_paths(set_name, shard):\n    d = os.path.join(WORK, "data", CFG["CODEC"], set_name)\n    os.makedirs(d, exist_ok=True)\n    return os.path.join(d, shard + ".f16"), os.path.join(d, shard + ".json")\n\n\ndef hf_files(repo, repo_type):\n    from huggingface_hub import HfApi\n    try:\n        return set(HfApi().list_repo_files(repo, repo_type=repo_type))\n    except Exception:\n        return set()\n\n\ndef batching_ok(cod, wavs):\n    """Batched encoding pads short clips on the right; that is exact only if the encoder never looks ahead.\n    Compare batched vs one-at-a-time on a few clips (Mimi through transformers fails this: 11-13% error)."""\n    if getattr(cod, "batch_ok", None) is None:\n        ws = sorted(wavs, key=len)[:2] + sorted(wavs, key=len)[-1:]\n        if hasattr(cod, "self_check") and not getattr(cod, "checked", False):\n            cod.self_check(ws[-1]); cod.checked = True\n        zb = cod.encode(ws)\n        err = max(float(np.linalg.norm(a[:, :cod.D] - b[:, :cod.D]) / (np.linalg.norm(b[:, :cod.D]) + 1e-9))\n                  for a, b in zip(zb, [cod.encode([w])[0] for w in ws]))\n        cod.batch_ok = err < 0.03\n        print(f"{cod.name}: batched vs single encode rel err {err:.4f} -> " + ("batched" if cod.batch_ok else "one clip at a time"), flush=True)\n    return cod.batch_ok\n\n\ndef encode_set(set_name, remote=None):\n    """Encode every shard of one set that isn\'t already done locally or in the HF data repo."""\n    cod = get_codec()\n    remote = hf_files(DATA_REPO, "dataset") if remote is None and CFG["PUSH"] else (remote or set())\n    for shard, load in source_shards(set_name):\n        fp, jp = shard_paths(set_name, shard)\n        rel = f"{set_name}/{shard}"\n        if os.path.exists(jp):\n            continue\n        if rel + ".json" in remote:\n            from huggingface_hub import hf_hub_download\n            for ext in (".f16", ".json"):\n                hf_hub_download(DATA_REPO, rel + ext, repo_type="dataset", local_dir=os.path.join(WORK, "data", CFG["CODEC"]))\n            continue\n        t0 = time.time()\n        clips = []\n        for cid, x, sr, text, spk, split in load():\n            dur = len(x) / sr\n            if not (CFG["MIN_S"] <= dur <= CFG["MAX_S"]) or len(text.strip()) < 3:\n                continue\n            clips.append((cid, resample(x, sr, cod.sr), text.strip(), f"{set_name.split(\'_\')[0]}:{spk}", split))\n        clips.sort(key=lambda c: len(c[1]))\n        lats = [None] * len(clips)\n        batched = batching_ok(cod, [c[1] for c in clips[:: max(1, len(clips) // 8)]]) if clips else False\n        i = 0\n        while i < len(clips):                           # length-sorted batches under an audio-seconds budget\n            j = i\n            while batched and j < len(clips) and (j - i + 1) * len(clips[j][1]) <= CFG["ENC_BATCH_S"] * cod.sr:\n                j += 1\n            j = max(j, i + 1)\n            for k, z in enumerate(cod.encode([c[1] for c in clips[i:j]])):\n                lats[i + k] = z\n            i = j\n        offs = np.concatenate([[0], np.cumsum([len(z) for z in lats])]).astype(np.int64)\n        allz = np.concatenate(lats).astype(np.float32) if lats else np.zeros((0, 2 * cod.D), np.float32)\n        allz.astype(np.float16).tofile(fp)\n        mean = allz[:, :cod.D]\n        meta = {"codec": cod.name, "D": cod.D, "frames": int(len(allz)), "ids": [c[0] for c in clips],\n                "texts": [c[2] for c in clips], "speakers": [c[3] for c in clips], "splits": [c[4] for c in clips],\n                "offs": offs.tolist(), "sum": mean.sum(0).tolist(), "sumsq": (mean.astype(np.float64) ** 2).sum(0).tolist(),\n                "sum_var": (allz[:, cod.D:].astype(np.float64) ** 2).sum(0).tolist(),\n                "hours": round(sum(len(c[1]) for c in clips) / cod.sr / 3600, 3)}\n        json.dump(meta, open(jp, "w"))\n        if CFG["PUSH"]:\n            from huggingface_hub import HfApi\n            api = HfApi()\n            api.create_repo(DATA_REPO, repo_type="dataset", private=True, exist_ok=True)\n            api.upload_file(path_or_fileobj=fp, path_in_repo=rel + ".f16", repo_id=DATA_REPO, repo_type="dataset")\n            api.upload_file(path_or_fileobj=jp, path_in_repo=rel + ".json", repo_id=DATA_REPO, repo_type="dataset")\n        print(f"{rel}: {len(clips)} clips, {meta[\'hours\']} h, {meta[\'frames\']} frames, {time.time() - t0:.0f} s", flush=True)\n\n\ndef encode_all():\n    remote = hf_files(DATA_REPO, "dataset") if CFG["PUSH"] else set()\n    for s in CFG["SETS"]:\n        encode_set(s, remote)\n\n# ---------------------------------------------------------------- dataset: shards -> training items\nclass Store:\n    """All shards of the run behind one global frame index; each row is [mean (D) | std (D)] in fp16."""\n    def __init__(self):\n        self.parts, self.base = [], [0]\n\n    def add(self, fp, frames, D):\n        self.parts.append(np.memmap(fp, dtype=np.float16, mode="r", shape=(frames, 2 * D)) if frames else None)\n        self.base.append(self.base[-1] + frames)\n        return self.base[-2]\n\n    def get(self, off, n):\n        k = int(np.searchsorted(self.base, off, side="right") - 1)\n        return self.parts[k][off - self.base[k]: off - self.base[k] + n]\n\n\n_CHAR = {}\n\n\ndef text_ids(tok, text):\n    """BOS + one Gemma token per character (33: subwords were too sparse at this data size)."""\n    ids = [tok.bos_token_id]\n    for ch in text:\n        if ch not in _CHAR:\n            _CHAR[ch] = tok.encode(ch, add_special_tokens=False)\n        ids.extend(_CHAR[ch])\n    return ids\n\n\ndef load_data(tok):\n    """Returns train items, val items, eval items (val-100 ids), the store, speaker map and latent stats."""\n    cod_D = {"ming": 64, "mimi": 512}[CFG["CODEC"]]\n    metas = []\n    for s in CFG["SETS"]:\n        for jp in sorted(glob.glob(os.path.join(WORK, "data", CFG["CODEC"], s, "*.json"))):\n            metas.append((s, jp, json.load(open(jp))))\n    assert metas, "no encoded shards found: run encode_all() first"\n    spk = {"lj:lj": 0}\n    for s in sorted({sp for _, _, m in metas for sp in m["speakers"]} - {"lj:lj"}):\n        spk[s] = len(spk)\n    n = s1 = s2 = sv = 0\n    store = Store()\n    train, val, ev = [], [], {}\n    want = set(VAL100[:CFG["EVAL_N"]])\n    for s, jp, m in metas:\n        base = store.add(jp[:-5] + ".f16", m["frames"], m["D"])\n        rep = CFG["LJ_REPEAT"] if s == "lj" else 1\n        for i, cid in enumerate(m["ids"]):\n            it = (tuple(text_ids(tok, m["texts"][i])), base + m["offs"][i], m["offs"][i + 1] - m["offs"][i], spk[m["speakers"][i]])\n            if m["splits"][i] == "train":\n                train += [it] * rep\n            elif m["splits"][i] == "val":\n                val.append(it)\n                if cid in want:\n                    ev[cid] = (it, m["texts"][i])\n        if any(sp == "train" for sp in m["splits"]):\n            n += m["frames"]; s1 += np.array(m["sum"]); s2 += np.array(m["sumsq"]); sv += np.array(m["sum_var"])\n    mu = s1 / n\n    var = s2 / n - mu ** 2 + (sv / n if CFG["LATENT"] == "sample" else 0)\n    sd = np.sqrt(np.maximum(var, 1e-8))\n    rng = random.Random(0)\n    rng.shuffle(val)\n    hours = sum(it[2] for it in train) / 12.5 / 3600\n    print(f"train {len(train)} clips ({hours:.0f} h incl. LJ x{CFG[\'LJ_REPEAT\']}), val {len(val)}, eval {len(ev)}, "\n          f"speakers {len(spk)}, D {cod_D}", flush=True)\n    return train, val[:CFG["VAL_N"]], ev, store, spk, mu.astype(np.float32), sd.astype(np.float32)\n\n\ndef batches_for_epoch(lens, budget, seed, epoch):\n    """33\'s length-bucketed batches under a padded-token budget; order is a pure function of (seed, epoch)."""\n    rng = np.random.default_rng(seed * 1000 + epoch)\n    lens = np.asarray(lens)\n    order = np.argsort(lens + rng.uniform(0, 8, len(lens)))\n    out, cur, L = [], [], 0\n    for i in order:\n        L2 = max(L, lens[i])\n        if cur and L2 * (len(cur) + 1) > budget:\n            out.append(cur); cur, L2 = [], lens[i]\n        cur.append(int(i)); L = L2\n    if cur:\n        out.append(cur)\n    rng.shuffle(out)\n    return out\n\n\ndef make_batch(items, store, D, mu, sd, rng):\n    """-> dict of tensors: text (B,Lt), tlen, lat (B,La,D) normalized (posterior-sampled if LATENT=\'sample\'), flen, spk."""\n    B = len(items)\n    Lt = max(len(it[0]) for it in items)\n    La = max(it[2] for it in items)\n    text = np.zeros((B, Lt), np.int64)\n    lat = np.zeros((B, La, D), np.float32)\n    tlen, flen, spk = (np.array([len(it[0]) for it in items]), np.array([it[2] for it in items]),\n                       np.array([it[3] for it in items]))\n    for b, (t, off, Fn, _) in enumerate(items):\n        text[b, :len(t)] = t\n        z = np.asarray(store.get(off, Fn), np.float32)\n        x = z[:, :D]\n        if CFG["LATENT"] == "sample":\n            x = x + z[:, D:] * rng.standard_normal(x.shape).astype(np.float32)\n        lat[b, :Fn] = (x - mu) / sd\n    t = lambda a: torch.from_numpy(a)\n    return {"text": t(text), "tlen": t(tlen), "lat": t(lat), "flen": t(flen), "spk": t(spk)}\n\n# ---------------------------------------------------------------- model: backbone + input projection + flow head + stop\nclass Backbone(nn.Module):\n    """Thin adapter so the backbone is a config string (Gemma 3 270M now; Gemma 4 E2B or another HF causal LM later).\n    embed() returns the model\'s own scaled token embeddings; forward() takes inputs_embeds."""\n    def __init__(self, name):\n        super().__init__()\n        from transformers import AutoModelForCausalLM\n        lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation="sdpa")\n        self.m = lm.model\n        self.d = lm.config.hidden_size\n        self.m.embed_tokens.weight.requires_grad_(False)     # 168M-row text table stays frozen (as in 33)\n\n    def embed(self, ids):\n        return self.m.embed_tokens(ids)\n\n    def forward(self, embeds, attention_mask=None, past=None, use_cache=False, position_ids=None):\n        o = self.m(inputs_embeds=embeds, attention_mask=attention_mask, past_key_values=past, use_cache=use_cache,\n                   position_ids=position_ids)\n        return o.last_hidden_state, o.past_key_values\n\n\ndef rope3(x):\n    """Rotary embedding over the head\'s 3 token positions (0, 1, 2). x (N, H, 3, hd)."""\n    hd = x.shape[-1]\n    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device, dtype=torch.float32) / hd))\n    f = torch.arange(x.shape[2], device=x.device, dtype=torch.float32)[:, None] * inv[None]\n    cos, sin = f.cos().to(x.dtype), f.sin().to(x.dtype)\n    x1, x2 = x[..., : hd // 2], x[..., hd // 2:]\n    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)\n\n\nclass HeadBlock(nn.Module):\n    def __init__(self, w, heads):\n        super().__init__()\n        self.h = heads\n        self.n1, self.n2 = nn.RMSNorm(w, eps=1e-6), nn.RMSNorm(w, eps=1e-6)\n        self.qkv, self.o = nn.Linear(w, 3 * w), nn.Linear(w, w)\n        self.f1, self.f2 = nn.Linear(w, 4 * w), nn.Linear(4 * w, w)\n\n    def forward(self, x):\n        N, T, w = x.shape\n        q, k, v = self.qkv(self.n1(x)).reshape(N, T, 3, self.h, w // self.h).permute(2, 0, 3, 1, 4)\n        a = F.scaled_dot_product_attention(rope3(q), rope3(k), v)          # bidirectional over the 3 tokens\n        x = x + self.o(a.transpose(1, 2).reshape(N, T, w))\n        return x + self.f2(F.gelu(self.f1(self.n2(x)), approximate="tanh"))\n\n\nclass FlowHead(nn.Module):\n    """Ming\'s DiT head: tokens [t_emb(t) + W_c h, W_x prev_frame, W_x x_t], 8 pre-RMSNorm blocks, velocity read off the\n    last token. Same W_x for the previous frame and the noisy frame (Ming\'s x_embedder)."""\n    def __init__(self, D, d_cond, w=1024, depth=8, heads=16):\n        super().__init__()\n        self.t1, self.t2 = nn.Linear(256, w), nn.Linear(w, w)\n        self.x_in, self.c_in = nn.Linear(D, w), nn.Linear(d_cond, w)\n        self.blocks = nn.ModuleList([HeadBlock(w, heads) for _ in range(depth)])\n        self.norm = nn.RMSNorm(w, eps=1e-6)\n        self.out = nn.Linear(w, D)\n        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)\n\n    def t_emb(self, t):\n        half = 128\n        f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))\n        a = 1000.0 * t.float()[:, None] * f[None]\n        return self.t2(F.silu(self.t1(torch.cat([a.sin(), a.cos()], -1).to(self.t1.weight.dtype))))\n\n    def forward(self, x, t, h, prev):\n        tok = torch.stack([self.t_emb(t) + self.c_in(h), self.x_in(prev), self.x_in(x)], 1)\n        for b in self.blocks:\n            tok = b(tok)\n        return self.out(self.norm(tok[:, 2]))\n\n\nclass GemmaVoice(nn.Module):\n    """Sequence per clip: [BOS text chars] [AUDIO_START + spk] [frame 0 + spk] ... ; the hidden state at AUDIO_START+s\n    conditions frame s through the flow head and says \'stop\' at s = F."""\n    def __init__(self, backbone_name, D, n_spk):\n        super().__init__()\n        self.bb = Backbone(backbone_name)\n        d = self.bb.d\n        with torch.no_grad():\n            target = float(self.bb.embed(torch.arange(20000)[None]).std())\n        self.D = D\n        self.in_proj = nn.Linear(D, d)\n        nn.init.normal_(self.in_proj.weight, std=target / math.sqrt(D)); nn.init.zeros_(self.in_proj.bias)\n        self.audio_start = nn.Parameter(torch.randn(d) * target)\n        self.spk = nn.Embedding(max(1, n_spk), d)\n        nn.init.zeros_(self.spk.weight)\n        self.head = FlowHead(D, d, CFG["HEAD_W"], CFG["HEAD_DEPTH"], CFG["HEAD_HEADS"])\n        self.stop = nn.Linear(d, 1)\n        nn.init.zeros_(self.stop.weight); nn.init.constant_(self.stop.bias, -4.0)\n\n    def new_params(self):\n        return [p for n, p in self.named_parameters() if not n.startswith("bb.") and p.requires_grad]\n\n    def bb_params(self):\n        return [p for n, p in self.named_parameters() if n.startswith("bb.") and p.requires_grad]\n\n\ndef amp():\n    return torch.autocast("cuda", dtype=AMP) if AMP is not None else torch.autocast("cpu", enabled=False)\n\n\ndef loss_fn(model, b, in_noise):\n    """Flow-matching MSE on every real frame (K draws) + stop BCE. Returns (total, flow, stop)."""\n    text, tlen, lat, flen, spk = (b[k].to(DEV, non_blocking=True) for k in ("text", "tlen", "lat", "flen", "spk"))\n    B, La, D = lat.shape\n    fed = lat + in_noise * torch.rand(B, 1, 1, device=DEV) * torch.randn_like(lat) if in_noise > 0 else lat\n    with amp():\n        te = model.bb.embed(text).float()\n        ae = model.in_proj(fed).float()\n        sv = model.spk(spk).float()\n        seqs = [torch.cat([te[i, :tlen[i]], (model.audio_start.float() + sv[i])[None], ae[i, :flen[i]] + sv[i]], 0)\n                for i in range(B)]\n        lens = torch.tensor([len(s) for s in seqs], device=DEV)\n        emb = nn.utils.rnn.pad_sequence(seqs, batch_first=True)\n        T = emb.shape[1]\n        mask = (torch.arange(T, device=DEV)[None] < lens[:, None]).long()\n        h, _ = model.bb(emb, attention_mask=mask)\n        s = torch.arange(La + 1, device=DEV)[None]\n        idx = (tlen[:, None] + s).clamp(max=T - 1)\n        hs = h.gather(1, idx[..., None].expand(-1, -1, h.shape[-1])).float()          # (B, La+1, d)\n        st_logit = model.stop(hs)[..., 0].float()\n        st_tgt = (s == flen[:, None]).float()\n        st_mask = (s <= flen[:, None]).float()\n        bce = F.binary_cross_entropy_with_logits(st_logit, st_tgt, reduction="none") * (1 + (CFG["POS_W"] - 1) * st_tgt)\n        stop = (bce * st_mask).sum() / st_mask.sum()\n        valid = torch.arange(La, device=DEV)[None] < flen[:, None]\n        hc = hs[:, :La][valid]\n        x1 = lat[valid]\n        prev = torch.cat([torch.zeros_like(lat[:, :1]), lat[:, :-1]], 1)[valid]        # Ming: zeros before frame 0\n        if CFG["CFG_DROP"] > 0:\n            hc = hc * (torch.rand(hc.shape[0], 1, device=DEV) >= CFG["CFG_DROP"]).float()\n        K = CFG["K"]\n        if K > 1:\n            hc, x1, prev = hc.repeat(K, 1), x1.repeat(K, 1), prev.repeat(K, 1)\n        x0 = torch.randn_like(x1)\n        t = torch.rand(x1.shape[0], device=DEV) if CFG["T_SAMPLING"] == "uniform" else torch.sigmoid(torch.randn(x1.shape[0], device=DEV))\n        xt = (1 - t[:, None]) * x0 + t[:, None] * x1\n        v = model.head(xt, t, hc, prev).float()\n        flow = ((v - (x1 - x0)) ** 2).mean()\n    return flow + CFG["STOP_W"] * stop, flow, stop\n\n# ---------------------------------------------------------------- generation (batched, KV cache) and evaluation\ndef time_grid(n, sched, dev):\n    if sched == "ming" and n == 10:                    # Ming\'s pruned grid + sway warp t\' = 1 - cos(pi t / 2)\n        t = torch.tensor([0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32], device=dev, dtype=torch.float32) / 32\n    else:\n        t = torch.linspace(0, 1, n + 1, device=dev)\n    return 1 - torch.cos(math.pi / 2 * t) if sched == "ming" else t\n\n\n@torch.no_grad()\ndef sample_frames(model, h, prev, cfg=3.0, steps=10, temp=1.0, sched="ming"):\n    """h (B,d), prev (B,D) normalized -> (B,D). Euler ODE; guidance v = v_u + cfg (v_c - v_u), v_u with h zeroed."""\n    x = torch.randn(h.shape[0], model.D, device=h.device) * temp\n    tg = time_grid(steps, sched, h.device)\n    h2 = torch.cat([h, torch.zeros_like(h)])\n    p2 = torch.cat([prev, prev])\n    for i in range(len(tg) - 1):\n        tt = tg[i].expand(2 * x.shape[0])\n        with amp():\n            v = model.head(torch.cat([x, x]), tt, h2, p2).float()\n        vc, vu = v.chunk(2)\n        x = x + (tg[i + 1] - tg[i]) * (vu + cfg * (vc - vu))\n    return x\n\n\n@torch.no_grad()\ndef generate(model, texts, spks, max_frames=250, min_frames=3, cfg=3.0, steps=10, temp=1.0, sched="ming", seed=0):\n    """texts: list of token-id lists, spks: list of ints -> list of (F, D) normalized latents, info.\n    Left-padded batch with explicit position ids, so every clip sees the same positions as in training."""\n    model.eval()\n    torch.manual_seed(seed)\n    B = len(texts)\n    P = max(len(t) for t in texts) + 1\n    emb = torch.zeros(B, P, model.bb.d, device=DEV)\n    mask = torch.zeros(B, P, dtype=torch.long, device=DEV)\n    sv = model.spk(torch.tensor(spks, device=DEV)).float()\n    for b, t in enumerate(texts):\n        e = torch.cat([model.bb.embed(torch.tensor(t, device=DEV)[None])[0].float(), (model.audio_start.float() + sv[b])[None]])\n        emb[b, P - len(e):] = e\n        mask[b, P - len(e):] = 1\n    pos = (mask.cumsum(1) - 1).clamp(min=0)\n    with amp():\n        h, past = model.bb(emb, attention_mask=mask, use_cache=True, position_ids=pos)\n    h = h[:, -1].float()\n    nxt = pos[:, -1:] + 1\n    prev = torch.zeros(B, model.D, device=DEV)\n    frames = [[] for _ in range(B)]\n    done = torch.zeros(B, dtype=torch.bool, device=DEV)\n    hit_max = torch.zeros(B, dtype=torch.bool, device=DEV)\n    for s in range(max_frames + 1):\n        p = torch.sigmoid(model.stop(h)[:, 0].float())\n        if s >= min_frames:\n            done |= p > 0.5\n        if s == max_frames:\n            hit_max = ~done\n            break\n        if bool(done.all()):\n            break\n        x = sample_frames(model, h, prev, cfg, steps, temp, sched)\n        for b in range(B):\n            if not done[b]:\n                frames[b].append(x[b].cpu().numpy())\n        mask = torch.cat([mask, torch.ones(B, 1, dtype=torch.long, device=DEV)], 1)\n        with amp():\n            h, past = model.bb((model.in_proj(x).float() + sv)[:, None], attention_mask=mask, past=past,\n                               use_cache=True, position_ids=nxt)\n        h = h[:, -1].float()\n        nxt = nxt + 1\n        prev = x\n    lats = [np.stack(f) if f else np.zeros((0, model.D), np.float32) for f in frames]\n    return lats, {"hit_max": int(hit_max.sum())}\n\n\ndef evaluate(model=None, tag=None, cfg=None, temp=None, steps=None, sched=None, n=None, batch=16):\n    """Generate 33\'s LJ val-100 in LJ\'s voice (speaker 0), decode, judge. Compare with f3\'s row of REF_33."""\n    model = model or STATE["model"]\n    cfg = cfg if cfg is not None else CFG["GEN_CFG"]\n    temp = temp if temp is not None else CFG["GEN_TEMP"]\n    steps, sched = steps or CFG["GEN_STEPS"], sched or CFG["GEN_SCHED"]\n    ev = STATE["eval"]\n    ids = [c for c in VAL100[: n or CFG["EVAL_N"]] if c in ev]\n    ids.sort(key=lambda c: len(ev[c][0][0]))            # similar lengths per batch = little padding\n    cod, mu, sd = get_codec(), STATE["mu"], STATE["sd"]\n    tag = tag or f"{CFG[\'RUN\']}_s{STATE[\'step\']}_cfg{cfg}_t{temp}_n{steps}{sched}"\n    out_dir = os.path.join(WORK, "eval", tag)\n    os.makedirs(out_dir, exist_ok=True)\n    items, t0, n_frames, hit = [], time.time(), 0, 0\n    for i in range(0, len(ids), batch):\n        chunk = ids[i:i + batch]\n        lats, info = generate(model, [list(ev[c][0][0]) for c in chunk], [0] * len(chunk), CFG["MAX_FRAMES"],\n                              cfg=cfg, steps=steps, temp=temp, sched=sched)\n        hit += info["hit_max"]\n        for c, z in zip(chunk, lats):\n            n_frames += len(z)\n            wav = cod.decode(z * sd + mu) if len(z) else np.zeros(0, np.float32)\n            sf.write(os.path.join(out_dir, c + ".wav"), wav, cod.sr)\n            items.append({"id": c, "wav": wav, "sr": cod.sr, "ref": ev[c][1]})\n    gen_s = time.time() - t0\n    r = judge().score(items, tag)\n    real_frames = sum(ev[c][0][2] for c in ids)\n    r.update(len_ratio=round(n_frames / max(real_frames, 1), 3), hit_max=hit, gen_s=round(gen_s, 1))\n    json.dump(r, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)\n    c3, w3, u3 = REF_33["f3 (33\'s model)"]\n    print(f"{tag}: CER {r[\'cer\']} / WER {r[\'wer\']} / UTMOS {r[\'utmos\']}   (f3 on val: {c3} / {w3} / {u3}); "\n          f"length ratio {r[\'len_ratio\']}, runaways {hit}", flush=True)\n    if CFG["PUSH"]:\n        from huggingface_hub import HfApi\n        HfApi().upload_folder(folder_path=out_dir, path_in_repo=f"{CFG[\'RUN\']}/eval/{tag}", repo_id=CKPT_REPO)\n    return r\n\n# ---------------------------------------------------------------- training with checkpoint / resume (local + HF)\nSTATE = {}\n\n\ndef ckpt_dir():\n    d = os.path.join(WORK, "ckpt", CFG["RUN"])\n    os.makedirs(d, exist_ok=True)\n    return d\n\n\ndef save_ckpt(model, opt, step, meta, push):\n    sd = {k: v for k, v in model.state_dict().items() if not k.startswith("bb.m.embed_tokens")}\n    p = os.path.join(ckpt_dir(), "last.pt")\n    torch.save({"model": sd, "opt": opt.state_dict(), "step": step, "meta": meta,\n                "rng": [random.getstate(), np.random.get_state(), torch.get_rng_state()]}, p + ".tmp")\n    os.replace(p + ".tmp", p)\n    json.dump(meta, open(os.path.join(ckpt_dir(), "meta.json"), "w"), indent=1)\n    if push and CFG["PUSH"]:\n        from huggingface_hub import HfApi\n        api = HfApi()\n        api.create_repo(CKPT_REPO, private=True, exist_ok=True)\n        api.upload_folder(folder_path=ckpt_dir(), path_in_repo=CFG["RUN"], repo_id=CKPT_REPO, allow_patterns=["last.pt", "meta.json", "log.jsonl"])\n    print(f"saved step {step}" + (" (pushed to HF)" if push and CFG["PUSH"] else ""), flush=True)\n\n\ndef find_ckpt():\n    p = os.path.join(ckpt_dir(), "last.pt")\n    if os.path.exists(p):\n        return p\n    if CFG["PUSH"] and f"{CFG[\'RUN\']}/last.pt" in hf_files(CKPT_REPO, "model"):\n        from huggingface_hub import hf_hub_download\n        for f in ("last.pt", "meta.json", "log.jsonl"):\n            try:\n                hf_hub_download(CKPT_REPO, f"{CFG[\'RUN\']}/{f}", local_dir=os.path.join(WORK, "ckpt"))\n            except Exception:\n                pass\n        return p\n    return None\n\n\ndef lr_lambda(step):\n    w = CFG["WARMUP"]\n    if step < w:\n        return 0.01 + 0.99 * step / max(1, w)\n    prog = min(1.0, (step - w) / max(1, CFG["STEPS"] - w))\n    return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog))\n\n\n@torch.no_grad()\ndef val_loss(model, val, store, D, mu, sd, in_noise):\n    model.eval()\n    st = (torch.get_rng_state(), torch.cuda.get_rng_state() if DEV == "cuda" else None)\n    torch.manual_seed(1234)\n    rng = np.random.default_rng(1234)\n    tot = fl = sp = 0.0\n    lens = [len(it[0]) + it[2] + 1 for it in val]\n    for bi in batches_for_epoch(lens, CFG["BUDGET"], 12345, 0):\n        b = make_batch([val[i] for i in bi], store, D, mu, sd, rng)\n        l, f, s = loss_fn(model, b, in_noise)\n        tot += float(l) * len(bi); fl += float(f) * len(bi); sp += float(s) * len(bi)\n    torch.set_rng_state(st[0])\n    if st[1] is not None:\n        torch.cuda.set_rng_state(st[1])\n    model.train()\n    n = len(val)\n    return tot / n, fl / n, sp / n\n\n\ndef train(max_steps_this_session=None):\n    """Train (or resume) CFG[\'RUN\'] until STEPS, SESSION_HOURS, or max_steps_this_session. Rerun after a disconnect."""\n    from transformers import AutoTokenizer\n    t_start = time.time()\n    tok = AutoTokenizer.from_pretrained(CFG["BACKBONE"])\n    train_items, val, ev, store, spk, mu, sd = load_data(tok)\n    D = {"ming": 64, "mimi": 512}[CFG["CODEC"]]\n    in_noise = CFG["IN_NOISE"] if CFG["IN_NOISE"] is not None else (0.0 if CFG["CODEC"] == "ming" else 0.2)\n    model = GemmaVoice(CFG["BACKBONE"], D, len(spk)).to(DEV)\n    opt = torch.optim.AdamW([{"params": model.bb_params(), "lr": CFG["LR"], "weight_decay": 0.01},\n                             {"params": model.new_params(), "lr": CFG["LR_NEW"], "weight_decay": 0.0}],\n                            betas=(0.9, 0.95), fused=(DEV == "cuda"))\n    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)\n    step = 0\n    meta = {"cfg": CFG, "codec": CFG["CODEC"], "D": D, "n_spk": len(spk), "lat_mean": mu.tolist(), "lat_std": sd.tolist()}\n    p = find_ckpt()\n    if p:\n        ck = torch.load(p, map_location=DEV, weights_only=False)\n        missing, unexpected = model.load_state_dict(ck["model"], strict=False)\n        assert not unexpected and all(k.startswith("bb.m.embed_tokens") for k in missing), (missing, unexpected)\n        opt.load_state_dict(ck["opt"])\n        step = ck["step"]\n        random.setstate(ck["rng"][0]); np.random.set_state(ck["rng"][1]); torch.set_rng_state(ck["rng"][2].cpu())\n        mu, sd = np.array(ck["meta"]["lat_mean"], np.float32), np.array(ck["meta"]["lat_std"], np.float32)\n        meta.update(lat_mean=ck["meta"]["lat_mean"], lat_std=ck["meta"]["lat_std"])\n        print(f"resumed {CFG[\'RUN\']} at step {step}", flush=True)\n    sched.last_epoch = step - 1\n    sched.step()\n    n_bb = sum(p.numel() for p in model.bb_params()); n_new = sum(p.numel() for p in model.new_params())\n    n_head = sum(p.numel() for p in model.head.parameters())\n    print(f"trainable: backbone {n_bb / 1e6:.1f}M, new {n_new / 1e6:.1f}M (flow head {n_head / 1e6:.1f}M)", flush=True)\n    STATE.update(model=model, eval=ev, mu=mu, sd=sd, step=step, spk=spk)\n    log = open(os.path.join(ckpt_dir(), "log.jsonl"), "a")\n    lens = [len(it[0]) + it[2] + 1 for it in train_items]\n    nb = len(batches_for_epoch(lens, CFG["BUDGET"], 0, 0))\n    rng = np.random.default_rng(step + 7)\n    last_save = time.time()\n    stop_at = CFG["STEPS"] if max_steps_this_session is None else min(CFG["STEPS"], step + max_steps_this_session)\n\n    def producer(q, start_step):                       # one batch ahead on a CPU thread\n        s = start_step\n        while s < stop_at:\n            ep, k = divmod(s, nb)\n            order = batches_for_epoch(lens, CFG["BUDGET"], 0, ep)\n            for bi in order[k:]:\n                q.put(make_batch([train_items[i] for i in bi], store, D, mu, sd, rng))\n                s += 1\n                if s >= stop_at:\n                    break\n        q.put(None)\n    q = queue.Queue(maxsize=3)\n    threading.Thread(target=producer, args=(q, step), daemon=True).start()\n    if step == 0:\n        vl = val_loss(model, val, store, D, mu, sd, in_noise)\n        print(json.dumps({"step": 0, "val": [round(x, 4) for x in vl]}), flush=True)\n    model.train()\n    tw, nt, t_log = time.time(), 0, time.time()\n    while True:\n        b = q.get()\n        if b is None:\n            break\n        loss, flow, stop = loss_fn(model, b, in_noise)\n        opt.zero_grad(set_to_none=True)\n        loss.backward()\n        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n        opt.step(); sched.step()\n        step += 1\n        STATE["step"] = step\n        nt += int(b["text"].shape[0] * (b["text"].shape[1] + b["lat"].shape[1] + 1))\n        if step % CFG["LOG_EVERY"] == 0:\n            dt = time.time() - tw\n            sps = CFG["LOG_EVERY"] / max(time.time() - t_log, 1e-6)\n            eta_h = (CFG["STEPS"] - step) / max(sps, 1e-6) / 3600\n            rec = {"step": step, "loss": round(float(loss), 4), "flow": round(float(flow), 4), "stop": round(float(stop), 4),\n                   "gnorm": round(float(gn), 3), "lr": round(sched.get_last_lr()[0], 7), "tok_s": round(nt / dt),\n                   "eta_h": round(eta_h, 2), "eta_units": round(eta_h * CFG["UNITS_PER_HOUR"], 1),\n                   "mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if DEV == "cuda" else None}\n            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\\n"); log.flush()\n            tw, nt, t_log = time.time(), 0, time.time()\n        if step % CFG["VAL_EVERY"] == 0:\n            vl = val_loss(model, val, store, D, mu, sd, in_noise)\n            rec = {"step": step, "val": [round(x, 4) for x in vl]}\n            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\\n"); log.flush()\n        out_of_time = (time.time() - t_start) / 3600 > CFG["SESSION_HOURS"]\n        timed = CFG["SAVE_MIN"] > 0 and (time.time() - last_save) / 60 > CFG["SAVE_MIN"]   # 0 = only at the end\n        if timed or step >= stop_at or out_of_time:\n            save_ckpt(model, opt, step, meta, push=True)\n            last_save = time.time()\n        if step >= stop_at or out_of_time:\n            break\n    print(f"stopped at step {step} of {CFG[\'STEPS\']} ({(time.time() - t_start) / 60:.1f} min this session)", flush=True)\n    return model\n\n"""Full-data Colab trainer. Requires the notebook\'s dataset/model/eval definitions.\n\nNo local execution: this module targets the authorized Colab A100 only.\n"""\nimport hashlib\nimport copy\nimport gc\n\nTRAINING_VERSION = 2\nTRAINING_KEYS = (\'CODEC\', \'BACKBONE\', \'SETS\', \'LJ_REPEAT\', \'HEAD_W\', \'HEAD_DEPTH\',\n    \'HEAD_HEADS\', \'BUDGET\', \'K\', \'CFG_DROP\', \'STOP_W\', \'POS_W\', \'T_SAMPLING\',\n    \'LATENT\', \'IN_NOISE\', \'LR\', \'LR_NEW\', \'WARMUP\', \'TRAIN_EPOCHS\', \'SEED\')\n\n\ndef stable_hash(value):\n    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(\',\', \':\')).encode()).hexdigest()\n\n\ndef capture_rng():\n    return dict(python=random.getstate(), numpy=np.random.get_state(),\n                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())\n\n\ndef restore_rng(value):\n    random.setstate(value[\'python\']); np.random.set_state(value[\'numpy\'])\n    torch.set_rng_state(value[\'torch\'].cpu())\n    torch.cuda.set_rng_state_all([s.cpu() for s in value[\'cuda\']])\n\n\ndef find_ckpt():\n    p = os.path.join(ckpt_dir(), \'last.pt\')\n    if os.path.exists(p):\n        return p\n    if CFG[\'PUSH\']:\n        from huggingface_hub import HfApi, hf_hub_download\n        files = set(HfApi().list_repo_files(CKPT_REPO))\n        prefix = CFG[\'RUN\']+\'/\'\n        if prefix+\'last.pt\' in files:\n            for name in (\'last.pt\',\'meta.json\',\'dataset.json\',\'log.jsonl\',\'resume_check.json\',\'calibration.json\'):\n                if prefix+name in files:\n                    hf_hub_download(CKPT_REPO, prefix+name, local_dir=os.path.join(WORK,\'ckpt\'))\n            return p\n    return None\n\n\ndef frozen_data(tok):\n    """Reject changed shards, speaker IDs, normalization, tokenizer or training policy on resume."""\n    items, val, ev, store, speakers, mu, sd = load_data(tok)\n    root = os.path.join(WORK, \'data\', CFG[\'CODEC\'])\n    manifest = []\n    for source in CFG[\'SETS\']:\n        for jp in sorted(glob.glob(os.path.join(root, source, \'*.json\'))):\n            with open(jp, \'rb\') as f:\n                digest = hashlib.sha256(f.read()).hexdigest()\n            manifest.append([os.path.relpath(jp, root), digest, os.path.getsize(jp[:-5]+\'.f16\')])\n    # Include actual tokenizer vocabulary/config, not only its mutable repository name.\n    identity = dict(version=TRAINING_VERSION, manifest=manifest, speakers=speakers,\n        mu=mu.tolist(), sd=sd.tolist(), tokenizer_vocab=stable_hash(tok.get_vocab()),\n        bos=tok.bos_token_id, config={k: CFG[k] for k in TRAINING_KEYS},\n        runtime_source=globals().get(\'TRAINING_SOURCE_HASH\'))\n    path = os.path.join(ckpt_dir(), \'dataset.json\')\n    digest = stable_hash(identity)\n    if os.path.exists(path):\n        with open(path) as f:\n            assert stable_hash(json.load(f)) == digest, \'Frozen dataset/training identity changed; use a new RUN.\'\n    else:\n        with open(path+\'.tmp\', \'w\') as f:\n            json.dump(identity, f)\n        os.replace(path+\'.tmp\', path)\n    lens = np.asarray([len(it[0])+it[2]+1 for it in items])\n    counts = [len(batches_for_epoch(lens, CFG[\'BUDGET\'], CFG[\'SEED\'], ep))\n              for ep in range(CFG[\'TRAIN_EPOCHS\'])]\n    CFG[\'STEPS\'] = sum(counts)\n    print(\'TRAINING_PLAN\', json.dumps(dict(epochs=CFG[\'TRAIN_EPOCHS\'], batches_per_epoch=counts,\n        total_steps=CFG[\'STEPS\'], train_clips=len(items), speakers=len(speakers),\n        padded_token_budget=CFG[\'BUDGET\'], data_identity=digest)), flush=True)\n    return dict(items=items, val=val, ev=ev, store=store, speakers=speakers,\n                mu=mu, sd=sd, lens=lens, epoch_counts=counts, identity=digest)\n\n\ndef build_training_context():\n    from transformers import AutoTokenizer\n    random.seed(CFG[\'SEED\']); np.random.seed(CFG[\'SEED\']); torch.manual_seed(CFG[\'SEED\'])\n    p = find_ckpt()\n    data = frozen_data(AutoTokenizer.from_pretrained(CFG[\'BACKBONE\']))\n    D = {\'ming\':64, \'mimi\':512}[CFG[\'CODEC\']]\n    net = GemmaVoice(CFG[\'BACKBONE\'], D, len(data[\'speakers\'])).to(DEV)\n    opt = torch.optim.AdamW([\n        {\'params\':net.bb_params(), \'lr\':CFG[\'LR\'], \'weight_decay\':0.01},\n        {\'params\':net.new_params(), \'lr\':CFG[\'LR_NEW\'], \'weight_decay\':0.0}],\n        betas=(0.9, 0.95), fused=True)\n    ctx = dict(data=data, model=net, opt=opt, step=0, epoch=0, batch=0, audio_frames=0,\n               order=None, order_epoch=None)\n    if p:\n        ck = torch.load(p, map_location=\'cpu\', weights_only=False)\n        assert ck.get(\'version\') == TRAINING_VERSION, \'Legacy checkpoint cannot resume exactly; use a new RUN.\'\n        assert ck[\'identity\'] == data[\'identity\'], \'Checkpoint dataset/config mismatch\'\n        assert ck[\'backbone_revision\'] == net.bb.m.config._commit_hash, \'Backbone revision changed\'\n        missing, unexpected = net.load_state_dict(ck[\'model\'], strict=False)\n        assert not unexpected and all(k.startswith(\'bb.m.embed_tokens\') for k in missing)\n        opt.load_state_dict(ck[\'opt\'])\n        for k in (\'step\', \'epoch\', \'batch\', \'audio_frames\'):\n            ctx[k] = ck[k]\n        assert ctx[\'step\'] == sum(data[\'epoch_counts\'][:ctx[\'epoch\']])+ctx[\'batch\']\n        restore_rng(ck[\'rng\'])\n        del ck\n        print(\'EXACT_RESUME\', json.dumps({k:ctx[k] for k in (\'step\',\'epoch\',\'batch\')}), flush=True)\n    STATE.update(model=net, eval=data[\'ev\'], mu=data[\'mu\'], sd=data[\'sd\'],\n                 step=ctx[\'step\'], spk=data[\'speakers\'])\n    return ctx\n\n\ndef next_training_batch(ctx):\n    ep, k = ctx[\'epoch\'], ctx[\'batch\']\n    d = ctx[\'data\']\n    assert ep < CFG[\'TRAIN_EPOCHS\']\n    if ctx[\'order_epoch\'] != ep:\n        ctx[\'order\'] = batches_for_epoch(d[\'lens\'], CFG[\'BUDGET\'], CFG[\'SEED\'], ep)\n        ctx[\'order_epoch\'] = ep\n    bi = ctx[\'order\'][k]\n    # Posterior sampling is keyed by the exact update, independent of prefetch/session length.\n    rng = np.random.default_rng(np.random.SeedSequence([CFG[\'SEED\'], ctx[\'step\'], 7]))\n    b = make_batch([d[\'items\'][i] for i in bi], d[\'store\'], ctx[\'model\'].D, d[\'mu\'], d[\'sd\'], rng)\n    return b\n\n\ndef optimizer_update(ctx):\n    net, opt = ctx[\'model\'], ctx[\'opt\']\n    net.train()\n    b = next_training_batch(ctx)\n    for group, base in zip(opt.param_groups, (CFG[\'LR\'], CFG[\'LR_NEW\'])):\n        group[\'lr\'] = base * lr_lambda(ctx[\'step\'])\n    in_noise = CFG[\'IN_NOISE\'] if CFG[\'IN_NOISE\'] is not None else (0.0 if CFG[\'CODEC\']==\'ming\' else 0.2)\n    loss, flow, stop = loss_fn(net, b, in_noise)\n    assert torch.isfinite(loss), \'Non-finite training loss; checkpoint preserved\'\n    opt.zero_grad(set_to_none=True)\n    loss.backward()\n    gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)\n    opt.step()\n    ctx[\'step\'] += 1; ctx[\'batch\'] += 1\n    ctx[\'audio_frames\'] += int(b[\'flen\'].sum())\n    if ctx[\'batch\'] == ctx[\'data\'][\'epoch_counts\'][ctx[\'epoch\']]:\n        ctx[\'epoch\'] += 1; ctx[\'batch\'] = 0\n    STATE[\'step\'] = ctx[\'step\']\n    return dict(step=ctx[\'step\'], loss=float(loss), flow=float(flow), stop=float(stop),\n                gnorm=float(gn), lr=opt.param_groups[0][\'lr\'], clips=len(b[\'flen\']),\n                audio_seconds=float(b[\'flen\'].sum())/12.5)\n\n\ndef save_training(ctx, push=True):\n    """Atomic local checkpoint; one Hub commit publishes weights + identity + logs."""\n    net, opt, d = ctx[\'model\'], ctx[\'opt\'], ctx[\'data\']\n    ck = dict(version=TRAINING_VERSION,\n        model={k:v.detach().cpu() for k,v in net.state_dict().items() if not k.startswith(\'bb.m.embed_tokens\')},\n        opt=opt.state_dict(), identity=d[\'identity\'], cfg=copy.deepcopy(CFG), rng=capture_rng(),\n        backbone_revision=net.bb.m.config._commit_hash,\n        **{k:ctx[k] for k in (\'step\',\'epoch\',\'batch\',\'audio_frames\')})\n    p = os.path.join(ckpt_dir(), \'last.pt\')\n    torch.save(ck, p+\'.tmp\'); os.replace(p+\'.tmp\', p)\n    meta = dict(step=ctx[\'step\'], epoch=ctx[\'epoch\'], batch=ctx[\'batch\'],\n        audio_hours_seen=ctx[\'audio_frames\']/45000, identity=d[\'identity\'], cfg=copy.deepcopy(CFG))\n    with open(os.path.join(ckpt_dir(), \'meta.json\'), \'w\') as f:\n        json.dump(meta, f)\n    if push and CFG[\'PUSH\']:\n        from huggingface_hub import HfApi\n        api = HfApi()\n        api.create_repo(CKPT_REPO, private=True, exist_ok=True)\n        api.upload_folder(folder_path=ckpt_dir(), path_in_repo=CFG[\'RUN\'], repo_id=CKPT_REPO,\n            allow_patterns=[\'last.pt\',\'meta.json\',\'dataset.json\',\'log.jsonl\',\'resume_check.json\',\'calibration.json\'])\n    print(\'CHECKPOINT_SAVED\', json.dumps(dict(step=ctx[\'step\'], pushed=bool(push and CFG[\'PUSH\']))), flush=True)\n\n\ndef state_digest(ctx):\n    """Compare every trained weight, optimizer tensor, and RNG state after the replayed update."""\n    h = hashlib.sha256()\n    def add(x):\n        if isinstance(x, torch.Tensor):\n            h.update(x.detach().cpu().contiguous().numpy().tobytes())\n        elif isinstance(x, dict):\n            for k in sorted(x, key=str):\n                h.update(str(k).encode()); add(x[k])\n        elif isinstance(x, (tuple,list)):\n            for v in x: add(v)\n        else:\n            h.update(repr(x).encode())\n    add({k:v for k,v in ctx[\'model\'].state_dict().items() if not k.startswith(\'bb.m.embed_tokens\')})\n    add(ctx[\'opt\'].state_dict()); add(capture_rng())\n    add([ctx[\'step\'],ctx[\'epoch\'],ctx[\'batch\'],ctx[\'audio_frames\']])\n    return h.hexdigest()\n\n\ndef cold_resume_reference():\n    """Save after 3 updates, record uninterrupted update 4, then deliberately terminate this worker."""\n    import signal\n    ctx = build_training_context()\n    assert ctx[\'step\'] == 0, \'Resume test expects a fresh isolated RUN\'\n    for _ in range(3): optimizer_update(ctx)\n    save_training(ctx, push=False)\n    rec = optimizer_update(ctx)\n    rec[\'state_sha256\'] = state_digest(ctx)\n    with open(os.path.join(ckpt_dir(),\'resume_expected.json\'), \'w\') as f:\n        json.dump(rec,f)\n    print(\'REFERENCE_READY: saved step 3; recorded uninterrupted step 4; terminating test worker\',flush=True)\n    os.kill(os.getpid(), signal.SIGTERM)\n\n\ndef cold_resume_replay():\n    ctx = build_training_context()\n    assert ctx[\'step\'] == 3\n    rec = optimizer_update(ctx)\n    rec[\'state_sha256\'] = state_digest(ctx)\n    with open(os.path.join(ckpt_dir(),\'resume_expected.json\')) as f:\n        expected = json.load(f)\n    assert rec == expected, {\'expected\':expected, \'actual\':rec}\n    result = dict(passed=True, exact_state_match=True, resumed_step=3, compared_step=4,\n                  loss=rec[\'loss\'], state_sha256=rec[\'state_sha256\'], identity=ctx[\'data\'][\'identity\'])\n    with open(os.path.join(ckpt_dir(),\'resume_check.json\'), \'w\') as f:\n        json.dump(result,f)\n    print(\'COLD_RESUME_PASSED\', json.dumps(result), flush=True)\n\n\ndef train_verified(max_steps_this_session=None, session_hours=None):\n    ctx = build_training_context()\n    check_path = os.path.join(ckpt_dir(),\'resume_check.json\')\n    assert os.path.exists(check_path), \'Run cold checkpoint test before sustained training\'\n    with open(check_path) as f:\n        check = json.load(f)\n    assert check[\'passed\'] and check[\'identity\'] == ctx[\'data\'][\'identity\']\n    start_step = ctx[\'step\']; start = time.monotonic(); last_save = start\n    stop_at = CFG[\'STEPS\'] if max_steps_this_session is None else min(CFG[\'STEPS\'], start_step+max_steps_this_session)\n    elapsed_steps, audio_seconds = [], []\n    log_path = os.path.join(ckpt_dir(), \'log.jsonl\')\n    try:\n        with open(log_path, \'a\') as log:\n            while ctx[\'step\'] < stop_at:\n                t0 = time.monotonic()\n                rec = optimizer_update(ctx)\n                dt = time.monotonic()-t0\n                elapsed_steps.append(dt); audio_seconds.append(rec[\'audio_seconds\'])\n                if ctx[\'step\'] % CFG[\'LOG_EVERY\'] == 0:\n                    recent = elapsed_steps[-CFG[\'LOG_EVERY\']:]\n                    speed = sum(recent)/len(recent)\n                    rec.update(epoch=ctx[\'epoch\']+ctx[\'batch\']/ctx[\'data\'][\'epoch_counts\'][min(ctx[\'epoch\'],CFG[\'TRAIN_EPOCHS\']-1)],\n                        seconds_per_step=speed, eta_hours=(CFG[\'STEPS\']-ctx[\'step\'])*speed/3600,\n                        audio_hours_seen=ctx[\'audio_frames\']/45000,\n                        gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)\n                    print(\'TRAIN\',json.dumps(rec),flush=True); log.write(json.dumps(rec)+\'\\n\'); log.flush()\n                if ctx[\'step\'] % CFG[\'VAL_EVERY\'] == 0:\n                    d = ctx[\'data\']\n                    vl = val_loss(ctx[\'model\'], d[\'val\'], d[\'store\'], ctx[\'model\'].D, d[\'mu\'], d[\'sd\'],0.0)\n                    rec = dict(step=ctx[\'step\'], val=list(vl))\n                    print(\'VALIDATION\',json.dumps(rec),flush=True); log.write(json.dumps(rec)+\'\\n\'); log.flush()\n                if time.monotonic()-last_save >= CFG[\'SAVE_MIN\']*60:\n                    save_training(ctx); last_save=time.monotonic()\n                if session_hours is not None and time.monotonic()-start >= session_hours*3600:\n                    break\n    except KeyboardInterrupt:\n        print(\'Interrupted: saving completed updates.\',flush=True)\n    save_training(ctx)\n    timing = dict(start_step=start_step, end_step=ctx[\'step\'], elapsed_minutes=(time.monotonic()-start)/60,\n        median_seconds_per_step=float(np.median(elapsed_steps[5:] or elapsed_steps)) if elapsed_steps else None,\n        mean_seconds_per_step=float(np.mean(elapsed_steps[5:] or elapsed_steps)) if elapsed_steps else None,\n        epoch_counts=ctx[\'data\'][\'epoch_counts\'], audio_hours_seen=ctx[\'audio_frames\']/45000,\n        gpu_peak_gb=torch.cuda.max_memory_allocated()/2**30)\n    with open(os.path.join(ckpt_dir(),\'calibration.json\'), \'w\') as f: json.dump(timing,f)\n    print(\'TRAINING_SESSION_SAVED\',json.dumps(timing),flush=True)\n    return ctx\n\nif __name__ == \'__main__\':\n    action = sys.argv[2]\n    if action == \'restore\':\n        print(\'CHECKPOINT_RESTORE\', bool(find_ckpt()), flush=True)\n    elif action == \'reference\':\n        cold_resume_reference()\n    elif action == \'replay\':\n        cold_resume_replay()\n    elif action == \'calibrate\':\n        ctx = train_verified(max_steps_this_session=100, session_hours=0.25)\n    elif action == \'train\':\n        ctx = train_verified(max_steps_this_session=CFG.get(\'SESSION_STEPS\'), session_hours=CFG.get(\'SESSION_HOURS\'))\n    elif action == \'evaluate\':\n        with open(os.path.join(WORK,\'training_evaluation.py\')) as f:\n            exec(compile(f.read(), \'training_evaluation.py\', \'exec\'))\n        evaluate_training_checkpoint()\n    else:\n        raise ValueError(action)\n')
with open(training_config_path,'w') as f:
    json.dump(dict(CFG=CFG,WORK=WORK,DATA_REPO=DATA_REPO,CKPT_REPO=CKPT_REPO),f)
def run_training_worker(action):
    from huggingface_hub import get_token
    worker_env = os.environ.copy()
    worker_env['HF_TOKEN'] = get_token() or ''
    proc = subprocess.Popen([sys.executable,'-u',training_worker_path,training_config_path,action],
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=worker_env)
    try:
        for line in proc.stdout:
            print(line,end='',flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        for line in proc.stdout:
            print(line,end='',flush=True)
        proc.wait()
        raise
# First process saves step 3, computes the uninterrupted step 4, then terminates.
# A completely fresh process reloads step 3 and must reproduce every state byte.
resume_report = os.path.join(WORK,'ckpt',CFG['RUN'],'resume_check.json')
assert run_training_worker('restore') == 0, 'Checkpoint restoration failed'
if not os.path.exists(resume_report):
    expected_report = os.path.join(WORK,'ckpt',CFG['RUN'],'resume_expected.json')
    if not os.path.exists(expected_report):
        rc = run_training_worker('reference')
        assert rc == -signal.SIGTERM, f'Reference worker failed: {rc}'
    assert run_training_worker('replay') == 0, 'Cold resume replay failed'
if not os.path.exists(os.path.join(WORK,'ckpt',CFG['RUN'],'calibration.json')):
    assert run_training_worker('calibrate') == 0, 'Calibration failed'
print('CALIBRATION_READY: choose session duration from measured throughput before the sustained run.',flush=True)

# Continue automatically through the first full epoch.
with open(os.path.join(WORK,"training_evaluation.py"),"w") as f: f.write('"""Fixed validation speech panels and checkpoint provenance, run between training segments."""\ndef evaluate_training_checkpoint():\n    from huggingface_hub import HfApi\n    ctx = build_training_context()\n    step = ctx[\'step\']\n    original_ids = list(VAL100)\n    original_eval = STATE[\'eval\']\n    n = 100 if ctx[\'epoch\'] > 0 and ctx[\'batch\'] == 0 else 24\n    results = {\'lj\': evaluate(n=n, tag=f\'{CFG["RUN"]}_s{step}_lj{n}\')}\n    # Unseen Emilia transcripts spoken with the learned LJ voice. This measures\n    # text generalization, not zero-shot speaker similarity (no trained ID exists for many heldouts).\n    rows = []\n    for jp in sorted(glob.glob(os.path.join(WORK,\'data\',\'ming\',\'emilia_yodas\',\'*.json\'))):\n        with open(jp) as f: m=json.load(f)\n        for i, cid in enumerate(m[\'ids\']):\n            if m[\'splits\'][i] == \'val\' and 20 <= len(m[\'texts\'][i]) <= 200:\n                rows.append((hashlib.sha256(cid.encode()).hexdigest(), cid, m[\'texts\'][i], m[\'offs\'][i+1]-m[\'offs\'][i]))\n    rows.sort()\n    from transformers import AutoTokenizer\n    tok=AutoTokenizer.from_pretrained(CFG[\'BACKBONE\'])\n    panel={cid:((tuple(text_ids(tok,text)),0,frames,0),text) for _,cid,text,frames in rows[:16]}\n    try:\n        VAL100[:] = list(panel)\n        STATE[\'eval\'] = panel\n        results[\'emilia_text_lj_voice\'] = evaluate(n=16,tag=f\'{CFG["RUN"]}_s{step}_emilia16_ljvoice\')\n    finally:\n        VAL100[:] = original_ids\n        STATE[\'eval\'] = original_eval\n    api=HfApi()\n    # Evaluation uploads are also commits, but last.pt remains the just-saved checkpoint.\n    revision=api.model_info(CKPT_REPO).sha if CFG[\'PUSH\'] else None\n    report=dict(step=step,epoch=ctx[\'epoch\'],batch=ctx[\'batch\'],checkpoint_revision=revision,\n                results={k:{kk:vv for kk,vv in v.items() if kk!=\'clips\'} for k,v in results.items()})\n    target=os.path.join(ckpt_dir(),f\'evaluation_{step}.json\')\n    with open(target,\'w\') as f: json.dump(report,f,indent=2)\n    if CFG[\'PUSH\']:\n        api.upload_file(path_or_fileobj=target,path_in_repo=f\'{CFG["RUN"]}/evaluation_{step}.json\',repo_id=CKPT_REPO)\n    print(\'SPEECH_EVALUATION_SAVED\',json.dumps(report),flush=True)\n    return report\n')

resume_report = os.path.join(WORK,'ckpt',CFG['RUN'],'resume_check.json')
assert os.path.exists(resume_report), 'Complete the cold-resume check first.'
with open(os.path.join(WORK,'ckpt',CFG['RUN'],'calibration.json')) as f:
    calibration = json.load(f)
first_epoch_steps = calibration['epoch_counts'][0]
review_steps = sorted(set([5000] + [round(first_epoch_steps*f) for f in (0.25,0.5,0.75,1.0)]))
# Keep the precise training code alongside checkpoints for recovery in a fresh Colab.
from huggingface_hub import HfApi, CommitOperationAdd
HfApi().create_commit(repo_id=CKPT_REPO,commit_message='Preserve verified full-corpus training code',
    operations=[CommitOperationAdd(path_in_repo=f'{CFG["RUN"]}/code/'+os.path.basename(p),path_or_fileobj=p)
                for p in (training_worker_path,os.path.join(WORK,'training_evaluation.py'))])
print('FULL_EPOCH_STARTED',json.dumps(dict(run=CFG['RUN'],target_step=first_epoch_steps,
      planned_epochs=CFG['TRAIN_EPOCHS'],speech_review_steps=review_steps)),flush=True)
while True:
    with open(os.path.join(WORK,'ckpt',CFG['RUN'],'meta.json')) as f:
        before = json.load(f)['step']
    if before >= first_epoch_steps:
        break
    target = next(s for s in review_steps if s > before)
    CFG.update(SESSION_STEPS=target-before,SESSION_HOURS=40/60)
    with open(training_config_path,'w') as f:
        json.dump(dict(CFG=CFG,WORK=WORK,DATA_REPO=DATA_REPO,CKPT_REPO=CKPT_REPO),f)
    assert run_training_worker('train') == 0, 'Training stopped; last completed checkpoint preserved.'
    with open(os.path.join(WORK,'ckpt',CFG['RUN'],'meta.json')) as f:
        after=json.load(f)['step']
    assert after > before, 'Training made no progress.'
    if after in review_steps:
        assert run_training_worker('evaluate') == 0, 'Speech evaluation failed; training checkpoint is safe.'
    if after < target:
        print('AUTOMATIC_CONTINUATION: checkpoint saved; loading the next segment.',flush=True)
# Also handles reconnecting after the final update but before final speech evaluation.
final_report = os.path.join(WORK,'ckpt',CFG['RUN'],f'evaluation_{first_epoch_steps}.json')
if not os.path.exists(final_report):
    assert run_training_worker('evaluate') == 0
print('FIRST_EPOCH_READY: all training examples seen once; review generated speech before epochs 2–3.',flush=True)

# %%
# Re-evaluate the latest saved full-data checkpoint, without training.
assert run_training_worker('evaluate') == 0
