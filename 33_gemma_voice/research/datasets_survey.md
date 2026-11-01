# More English TTS data for 33_gemma_voice (Gemma 3 270M → Mimi, 24 kHz)

How I judged each set, in priority order:
- (a) native audio at ≥ 24 kHz, because 16 kHz sources would teach the model muffled output (my inference);
- (b) cased, punctuated transcripts;
- (c) clean or studio audio;
- (d) fits the budget of a few hundred to about 2k hours;
- (e) audio hosted on HF, versus metadata only with the audio fetched separately;
- (f) download size.

Your code uses 4 to 8 Mimi codebooks (`--nq 4/8/16` in `logs/`). `mimi_io.py` asserts the RVQ prefix property, so a pre-encoded set with 8 codebooks is enough for you.

## Shortcut: datasets already encoded as Mimi codes

`shangeth` has published `kyutai/mimi` codes (24 kHz, 12.5 fps, all 8 codebooks, sliceable as `codes[:k]`) for these sets:
- LibriTTS-R, all 7 splits, mixed-case text with punctuation, CC BY 4.0: https://huggingface.co/datasets/shangeth/libritts-r-mimi-codes
- Expresso, Jenny, VCTK, LJSpeech, MLS (no English) and LibriSpeech: https://huggingface.co/shangeth

**Inference:** these could save the LibriTTS-R download (74 GB) and roughly 4 h of encoding for clean-360 and other-500. The codes come from the PyTorch Mimi, while yours come from `moshi_mlx` `mimi_202407`. Before mixing them in, re-encode about 50 clips and check that the tokens match exactly, or nearly.

---

## Candidates

### Read, multi-speaker
| Dataset | Hours / speakers | SR | Transcripts | License / access | Size | Verdict |
|---|---|---|---|---|---|---|
| **LibriTTS-R train-clean-360** | 191.3 h / 904 [1] | 24 kHz | `text_original` cased and punctuated; `text_normalized` also given [2] | CC BY 4.0 [3] (the HF card's BibTeX says NC-ND [2]); open | 28 GB [3] | **Top pick.** Same domain as your clean-100 and fills in the same speaker distribution. |
| **LibriTTS-R train-other-500** | 310.1 h / 1,160 [1] | 24 kHz | same as above | same as above | 46 GB [3] | **Strong.** Restoration lifts the "other" quality. Parler's filtered version drops failed restorations and multi-speaker voices [4]. |
| **NVIDIA HiFiTTS-2** | 36.7k h at 22 kHz / ~5k speakers; 31.7k h at 44 kHz [5][6] | fetched at 48 kHz from LibriVox | `text` (book or MLS source) plus `normalized_text`, with per-utterance WER/CER and per-chapter bandwidth [5]. **Inference:** the `book`-source text is cased; the `mls` text is lowercase. | CC BY 4.0. Metadata only (~17.8 GB); audio comes from LibriVox via a NeMo SDP script [5] | 2.8 TB for the full 22 kHz set [5] | **Best scalable read source.** Filter by CER, bandwidth, speaker_count and `text_source=book`. **Inference:** about 76 GB per 1k h, plus a few hours of archive.org download time per 1k h. Some overlap with your Hi-Fi TTS books is likely. |
| MLS English | 44.5k h / 5,490 [7][8] | 16 kHz (inference; archive sources are 64 kbps MP3 [8]) | lowercase, no punctuation [8] | CC BY 4.0; hosted | 651 GB Opus / 2.4 TB FLAC [9] | **Skip.** HiFiTTS-2 is the same LibriVox material with better text and bandwidth filtering. |
| Libriheavy | ~50k h [10] | 16 kHz (Libri-Light; inference) | cased and punctuated [10] | Apache-2.0 (labels); audio from Libri-Light: small 577 h / 35 GB, medium 5.2k h / 321 GB [11] | metadata 15.9 GB [10] | **Skip for 24 kHz.** Good text but band-limited audio. Used by T5Gemma-TTS [12]. |
| VCTK | 44 h / 110 accents [13] | 48 kHz | cased short sentences [13] | CC BY 4.0; the HF repo uses a loading script | n/a | **Mild help.** Accent diversity, but short and repetitive prompts. Mimi codes also exist [shangeth]. |
| GLOBE_V2 (from Common Voice) | 535 h / 23,519 / 164 accents [14] | 44.1 kHz *upsampled and enhanced* | mostly lowercase [14] | CC0; open | 76 GB [14] | **Skip/low.** Enhancement artifacts and lowercase text. |
| Common Voice filtered (UTMOS > 3.75, ~7% kept) | small MVP [15] | CV | CV text | CC BY-SA 4.0 | small | **Skip.** |
| GigaSpeech | 10k h (XL), with S = 250 h and M = 1k h [16] | 16 kHz | UPPERCASE with `<COMMA>`-style tags [16] | gated, non-commercial, Google form [16] | 2.6 TB | **Skip.** 16 kHz audio and poor text. |
| People's Speech | 30k+ h [17] | 16 kHz | lowercase, partly misaligned, noisy [17] | CC BY / BY-SA | 2.12 TB | **Skip.** The card itself warns it may be unsuitable for TTS. |
| **Raon-OpenTTS-Pool** (KRAFTON, May 2026) | 615k h; Core subset 510k h [18][19] | **16 kHz Opus 64 kbps** [18] | Whisper-large-v3 for its YouTube portion; other sets as-is [18] | mixed per source; GigaSpeech and SPGI must be fetched separately [18] | huge | **Skip** for a 24 kHz target. Its per-source DNSMOS/WER table is useful for ranking sources: LibriTTS-R has the lowest WER at 0.06 [18]. |

### Single-speaker studio
| Dataset | Hours | SR | Transcripts | License / access | Size | Verdict |
|---|---|---|---|---|---|---|
| **Jenny (Dioco)** | ~30 h, 1 Irish female [20] | 48 kHz | `transcription` cased and punctuated, plus a normalized version [20] | custom Dioco license (attribution "Jenny (Dioco)", commercial OK); open | 3.74 GB [20] | **Cheap, good add.** Very varied text: headlines, recipes, Reddit, *Friends*, Wikipedia. Needs silence and keyclick trimming [20]. Mimi codes exist [shangeth]. |
| RyanSpeech | 10+ h, 1 male actor [21] | 44.1 kHz [21] | original via a request form [22]; the HF mirror is lowercase with punctuation removed [23] | CC BY-NC 4.0 [22] | 1.09 GB (mirror) [23] | **Low.** Small, and only the lowercase mirror is easy to get. |

### Expressive / conversational
| Dataset | Hours / speakers | SR | Transcripts | License / access | Size | Verdict |
|---|---|---|---|---|---|---|
| **Expresso** | 45.9 h (11.5 read + 34.4 improvised) / 4 [24][25] | 48 kHz | Read: human-written, cased, with `*emphasis*`. Conversational: Whisper-large-v3-turbo on `shangeth/expresso` (3.0% WER on a read check) [26]. Another version was transcribed with Parakeet (gated) [27]. | CC BY-NC 4.0 | 5.76 GB read only [24]; 13.8 GB full [26]. Mimi codes exist [shangeth]. | **Top expressive pick.** Studio audio with 34 styles. |
| **Emilia-YODAS (EN)** | 92.2k h [28] | 24 kHz (Emilia-Pipe default [29]; inferred for YODAS) | WhisperX, cased and punctuated [29]; per-clip `dnsmos` field [28] | **CC BY 4.0 for YODAS, NC for Emilia** per amphion [28]. Raon's table states the reverse [18], so trust amphion. Gated. | ~1.09 GB per tar [30]; **inference:** about 18 GB per 1k h | **Top conversational pick at scale.** In-the-wild talk with natural prosody. Filter DNSMOS ≥ ~3.3 and drop short or very long clips. Used by NeuTTS Air [31]. |
| Emilia (non-YODAS EN) | 46.8k h [28] | 24 kHz | same as above | CC BY-NC 4.0 [28]; gated | ~24 GB per 1k h (inference) | Alternative to YODAS, with higher DNSMOS (3.02) [18]. |
| DailyTalk | ~20 h (21.7 h in the table), 2 actors, 23,773 clips, studio, about half of dialogues with fillers [32] | not stated (44.1 kHz is unverified) | DailyDialog-derived scripts; casing unverified | CC BY-SA 4.0; Google Drive [33] | n/a | **Good runner-up** for dialog prosody. |
| EARS | 100 h / 107 [34][35] | 48 kHz anechoic | only the read portion has transcripts [34] | CC BY-NC 4.0 | ~69 GB if 32-bit (inference) | **Low.** Extremely dry audio, and the freeform and emotional parts have no transcripts. |
| NonverbalTTS | 17 h [36] | VoxCeleb 16 kHz + Expresso 48 kHz | tags for 10 nonverbal-vocalization types and 8 emotions [36] | Apache-2.0 [37] | small | **Niche.** Laughs and sighs, mostly from noisy VoxCeleb. |
| Spotify Podcasts | 50k h [38] | n/a | ASR | **No longer granting new access** [39] | n/a | Unavailable. |
| Elise | 3 h | n/a | emotion tags | **DMCA takedown** [40] | n/a | Unavailable. |

### Synthetic but open
| Dataset | Hours | SR | Transcripts | License | Size | Verdict |
|---|---|---|---|---|---|---|
| **gemini-flash-2.0-speech** | 284 h, 2 voices (Puck, Kore) [41] | 24 kHz | Wikipedia, Reddit, podcasts, support chats, numbers, news [41]. **Inference:** cased. | Apache-2.0 (Gemini ToS caveat) | 47.3 GB [41] | **Useful for text coverage** (numbers, conversational phrasing) and perfectly clean audio. The dataset author says it was used to train Kokoro and LLaSA 1B [41]. Risk: copying synthetic timbre. Kani uses it [42]. |
| EmoVoice-DB | ~40 h, GPT-4o [43] | n/a | emotion | n/a | n/a | Optional emotion add. |
| LAION's Got Talent | 5,000+ h, 11 voices, 4 languages [44] | n/a | n/a | n/a | n/a | Optional; English is only part of it. |

### What recent small open TTS models trained on
- **Kokoro-82M:** a few hundred hours of public-domain, permissively licensed and closed-model synthetic audio [45].
- **Kani-TTS 370M:** about 80k h from LibriTTS, Common Voice and Emilia, on an LFM2 350M backbone with NanoCodec [42].
- **NeuTTS Air:** Emilia plus English Emilia-YODAS, on Qwen 0.5B [31].
- **T5Gemma-TTS (Aratako):** about 100k h of English from Emilia and Libriheavy, using XCodec2 [12].
- **Orpheus:** 100k+ h of English, not released [46].
- **Dia:** data not disclosed; uses the DAC codec [47].
- **Sesame CSM:** about 1M h of "publicly available" audio, not released [48].
- **Kyutai DSM TTS:** the arXiv abstract doesn't state its data, and I found no released corpus [49].

The common thread is Emilia (and YODAS) plus LibriTTS or Libriheavy. Everything you can actually get is listed above.

### Evaluation extra
`gradium/tts-hard-sentences-202610` (CC BY 4.0, Oct 2026) is a set of hard TTS prompts such as dates and structured text. It could serve as a stress test for CER [50].

---

## Ranked shortlist

| # | Add | Hours | Download | Why |
|---|---|---|---|---|
| 1 | **LibriTTS-R train-clean-360** (cased `text_original`) | 191 h, 904 speakers | 28 GB, or the shangeth Mimi codes | Lowest-WER source in Raon's audit [18]; same pipeline as your clean-100. |
| 2 | **LibriTTS-R train-other-500**, Parler-filtered | ~300 h, ~1,100 speakers | 46 GB, or Mimi codes | Speaker diversity; restored audio. |
| 3 | **HiFiTTS-2 22 kHz slice** (`text_source=book`, CER < ~2%, bandwidth ≥ 11 kHz, single speaker, cap ~1 h per speaker) | 600–1,000 h | metadata 17.8 GB, plus ~45–75 GB of fetched audio (inference) | The biggest clean, cased, read-speech lever within your budget. |
| 4 | **Expresso full**, read plus Whisper-transcribed conversational | ~46 h, 4 speakers | 13.8 GB, or Mimi codes | Studio-quality expressive styles. Non-commercial license. |
| 5 | **Emilia-YODAS EN**, DNSMOS-filtered | 500–1,000 h | ~10–20 GB, i.e. ~10–20 tars (inference) | Conversational prosody at scale. CC BY. ASR transcripts. |

Cheap extras worth adding:
- **Jenny:** 30 h, 3.7 GB, varied cased text.
- **gemini-flash-2.0-speech:** 284 h, 47 GB, if CER on numbers and conversational text lags.
- **DailyTalk:** ~20 h, as a second dialog set.

Total for picks 1–5 is about 1.6k–2.5k h. **Inference:** at your rates that is about 14–21 h of encoding (only 4–8 h if you use the Mimi codes for picks 1, 2 and 4), and about 3–5 h per epoch of training.

## Sources
[1] https://ar5iv.labs.arxiv.org/html/1904.02882
[2] https://huggingface.co/datasets/mythicinfinity/libritts_r
[3] https://www.openslr.org/141/
[4] https://huggingface.co/datasets/parler-tts/libritts_r_filtered
[5] https://huggingface.co/datasets/nvidia/hifitts-2
[6] https://www.isca-archive.org/interspeech_2025/langman25_interspeech.html (and https://arxiv.org/abs/2506.04152)
[7] https://arxiv.org/abs/2012.03411
[8] https://huggingface.co/datasets/parler-tts/mls_eng
[9] https://www.openslr.org/94/
[10] https://huggingface.co/datasets/pkufool/libriheavy
[11] https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md
[12] https://huggingface.co/Aratako/T5Gemma-TTS-2b-2b
[13] https://huggingface.co/datasets/CSTR-Edinburgh/vctk
[14] https://huggingface.co/datasets/MushanW/GLOBE_V2
[15] https://huggingface.co/datasets/styletts2-community/common-voice-filtered
[16] https://huggingface.co/datasets/speechcolab/gigaspeech
[17] https://huggingface.co/datasets/MLCommons/peoples_speech
[18] https://huggingface.co/datasets/KRAFTON/Raon-OpenTTS-Pool/blob/main/README.md
[19] https://arxiv.org/abs/2605.20830
[20] https://huggingface.co/datasets/reach-vb/jenny_tts_dataset
[21] https://arxiv.org/abs/2106.08468
[22] https://github.com/roholazandie/ryan-tts
[23] https://huggingface.co/datasets/m-aliabbas/idrak_ryanspeech
[24] https://huggingface.co/datasets/ylacombe/expresso
[25] https://speechbot.github.io/expresso/
[26] https://huggingface.co/datasets/shangeth/expresso
[27] https://huggingface.co/datasets/somu9/expresso-conversational
[28] https://huggingface.co/datasets/amphion/Emilia-Dataset
[29] https://arxiv.org/abs/2501.15907 (and the Emilia-Pipe README mirror https://huggingface.co/spaces/Svngoku/maskgct-audio-lab/blob/main/preprocessors/Emilia/README.md)
[30] https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/main/Emilia-YODAS/EN
[31] https://huggingface.co/neuphonic/neutts-air
[32] https://ar5iv.labs.arxiv.org/html/2207.01063
[33] https://github.com/keonlee9420/DailyTalk
[34] https://github.com/facebookresearch/ears_dataset
[35] https://arxiv.org/abs/2406.06185
[36] https://arxiv.org/abs/2507.13155
[37] https://huggingface.co/datasets/deepvk/NonverbalTTS
[38] https://engineering.atspotify.com/2020/4/introducing-the-spotify-podcast-dataset-and-trec-challenge-2020
[39] https://arxiv.org/html/2411.07892v1
[40] https://huggingface.co/datasets/MrDragonFox/Elise
[41] https://huggingface.co/datasets/shb777/gemini-flash-2.0-speech
[42] https://huggingface.co/nineninesix/kani-tts-370m
[43] https://arxiv.org/pdf/2504.12867
[44] https://arxiv.org/html/2506.09827v3
[45] https://huggingface.co/hexgrad/Kokoro-82M
[46] https://github.com/canopyai/Orpheus-TTS (claims summarized via https://www.hackster.io/news/canopy-labs-releases-orpheus-a-permissively-licensed-llm-for-convincing-text-to-speech-afc64416767d)
[47] https://github.com/nari-labs/dia
[48] https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice
[49] https://arxiv.org/abs/2509.08753
[50] https://huggingface.co/datasets/gradium/tts-hard-sentences-202610
