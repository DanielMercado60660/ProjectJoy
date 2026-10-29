"""
BPE tokenizer for nanochat-mlx.

Port of upstream nanochat/tokenizer.py. Upstream trains its own BPE with `rustbpe` and runs
inference with `tiktoken`. Neither rustbpe nor a way to build a tiktoken encoding from scratch
cheaply is the point here, so we train a byte-level BPE with the installed HuggingFace
`tokenizers` library, using upstream's GPT-4-style split regex and upstream's special tokens.

Usage:
    python tokenizer.py --train --vocab-size 8192 --max-chars 200000000
    python tokenizer.py --eval
"""
import argparse
import json
import os
import time

from tokenizers import Tokenizer, Regex, decoders, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_DIR = os.path.join(HERE, "checkpoints", "tokenizer")

# same special tokens as upstream (bos doubles as the document delimiter)
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]

# upstream's split pattern (GPT-4 style, but numbers grouped 1-2 digits instead of 1-3)
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


def _bytes_to_unicode():
    """GPT-2 / HF ByteLevel byte -> printable-char map."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class ChatTokenizer:
    """Thin wrapper with the upstream interface (encode / decode / render_conversation)."""

    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.special = {s: tok.token_to_id(s) for s in SPECIAL_TOKENS}
        self.bos = self.special["<|bos|>"]

    @classmethod
    def load(cls, directory=TOKENIZER_DIR):
        return cls(Tokenizer.from_file(os.path.join(directory, "tokenizer.json")))

    def get_vocab_size(self):
        return self.tok.get_vocab_size()

    def encode_special(self, s):
        return self.special[s]

    def encode(self, text, prepend_bos=False):
        ids = self.tok.encode(text, add_special_tokens=False).ids
        return [self.bos] + ids if prepend_bos else ids

    def encode_batch(self, texts, prepend_bos=True):
        out = self.tok.encode_batch(texts, add_special_tokens=False)
        return [([self.bos] + e.ids) if prepend_bos else e.ids for e in out]

    def decode(self, ids):
        return self.tok.decode(list(ids), skip_special_tokens=False)

    def token_bytes(self):
        """Number of UTF-8 bytes per token id (0 for special tokens), used for bits-per-byte."""
        # Count the token's raw bytes via the ByteLevel alphabet (1 char = 1 byte). Decoding a lone
        # token and re-encoding over-counts tokens that are partial UTF-8 sequences: they decode to
        # U+FFFD (3 bytes) instead of 1 byte. (Review fix; the old way overstated val bytes by 0.12%.)
        u2b = {c: b for b, c in _bytes_to_unicode().items()}
        n = self.get_vocab_size()
        out = [0] * n
        specials = set(self.special.values())
        for i in range(n):
            if i in specials:
                continue
            out[i] = len(self.tok.id_to_token(i))
            assert all(c in u2b for c in self.tok.id_to_token(i))
        return out

    # ---- chat rendering, same layout as upstream render_conversation ----
    def render_conversation(self, messages, max_tokens=2048):
        """messages: list of {"role": "user"|"assistant", "content": str}.
        Returns (ids, mask) where mask=1 marks tokens the assistant is trained to produce."""
        ids, mask = [], []

        def add(tok_ids, m):
            if isinstance(tok_ids, int):
                tok_ids = [tok_ids]
            ids.extend(tok_ids)
            mask.extend([m] * len(tok_ids))

        add(self.bos, 0)
        for i, msg in enumerate(messages):
            assert msg["role"] == ("user" if i % 2 == 0 else "assistant")
            if msg["role"] == "user":
                add(self.special["<|user_start|>"], 0)
                add(self.encode(msg["content"]), 0)
                add(self.special["<|user_end|>"], 0)
            else:
                add(self.special["<|assistant_start|>"], 0)
                add(self.encode(msg["content"]), 1)
                add(self.special["<|assistant_end|>"], 1)
        return ids[:max_tokens], mask[:max_tokens]

    def render_for_completion(self, messages):
        """Prime the assistant for a reply (messages ends with a user message)."""
        ids, _ = self.render_conversation(messages, max_tokens=10**9)
        return ids + [self.special["<|assistant_start|>"]]


def build_tokenizer():
    tok = Tokenizer(models.BPE(byte_fallback=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERN), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def text_iterator(max_chars, doc_cap=10000):
    """Iterate training documents from the train shards (never the val shard)."""
    from data import list_shards, iter_docs
    n = 0
    for path in list_shards("train"):
        for doc in iter_docs(path):
            doc = doc[:doc_cap]
            n += len(doc)
            yield doc
            if n >= max_chars:
                return


def train(vocab_size, max_chars):
    t0 = time.time()
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size - len(SPECIAL_TOKENS),
        special_tokens=[],  # added after, at the end of the vocab like upstream
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
        min_frequency=2,
    )
    tok.train_from_iterator(text_iterator(max_chars), trainer=trainer)
    # upstream: specials sit at the end of the vocab; vocab_size includes them
    tok.add_special_tokens(SPECIAL_TOKENS)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    tok.save(os.path.join(TOKENIZER_DIR, "tokenizer.json"))
    ct = ChatTokenizer(tok)
    with open(os.path.join(TOKENIZER_DIR, "token_bytes.json"), "w") as f:
        json.dump(ct.token_bytes(), f)
    print(f"trained BPE vocab={tok.get_vocab_size()} on {max_chars/1e6:.0f}M chars in {time.time()-t0:.1f}s")
    return ct


def evaluate(ct: ChatTokenizer, n_docs=2000):
    """Compression ratio (bytes/token) on held-out val docs, vs tiktoken gpt2 / cl100k if available."""
    from data import list_shards, iter_docs
    docs = []
    for d in iter_docs(list_shards("val")[0]):
        docs.append(d)
        if len(docs) >= n_docs:
            break
    nbytes = sum(len(d.encode("utf-8")) for d in docs)
    res = {"ours": nbytes / sum(len(x) for x in ct.encode_batch(docs, prepend_bos=False))}
    try:
        import tiktoken
        for name in ["gpt2", "cl100k_base"]:
            enc = tiktoken.get_encoding(name)
            res[name] = nbytes / sum(len(x) for x in enc.encode_ordinary_batch(docs))
    except Exception as e:  # tiktoken may need to download its BPE files
        res["tiktoken_error"] = str(e)[:100]
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--max-chars", type=int, default=200_000_000)
    args = ap.parse_args()
    if args.train:
        ct = train(args.vocab_size, args.max_chars)
    else:
        ct = ChatTokenizer.load()
    if args.eval or args.train:
        print("val bytes/token:", json.dumps(evaluate(ct)))
        s = "Hello world! The year 2026 has 12 months. <|bos|> move north"
        ids = ct.encode(s)
        print(len(ids), ids[:20], repr(ct.decode(ids)))
