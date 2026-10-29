"""Text front end: normalized characters (no phonemizer / G2P is installed).

Training text is LJSpeech metadata column 3 (numbers already expanded by the dataset authors).
`normalize()` lowercases, strips accents, expands common abbreviations and simple numbers
(cardinals, years, ordinals, $ amounts, %) so hand-written sentences use the same alphabet.
Punctuation , . ? ! ; : - ' " is kept as pause/prosody cues.
"""
import re
import unicodedata

PAD = "_"
SYMBOLS = [PAD] + list(" abcdefghijklmnopqrstuvwxyz,.?!;:-'\"")
SYM2ID = {s: i for i, s in enumerate(SYMBOLS)}

ABBREV = [(re.compile(r"\b%s\." % a, re.I), b) for a, b in [
    ("mrs", "missus"), ("mr", "mister"), ("dr", "doctor"), ("st", "saint"), ("co", "company"),
    ("jr", "junior"), ("sr", "senior"), ("maj", "major"), ("gen", "general"), ("drs", "doctors"),
    ("rev", "reverend"), ("lt", "lieutenant"), ("hon", "honorable"), ("sgt", "sergeant"),
    ("capt", "captain"), ("esq", "esquire"), ("ltd", "limited"), ("col", "colonel"), ("ft", "fort"),
    ("vs", "versus"), ("no", "number"), ("etc", "et cetera")]]

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
        "nine": "ninth", "twelve": "twelfth"}


def num2words(n):
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num2words(n % 100))
    for div, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= div:
            r = n % div
            return num2words(n // div) + " " + name + ("" if r == 0 else " " + num2words(r))


def year2words(n):
    n = int(n)
    if 2000 <= n < 2010:
        return num2words(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return num2words(hi) + " hundred"
    return num2words(hi) + " " + ("oh " + num2words(lo) if lo < 10 else num2words(lo))


def ordinal(n):
    w = num2words(n)
    last = re.split(r"([ -])", w)
    t = last[-1]
    if t in _ORD:
        t = _ORD[t]
    elif t.endswith("y"):
        t = t[:-1] + "ieth"
    else:
        t = t + "th"
    return "".join(last[:-1]) + t


def expand_numbers(s):
    s = re.sub(r"\d{1,3}(?:,\d{3})+", lambda m: num2words(m[0].replace(",", "")), s)  # 1,200 is not a year
    s = re.sub(r"\b(\d{1,2}):(\d\d)\b", lambda m: num2words(m[1]) + " " + ("o'clock" if m[2] == "00" else
               ("oh " + num2words(m[2]) if m[2][0] == "0" else num2words(m[2]))), s)
    s = re.sub(r"\$(\d+)\.(\d\d)\b", lambda m: f"{num2words(m[1])} dollars and {num2words(m[2])} cents", s)
    s = re.sub(r"\$(\d+)", lambda m: num2words(m[1]) + (" dollar" if m[1] == "1" else " dollars"), s)
    s = re.sub(r"(\d+)%", lambda m: num2words(m[1]) + " percent", s)
    s = re.sub(r"(\d+)(st|nd|rd|th)\b", lambda m: ordinal(m[1]), s)
    s = re.sub(r"(\d+)\.(\d+)", lambda m: num2words(m[1]) + " point " + " ".join(_ONES[int(c)] for c in m[2]), s)
    s = re.sub(r"\b(1\d\d\d|20\d\d)\b", lambda m: year2words(m[1]), s)
    s = re.sub(r"\d+", lambda m: num2words(m[0]), s)
    return s


def normalize(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    for pat, rep in ABBREV:
        s = pat.sub(rep, s)
    s = expand_numbers(s)
    s = s.lower()
    s = re.sub(r"[()\[\]]", " , ", s)
    s = s.replace("&", " and ")
    s = "".join(c if c in SYM2ID else " " for c in s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,.?!;:])", r"\1", s)
    s = re.sub(r",(\s*,)+", ",", s)
    s = re.sub(r",\s*([.?!;:])", r"\1", s)
    s = re.sub(r"([,.?!;:])(?=[a-z])", r"\1 ", s)
    s = s.strip(" ,")
    return s


def encode(s, pad_edges=True):
    """normalized text -> list of ids, with a leading and trailing space to absorb edge silence."""
    if pad_edges:
        s = " " + s + " "
    return [SYM2ID[c] for c in s]


def decode(ids):
    return "".join(SYMBOLS[i] for i in ids if i != 0)


if __name__ == "__main__":
    for t in ["Dr. Smith paid $15.50 on March 3rd, 1963.", "Mr. Neild's “State of Prisons” (1812).",
              "It rose 45% to 1,234,567 in 2005.", "Lee Harvey Oswald, née Café."]:
        print(t, "->", normalize(t))
