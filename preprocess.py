"""
=========================================================
NAZALYST
Preprocess
=========================================================
"""

import re
from typing import List, Iterable

# =========================
# STEMMER (TETAP DIPAKAI)
# =========================
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    _STEMMER = StemmerFactory().create_stemmer()
except Exception:
    _STEMMER = None

# =========================
# STOPWORDS NLTK
# =========================
try:
    from nltk.corpus import stopwords
    _ID_STOP = set(stopwords.words("indonesian"))
except:
    _ID_STOP = {
        "yang","dan","di","ke","dari","dengan","atau","untuk",
        "ini","itu","saya","kamu","dia","kami","kita","pada","ada"
    }

# 🔥 tambahan stopwords (UNTUK WORDCLOUD)
EXTRA_STOPWORDS = {
    "aja","nih","dong","deh","kan","lah","pun",
    "yg","nya","nyaa","nyaaa","sih","kok",
    "banget","sangat","juga","sudah","belum","lagi"
}

# =========================
# SLANG CSV (WAJIB DIPAKAI)
# =========================
_SLANG_MAP = {}
try:
    import pandas as pd
    df_slang = pd.read_csv("data/slang_dict.csv")
    _SLANG_MAP = {
        str(k).lower(): str(v).lower()
        for k, v in zip(df_slang["slang"], df_slang["normal"])
    }
except:
    _SLANG_MAP = {}

# tambahan slang kecil (optional)
_EXTRA_SLANG = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "bgt": "banget",
    "bgtt": "banget",
    "mulu": "terus",
    "lemot": "lambat",
}
_SLANG_MAP.update(_EXTRA_SLANG)

# =========================
# REGEX
# =========================
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
NON_ALNUM = re.compile(r"[^a-z0-9\s_]")

# =========================
# NEGASI (WAJIB)
# =========================
NEGATIONS = {"tidak","tak","bukan","nggak","gak","ga","enggak","kurang","tanpa"}

# =========================
# CLEAN DASAR
# =========================
def basic_clean(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = HASHTAG_PATTERN.sub(lambda m: m.group(1), text)
    text = URL_PATTERN.sub(" ", text)
    text = MENTION_PATTERN.sub(" ", text)

    text = text.lower()
    text = NON_ALNUM.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
# NORMALISASI SLANG
# =========================
def normalize_slang(tokens: Iterable[str]) -> List[str]:
    return [_SLANG_MAP.get(t, t) for t in tokens]

# =========================
# NEGATION JOIN
# =========================
def join_negations(tokens: List[str]) -> List[str]:
    result = []
    i = 0
    while i < len(tokens):
        if tokens[i] in NEGATIONS and i + 1 < len(tokens):
            result.append(tokens[i] + "_" + tokens[i+1])
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result

# =========================
# STEM TOKEN
# =========================
def stem_token(token: str) -> str:
    if _STEMMER is None:
        return token
    if "_" in token:
        a, b = token.split("_", 1)
        return f"{_STEMMER.stem(a)}_{_STEMMER.stem(b)}"
    return _STEMMER.stem(token)

# =========================
# 🔹 CLEAN UNTUK MODEL
# =========================
def clean_text(text: str,
               remove_stopwords=True,
               stemming=False,
               preserve_negations=True) -> str:

    text = basic_clean(text)
    tokens = text.split()

    tokens = normalize_slang(tokens)

    if remove_stopwords:
        tokens = [
            t for t in tokens
            if (t not in _ID_STOP) or (t in NEGATIONS)
        ]

    if preserve_negations:
        tokens = join_negations(tokens)

    if stemming:
        tokens = [stem_token(t) for t in tokens]

    return " ".join(tokens)

# =========================
# 🔥 CLEAN KHUSUS WORDCLOUD
# =========================
def clean_for_wordcloud(text: str) -> str:
    text = basic_clean(text)
    tokens = text.split()

    tokens = normalize_slang(tokens)

    tokens = [
        t for t in tokens
        if t not in _ID_STOP
        and t not in EXTRA_STOPWORDS
        and len(t) > 3
    ]

    return " ".join(tokens)

# =========================
# BATCH
# =========================
def batch_clean(texts: List[str], **kwargs) -> List[str]:
    return [clean_text(t, **kwargs) for t in texts]

def batch_clean_wordcloud(texts: List[str]) -> List[str]:
    return [clean_for_wordcloud(t) for t in texts]