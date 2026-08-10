"""
Helpers.py Nazalyst
"""

from collections import Counter
import re

import pandas as pd


# ==========================================================
# FORMAT ANGKA
# ==========================================================

def format_number(value):

    try:
        return f"{int(value):,}"
    except Exception:
        return "0"


# ==========================================================
# FORMAT PERSEN
# ==========================================================

def format_percent(value):

    try:
        return f"{float(value):.2f}%"
    except Exception:
        return "0.00%"


# ==========================================================
# RINGKASAN SENTIMEN
# ==========================================================

def sentiment_distribution(df):

    total = len(df)

    if total == 0:

        return {

            "total": 0,

            "positif": 0,

            "negatif": 0,

            "netral": 0,

            "positif_percent": 0,

            "negatif_percent": 0,

            "netral_percent": 0

        }

    counts = df["pred_label"].value_counts()

    positif = counts.get("positif", 0)

    negatif = counts.get("negatif", 0)

    netral = counts.get("netral", 0)

    return {

        "total": total,

        "positif": positif,

        "negatif": negatif,

        "netral": netral,

        "positif_percent": positif / total * 100,

        "negatif_percent": negatif / total * 100,

        "netral_percent": netral / total * 100

    }


# ==========================================================
# FILTER SENTIMEN
# ==========================================================

def filter_sentiment(df, labels):

    if not labels:

        return df.copy()

    return df[df["pred_label"].isin(labels)].copy()


# ==========================================================
# FILTER KEYWORD
# ==========================================================

def filter_keyword(df, keyword):

    if keyword is None or keyword.strip() == "":

        return df.copy()

    return df[
        df["text"]
        .astype(str)
        .str.contains(
            keyword,
            case=False,
            na=False
        )
    ].copy()


# ==========================================================
# MEMBERSIHKAN TEKS
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# TOP WORD
# ==========================================================

def top_words(df, n=20):

    if "text" not in df.columns:

        return pd.Series(dtype=int)

    words = []

    for sentence in df["text"].astype(str):

        sentence = clean_text(sentence)

        words.extend(sentence.split())

    counter = Counter(words)

    data = pd.DataFrame(

        counter.items(),

        columns=[

            "Kata",

            "Frekuensi"

        ]

    )

    data = data.sort_values(

        by="Frekuensi",

        ascending=False

    )

    return data.head(n)


# ==========================================================
# FILTER RATING
# ==========================================================

def filter_rating(df, ratings):

    if "score" not in df.columns:

        return df.copy()

    if not ratings:

        return df.copy()

    return df[
        df["score"].isin(ratings)
    ].copy()


# ==========================================================
# TOTAL RATING
# ==========================================================

def rating_summary(df):

    if "score" not in df.columns:

        return {}

    return (

        df["score"]

        .value_counts()

        .sort_index()

        .to_dict()

    )


# ==========================================================
# DATASET INFO
# ==========================================================

def dataset_info(df):

    info = {

        "rows": len(df),

        "columns": len(df.columns),

        "missing": int(df.isna().sum().sum()),

        "duplicates": int(df.duplicated().sum())

    }

    return info