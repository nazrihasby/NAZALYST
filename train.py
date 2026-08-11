"""
=========================================================
NAZALYST
Train Data
=========================================================
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from preprocess import batch_clean


# ──────────────────────────────────────────────────────────────────────────────
# Utilitas: baca & gabung kamus slang (berbagai skema kolom)
# ──────────────────────────────────────────────────────────────────────────────
def _read_csv_any(path: Path) -> pd.DataFrame:
    """Baca CSV dengan beberapa fallback encoding."""
    if not path or not Path(path).exists():
        return pd.DataFrame()
    for enc in ("utf-8", "utf-8-sig", "latin1", "ISO-8859-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise RuntimeError(f"Gagal membaca file: {path}")


def _standardize_kamus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standarkan kolom menjadi [slang, normal].
    Mendukung skema umum: (slang,formal) / (alay,baku) / (slangword,formalword)
    atau fallback 2 kolom string pertama.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["slang", "normal"])

    lower = {c.lower(): c for c in df.columns}
    if {"slang", "formal"}.issubset(lower):
        df = df[[lower["slang"], lower["formal"]]].copy()
    elif {"alay", "baku"}.issubset(lower):
        df = df[[lower["alay"], lower["baku"]]].copy()
    elif {"slangword", "formalword"}.issubset(lower):
        df = df[[lower["slangword"], lower["formalword"]]].copy()
    else:
        # ambil 2 kolom bertipe string / fallback 2 kolom pertama
        str_cols = [c for c in df.columns if df[c].dtype == "object"]
        if len(str_cols) < 2:
            str_cols = list(df.columns)[:2]
        df = df[str_cols[:2]].copy()

    df.columns = ["slang", "normal"]
    for c in ("slang", "normal"):
        df[c] = df[c].astype(str).str.strip().str.lower()
    df = df[(df["slang"] != "") & (df["normal"] != "")]
    df = df[df["slang"] != df["normal"]]
    return df


def merge_slang_sources(
    sources: list[str],
    out_path: Path,
    indo_words_path: Path | None = None,
) -> pd.DataFrame:
    """
    Gabungkan beberapa kamus slang → satu CSV.
    Prioritas: urutan sumber di 'sources' (awal = paling diutamakan).
    Tie-break: normal ∈ indo_words (jika ada), lalu normal terpendek.
    """
    frames = []
    for rank, src in enumerate(sources):
        src = src.strip()
        if not src:
            continue
        df = _read_csv_any(Path(src))
        df = _standardize_kamus(df)
        if not df.empty:
            df["__src_rank"] = rank  # simpan urutan sumber
            frames.append(df)

    if not frames:
        print("[merge] Tidak ada sumber kamus yang valid. Lewati penggabungan.")
        return pd.DataFrame(columns=["slang", "normal"])

    merged = pd.concat(frames, ignore_index=True)

    # Muat daftar kata baku (opsional)
    dict_words = set()
    if indo_words_path and Path(indo_words_path).exists():
        try:
            with open(indo_words_path, "r", encoding="utf-8") as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        dict_words.add(w)
        except Exception:
            pass

    tmp = merged.copy()
    tmp["_len"] = tmp["normal"].str.len()  # kolom bantu utk sort

    if dict_words:
        tmp["_is_dict"] = tmp["normal"].isin(dict_words)
        # urutkan per slang: prioritas sumber → kata baku → yang terpendek
        tmp = tmp.sort_values(
            by=["slang", "__src_rank", "_is_dict", "_len"],
            ascending=[True, True, False, True],
        ).drop(columns=["_is_dict"])
    else:
        # tanpa kamus baku: prioritas sumber → yang terpendek
        tmp = tmp.sort_values(
            by=["slang", "__src_rank", "_len"],
            ascending=[True, True, True],
        )

    # ambil satu mapping per slang (yang pertama sesuai prioritas di atas)
    tmp = tmp.drop_duplicates(subset=["slang"], keep="first")
    # bersihkan kolom bantu
    tmp = tmp.drop(columns=["__src_rank", "_len"], errors="ignore")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[merge] Kamus slang tersimpan: {out_path} (baris={len(tmp)})")
    return tmp


# ──────────────────────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train SVM (Linear) untuk Analisis Sentimen Bahasa Indonesia"
    )
    ap.add_argument("--csv", required=True, help="Path dataset CSV (wajib kolom text,label)")
    ap.add_argument("--text_col", default="text")
    ap.add_argument("--label_col", default="label")

    # Preprocess flags
    ap.add_argument("--stemming", action="store_true", help="Aktifkan stemming Sastrawi")
    ap.add_argument("--remove_stopwords", action="store_true", help="Hapus stopwords NLTK")

    # Featurization
    ap.add_argument("--ngram_max", type=int, default=3)
    ap.add_argument("--max_features", type=int, default=100_000)

    # Model
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--class_weight", default="balanced", help="balanced atau none")
    ap.add_argument("--seed", type=int, default=42)

    # I/O
    ap.add_argument("--output_dir", default="models")
    ap.add_argument("--report_path", default="reports/evaluation.txt")

    # Slang merging (opsional)
    ap.add_argument("--merge_slang", action="store_true", help="Merge kamus slang sebelum training")
    ap.add_argument(
        "--slang_sources",
        default="data/slang_dict.csv,data/kamusalay.csv",
        help="Daftar file kamus (dipisah koma), urut = prioritas",
    )
    ap.add_argument(
        "--indo_words",
        default="data/indonesian-words.txt",
        help="Daftar kata baku (opsional) untuk tie-breaking",
    )

    args = ap.parse_args()

    # Class weight
    class_weight = None if str(args.class_weight).lower() in {"none", "null", "", "-"} else "balanced"

    # (Opsional) gabungkan kamus slang → data/slang_dict.csv
    if args.merge_slang:
        srcs = [s.strip() for s in str(args.slang_sources).split(",") if s.strip()]
        merge_slang_sources(srcs, Path("data/slang_dict.csv"), Path(args.indo_words))

    # Baca dataset
    df = pd.read_csv(args.csv)
    if not {args.text_col, args.label_col}.issubset(df.columns):
        raise ValueError(f"CSV harus mengandung kolom: {args.text_col},{args.label_col}")

    texts = df[args.text_col].astype(str).tolist()
    labels = df[args.label_col].astype(str).tolist()

    # Praproses
    print(">> Praproses teks…")
    X_clean = batch_clean(
        texts,
        stemming=args.stemming,
        remove_stopwords=args.remove_stopwords,
    )

    # Encode label
    le = LabelEncoder()
    y = le.fit_transform(labels)
    print(f">> Kelas: {list(le.classes_)}")

    # Split data
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    # TF-IDF
    vect = TfidfVectorizer(
        ngram_range=(1, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
    )
    Xtr = vect.fit_transform(X_tr)
    Xte = vect.transform(X_te)

    # Model: Linear SVC + kalibrasi (sigmoid) agar ada predict_proba
    base = LinearSVC(C=args.C, class_weight=class_weight, random_state=args.seed)
    clf = CalibratedClassifierCV(
        estimator=base,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed),
        method="sigmoid",
    )

    print(">> Latih model…")
    clf.fit(Xtr, y_tr)

    # Evaluasi
    print(">> Evaluasi…")

    y_pred = clf.predict(Xte)

    # Accuracy
    acc = accuracy_score(y_te, y_pred)

    # Weighted Metrics
    precision = precision_score(
        y_te,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_te,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_te,
        y_pred,
        average="weighted"
    )

    # Classification Report
    report_str = classification_report(
        y_te,
        y_pred,
        target_names=le.classes_
    )

    # Confusion Matrix
    cm = confusion_matrix(y_te, y_pred)

    # Tulis laporan
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Evaluation\n")
        f.write(json.dumps({
            "accuracy": acc,
            "weighted_precision": precision,
            "weighted_recall": recall,
            "weighted_f1": f1
        }))
        f.write("\n\n## Classification Report\n")
        f.write(report_str)
        f.write("\n\n## Confusion Matrix\n")
        f.write(np.array2string(cm))
    print(f">> Accuracy           : {acc:.4f}")
    print(f">> Weighted Precision : {precision:.4f}")
    print(f">> Weighted Recall    : {recall:.4f}")
    print(f">> Weighted F1-score  : {f1:.4f}")
    print(f">> Laporan: {report_path}")

    # Simpan artefak
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump(vect, out_dir / "vectorizer.pkl")
    dump(clf, out_dir / "model.pkl")
    dump(le, out_dir / "label_encoder.pkl")
    print(f">> Artefak tersimpan di: {out_dir}")


if __name__ == "__main__":
    main()
