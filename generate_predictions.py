"""
=========================================================
NAZALYST
Generate Prediction For Visualization
=========================================================
"""
import pandas as pd
import numpy as np
import joblib
from preprocess import batch_clean

# =========================
# LOAD DATASET
# =========================
print("📥 Loading dataset...")
df = pd.read_csv("data/dataset.csv")

# =========================
# VALIDASI KOLOM
# =========================
if "text" not in df.columns:
    raise ValueError("❌ Dataset harus memiliki kolom 'text'")

print(f"✅ Jumlah data: {len(df)}")

# =========================
# LOAD MODEL
# =========================
print("📦 Loading model...")
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

# =========================
# PREPROCESS
# =========================
print("🔄 Cleaning text...")
cleaned_text = batch_clean(
    df["text"].astype(str).tolist(),
    stemming=False,
    remove_stopwords=True
)

# =========================
# TRANSFORM & PREDICT
# =========================
print("🤖 Predicting...")
X = vectorizer.transform(cleaned_text)
pred = model.predict(X)

pred_label = label_encoder.inverse_transform(pred)
df["pred_label"] = pred_label

print("📊 Distribusi prediksi:")
print(df["pred_label"].value_counts())

# =========================
# HANDLE KOLOM TANGGAL → YEAR
# =========================
print("\n📅 Processing tanggal...")

date_col = None

# kemungkinan nama kolom tanggal
possible_cols = ["date", "created_at", "timestamp", "review_date", "datetime"]

for col in possible_cols:
    if col in df.columns:
        date_col = col
        break

if date_col:
    print(f"✅ Kolom tanggal ditemukan: {date_col}")

    # convert ke datetime
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # buat year
    df["year"] = df[date_col].dt.year

    # cek hasil
    print("\n📊 Distribusi tahun:")
    print(df["year"].value_counts(dropna=False))

    # cek NaN
    missing_year = df["year"].isna().sum()
    print(f"\n⚠️ Year kosong: {missing_year} baris")

    if missing_year > 0:
        print("⚠️ Beberapa tanggal gagal diparse")

else:
    print("⚠️ Tidak ditemukan kolom tanggal!")

    # fallback (biar dashboard tetap jalan)
    print("⚠️ Membuat tahun default (2023)...")
    df["year"] = 2023

# =========================
# VALIDASI AKHIR
# =========================
print("\n🔍 Validasi akhir...")

if df["year"].isna().all():
    print("❌ Semua year kosong! Periksa kolom tanggal.")
else:
    print("✅ Kolom year siap digunakan")

# =========================
# SIMPAN HASIL
# =========================
output_path = "data/hasil_prediksi.csv"
df.to_csv(output_path, index=False)

print(f"\n✅ Selesai! File tersimpan di: {output_path}")