"""
Config.py Nazalyst
"""

from pathlib import Path
import pandas as pd
import streamlit as st
import joblib
import json

from services.updater import DatasetUpdater
from services.metadata import MetadataService
from services.trainer import ModelTrainer
from services.predictor import SentimentPredictor

# ==========================================================
# ROOT PROJECT
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

STYLE_DIR = ROOT_DIR / "styles"


# ==========================================================
# PAGE CONFIG
# ==========================================================

def setup_page():

    st.set_page_config(

        page_title="NAZALYST",

        page_icon="📊",

        layout="wide",

        initial_sidebar_state="expanded"

    )


# ==========================================================
# CSS
# ==========================================================

def load_css():

    css_path = STYLE_DIR / "style.css"

    if css_path.exists():

        with open(css_path, encoding="utf-8") as f:

            st.markdown(

                f"<style>{f.read()}</style>",

                unsafe_allow_html=True

            )


# ==========================================================
# COLOR MAP
# ==========================================================

COLOR_MAP = {

    "positif": "#16A34A",

    "negative": "#DC2626",

    "negatif": "#DC2626",

    "neutral": "#0EA5E9",

    "netral": "#0EA5E9"

}


# ==========================================================
# LOAD DATASET
# ==========================================================

@st.cache_data(show_spinner=False)
def load_dataset():

    dataset_path = DATA_DIR / "hasil_prediksi.csv"

    if not dataset_path.exists():

        st.error(
            f"Dataset tidak ditemukan:\n{dataset_path}"
        )

        st.stop()

    df = pd.read_csv(dataset_path)

    # -------------------------------------
    # Validasi kolom
    # -------------------------------------

    required = [

        "text",

        "pred_label"

    ]

    for col in required:

        if col not in df.columns:

            st.error(

                f"Kolom '{col}' tidak ditemukan."

            )

            st.stop()

    # -------------------------------------
    # Parsing tanggal
    # -------------------------------------

    if "created_at" in df.columns:

        df["created_at"] = pd.to_datetime(

            df["created_at"],

            errors="coerce"

        )

        df["year"] = df["created_at"].dt.year

        df["month"] = df["created_at"].dt.strftime("%b")

    # -------------------------------------
    # Score
    # -------------------------------------

    if "score" in df.columns:

        df["score"] = pd.to_numeric(

            df["score"],

            errors="coerce"

        )

    return df


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_model():

    model = joblib.load(

        MODEL_DIR / "model.pkl"

    )

    vectorizer = joblib.load(

        MODEL_DIR / "vectorizer.pkl"

    )

    encoder = joblib.load(

        MODEL_DIR / "label_encoder.pkl"

    )

    return model, vectorizer, encoder

# ==========================================================
# LOAD METADATA
# ==========================================================

@st.cache_data(show_spinner=False)
def load_metadata():
    """
    Membaca metadata.json.
    """

    metadata_path = DATA_DIR / "metadata.json"

    if not metadata_path.exists():

        return {

            "total_reviews": 0,

            "positive_reviews": 0,

            "neutral_reviews": 0,

            "negative_reviews": 0,

            "average_rating": 0,

            "latest_review": None,

            "last_updated": None,

        }

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# LOAD TRAINING INFO
# ==========================================================

@st.cache_data(show_spinner=False)
def load_training_info():
    """
    Membaca training_info.json.
    """

    info_path = DATA_DIR / "training_info.json"

    if not info_path.exists():

        return {

            "status": "never_trained",

            "last_training": None,

            "duration": 0,

            "dataset_size": 0,

        }

    with open(
        info_path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)
    
    # ==========================================================
# DATASET UPDATER
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_dataset_updater():
    """
    Mengembalikan instance DatasetUpdater.
    """

    return DatasetUpdater()


# ==========================================================
# METADATA SERVICE
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_metadata_service():
    """
    Mengembalikan instance MetadataService.
    """

    return MetadataService()


# ==========================================================
# MODEL TRAINER
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_model_trainer():
    """
    Mengembalikan instance ModelTrainer.
    """

    return ModelTrainer()


# ==========================================================
# SENTIMENT PREDICTOR
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_sentiment_predictor():
    """
    Mengembalikan instance SentimentPredictor.
    """

    return SentimentPredictor()


# ==========================================================
# SERVICE STATUS
# ==========================================================

@st.cache_data(show_spinner=False)
def get_service_status():
    """
    Mengecek status seluruh service dan file utama.
    """

    return {

        "dataset_exists": (DATA_DIR / "dataset.csv").exists(),

        "prediction_dataset_exists": (
            DATA_DIR / "hasil_prediksi.csv"
        ).exists(),

        "metadata_exists": (
            DATA_DIR / "metadata.json"
        ).exists(),

        "training_info_exists": (
            DATA_DIR / "training_info.json"
        ).exists(),

        "model_exists": (
            MODEL_DIR / "model.pkl"
        ).exists(),

        "vectorizer_exists": (
            MODEL_DIR / "vectorizer.pkl"
        ).exists(),

        "label_encoder_exists": (
            MODEL_DIR / "label_encoder.pkl"
        ).exists(),

    }

    # ==========================================================
# CACHE
# ==========================================================

def refresh_cache():
    """
    Membersihkan seluruh cache Streamlit.
    Dipanggil setelah update dataset atau retraining model.
    """

    st.cache_data.clear()

    st.cache_resource.clear()


# ==========================================================
# RELOAD
# ==========================================================

def reload_dataset():
    """
    Refresh dataset kemudian memuat ulang.
    """

    st.cache_data.clear()

    return load_dataset()


def reload_metadata():
    """
    Refresh metadata kemudian memuat ulang.
    """

    st.cache_data.clear()

    return load_metadata()


def reload_training_info():
    """
    Refresh training_info kemudian memuat ulang.
    """

    st.cache_data.clear()

    return load_training_info()


# ==========================================================
# SYSTEM HEALTH
# ==========================================================

def system_health():
    """
    Menghasilkan status kesehatan sistem.
    """

    status = get_service_status()

    healthy = all(status.values())

    return {

        "healthy": healthy,

        "items": status

    }


# ==========================================================
# VERSION
# ==========================================================

APP_NAME = "NAZALYST"

APP_VERSION = "2.0"

APP_AUTHOR = "Naz Hasby"

MODEL_NAME = "Support Vector Machine"

FEATURE_EXTRACTION = "TF-IDF"

DATA_SOURCE = "Google Play Store (MyPertamina)"