"""
=========================================================
NAZALYST v2.0
Prediction Service
=========================================================
"""

from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np

from preprocess import clean_text


logger = logging.getLogger(__name__)


# ==========================================================
# PATH
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = ROOT_DIR / "models"

MODEL_PATH = MODEL_DIR / "model.pkl"

VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"


# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class PredictionResult:

    text: str

    prediction: str

    confidence: float

    probabilities: dict

# ==========================================================
# PREDICTOR
# ==========================================================

class SentimentPredictor:

    """
    Sentiment Prediction Service.
    """

    def __init__(self):

        logger.info("Loading model...")

        self.model = joblib.load(
            MODEL_PATH
        )

        self.vectorizer = joblib.load(
            VECTORIZER_PATH
        )

        self.label_encoder = joblib.load(
            LABEL_ENCODER_PATH
        )

        logger.info(
            "Model berhasil dimuat."
        )

    # ======================================================
    # PREPROCESS
    # ======================================================

    @staticmethod
    def preprocess(
        text: str,
    ) -> str:

        return clean_text(text)

    # ======================================================
    # TRANSFORM
    # ======================================================

    def transform(
        self,
        text: str,
    ):

        cleaned = self.preprocess(
            text
        )

        return self.vectorizer.transform(
            [cleaned]
        )
        # ======================================================
    # PREDICT
    # ======================================================

    def predict(
        self,
        text: str,
    ) -> PredictionResult:
        """
        Melakukan prediksi sentimen.

        Parameters
        ----------
        text : str
            Teks yang akan diprediksi.

        Returns
        -------
        PredictionResult
        """

        if not text.strip():

            raise ValueError(
                "Teks tidak boleh kosong."
            )

        features = self.transform(
            text
        )

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction_index = self.model.predict(
            features
        )[0]

        prediction = self.label_encoder.inverse_transform(
            [prediction_index]
        )[0]

        # ------------------------------------------
        # Probability
        # ------------------------------------------

        probabilities = self.model.predict_proba(
            features
        )[0]

        labels = self.label_encoder.classes_

        probability_dict = {

            label: round(float(prob) * 100, 2)

            for label, prob in zip(
                labels,
                probabilities,
            )

        }

        confidence = max(
            probability_dict.values()
        )

        return PredictionResult(

            text=text,

            prediction=prediction,

            confidence=confidence,

            probabilities=probability_dict,

        )

    # ======================================================
    # PREDICT BATCH
    # ======================================================

    def predict_batch(
        self,
        texts: List[str],
    ) -> List[PredictionResult]:
        """
        Melakukan prediksi beberapa teks sekaligus.
        """

        results = []

        for text in texts:

            results.append(

                self.predict(text)

            )

        return results
        # ======================================================
    # MODEL INFORMATION
    # ======================================================

    @staticmethod
    def model_information() -> dict:
        """
        Mengembalikan informasi ketersediaan model.
        """

        return {

            "model_exists": MODEL_PATH.exists(),

            "vectorizer_exists": VECTORIZER_PATH.exists(),

            "label_encoder_exists": LABEL_ENCODER_PATH.exists(),

        }

    # ======================================================
    # MODEL READY
    # ======================================================

    @staticmethod
    def is_model_ready() -> bool:
        """
        Mengecek apakah seluruh file model tersedia.
        """

        return (

            MODEL_PATH.exists()

            and VECTORIZER_PATH.exists()

            and LABEL_ENCODER_PATH.exists()

        )

    # ======================================================
    # SUMMARY
    # ======================================================

    def summary(self) -> dict:
        """
        Ringkasan model prediction.
        """

        info = self.model_information()

        info["classes"] = list(
            self.label_encoder.classes_
        )

        info["total_classes"] = len(
            self.label_encoder.classes_
        )

        return info
    # ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    predictor = SentimentPredictor()

    text = input(
        "\nMasukkan kalimat : "
    )

    result = predictor.predict(
        text
    )

    print()

    print("=" * 60)

    print("PREDICTION RESULT")

    print("=" * 60)

    print(f"Text        : {result.text}")

    print(f"Prediction  : {result.prediction}")

    print(f"Confidence  : {result.confidence:.2f}%")

    print()

    print("Probability")

    print("-" * 60)

    for label, value in result.probabilities.items():

        print(f"{label:<12}: {value:.2f}%")

    print("=" * 60)

    print()

    print("MODEL INFORMATION")

    print("-" * 60)

    summary = predictor.summary()

    for key, value in summary.items():

        print(f"{key:<20}: {value}")

    print("=" * 60)