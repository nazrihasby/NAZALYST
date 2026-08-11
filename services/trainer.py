"""
=========================================================
NAZALYST
Training Service
=========================================================
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import pandas as pd

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Optional


logger = logging.getLogger(__name__)


# ==========================================================
# PATH
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

MODEL_DIR = ROOT_DIR / "models"

TRAIN_SCRIPT = ROOT_DIR / "train.py"

PREDICTION_SCRIPT = ROOT_DIR / "generate_predictions.py"

TRAINING_INFO_PATH = DATA_DIR / "training_info.json"

DATASET_PATH = DATA_DIR / "dataset.csv"


# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class TrainingResult:

    status: str

    started_at: datetime

    finished_at: datetime

    duration: float

    message: str
# ==========================================================
# TRAINER
# ==========================================================

class ModelTrainer:

    """
    Service untuk menjalankan training model.
    """

    def __init__(

        self,

        train_script: Path = TRAIN_SCRIPT,

        dataset_path: Path = DATASET_PATH,

    ):

        self.train_script = train_script

        self.dataset_path = dataset_path

    # ======================================================
    # CALLBACK
    # ======================================================

    @staticmethod
    def _emit_progress(

        callback: Optional[Callable],

        stage: str,

        progress: int,

    ):

        if callback is None:
            return

        callback(

            {

                "stage": stage,

                "progress": progress,

            }

        )

    # ======================================================
    # SAVE TRAINING INFO
    # ======================================================

    @staticmethod
    def save_training_info(

        info: dict,

    ):

        DATA_DIR.mkdir(

            parents=True,

            exist_ok=True,

        )

        with open(

            TRAINING_INFO_PATH,

            "w",

            encoding="utf-8",

        ) as file:

            json.dump(

                info,

                file,

                indent=4,

                ensure_ascii=False,

            )

        logger.info(

            "training_info.json berhasil disimpan."

        )
        # ======================================================
    # LOAD TRAINING INFO
    # ======================================================

    @staticmethod
    def load_training_info() -> dict:
        """
        Membaca training_info.json.
        """

        if not TRAINING_INFO_PATH.exists():

            return {

                "status": "never_trained",

                "last_training": None,

                "duration": 0,

                "dataset_size": 0,

                "model_path": None,

                "vectorizer_path": None,

                "label_encoder_path": None,

            }

        with open(
            TRAINING_INFO_PATH,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    def train(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> TrainingResult:
        """
        Menjalankan pipeline retraining:

        1. train.py
        2. generate_predictions.py

        Sama seperti workflow manual di CMD.
        """

        started_at = datetime.now()

        # ==================================================
        # 1. PREPARING
        # ==================================================

        self._emit_progress(
            progress_callback,
            "Preparing",
            5,
        )

        logger.info("=" * 60)
        logger.info("NAZALYST RETRAINING PIPELINE")
        logger.info("=" * 60)

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {self.dataset_path}"
            )

        if not self.train_script.exists():
            raise FileNotFoundError(
                f"Training script tidak ditemukan: {self.train_script}"
            )

        if not PREDICTION_SCRIPT.exists():
            raise FileNotFoundError(
                f"Prediction script tidak ditemukan: {PREDICTION_SCRIPT}"
            )

        # ==================================================
        # 2. TRAIN MODEL
        # ==================================================

        self._emit_progress(
            progress_callback,
            "Training model...",
            20,
        )

        train_command = [
            sys.executable,
            str(self.train_script),
            "--csv",
            str(self.dataset_path),
        ]

        logger.info("TRAIN COMMAND:")
        logger.info(" ".join(train_command))

        train_process = subprocess.run(
            train_command,
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR),
        )

        # Tampilkan output training ke log
        if train_process.stdout:
            logger.info("TRAIN STDOUT:\n%s", train_process.stdout)

        if train_process.returncode != 0:
            error_message = (
                "Training model gagal.\n\n"
                f"Exit code: {train_process.returncode}\n\n"
                f"STDOUT:\n{train_process.stdout}\n\n"
                f"STDERR:\n{train_process.stderr}"
            )

            logger.error(error_message)

            raise RuntimeError(error_message)

        logger.info("Training model berhasil.")

        # ==================================================
        # 3. TRAINING SELESAI
        # ==================================================

        self._emit_progress(
            progress_callback,
            "Training selesai",
            60,
        )

        # ==================================================
        # 4. GENERATE PREDICTION
        # ==================================================

        self._emit_progress(
            progress_callback,
            "Generating predictions...",
            70,
        )

        prediction_command = [
            sys.executable,
            str(PREDICTION_SCRIPT),
        ]

        logger.info("PREDICTION COMMAND:")
        logger.info(" ".join(prediction_command))

        prediction_process = subprocess.run(
            prediction_command,
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR),
        )

        # Tampilkan output prediction ke log
        if prediction_process.stdout:
            logger.info(
                "PREDICTION STDOUT:\n%s",
                prediction_process.stdout,
            )

        if prediction_process.returncode != 0:
            error_message = (
                "Generate prediction gagal.\n\n"
                f"Exit code: {prediction_process.returncode}\n\n"
                f"STDOUT:\n{prediction_process.stdout}\n\n"
                f"STDERR:\n{prediction_process.stderr}"
            )

            logger.error(error_message)

            raise RuntimeError(error_message)

        logger.info("Generate prediction berhasil.")

        # ==================================================
        # 5. VALIDASI OUTPUT
        # ==================================================

        self._emit_progress(
            progress_callback,
            "Validating results...",
            90,
        )

        model_path = MODEL_DIR / "model.pkl"
        vectorizer_path = MODEL_DIR / "vectorizer.pkl"
        label_encoder_path = MODEL_DIR / "label_encoder.pkl"
        prediction_path = DATA_DIR / "hasil_prediksi.csv"

        required_files = {
            "model": model_path,
            "vectorizer": vectorizer_path,
            "label_encoder": label_encoder_path,
            "hasil_prediksi": prediction_path,
        }

        missing_files = [
            name
            for name, path in required_files.items()
            if not path.exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                "Retraining selesai tetapi file berikut tidak ditemukan: "
                + ", ".join(missing_files)
            )

        # ==================================================
        # 6. SAVE TRAINING INFO
        # ==================================================

        finished_at = datetime.now()

        duration = (
            finished_at - started_at
        ).total_seconds()

        dataset_size = len(
            pd.read_csv(self.dataset_path)
        )

        prediction_size = len(
            pd.read_csv(prediction_path)
        )

        info = {
            "status": "success",
            "last_training": finished_at.isoformat(),
            "duration": duration,
            "dataset_size": dataset_size,
            "prediction_size": prediction_size,
            "model_path": str(model_path),
            "vectorizer_path": str(vectorizer_path),
            "label_encoder_path": str(label_encoder_path),
            "prediction_path": str(prediction_path),
        }

        self.save_training_info(info)

        # ==================================================
        # 7. FINISHED
        # ==================================================

        self._emit_progress(
            progress_callback,
            "Retrain selesai",
            100,
        )

        logger.info("=" * 60)
        logger.info("RETRAINING SELESAI")
        logger.info("Dataset       : %s", dataset_size)
        logger.info("Prediction    : %s", prediction_size)
        logger.info("Duration      : %.2f sec", duration)
        logger.info("=" * 60)

        return TrainingResult(
            status="success",
            started_at=started_at,
            finished_at=finished_at,
            duration=duration,
            message=(
                "Training model dan generate prediction "
                "berhasil."
            ),
        )
        # ======================================================
    # MODEL INFORMATION
    # ======================================================

    @staticmethod
    def model_information() -> dict:
        """
        Mengambil informasi file model.
        """

        return {

            "model_exists": (MODEL_DIR / "model.pkl").exists(),

            "vectorizer_exists": (
                MODEL_DIR / "vectorizer.pkl"
            ).exists(),

            "label_encoder_exists": (
                MODEL_DIR / "label_encoder.pkl"
            ).exists(),

            "training_info_exists": (
                TRAINING_INFO_PATH.exists()
            ),

        }

    # ======================================================
    # SUMMARY
    # ======================================================

    def summary(self) -> dict:
        """
        Ringkasan informasi training.
        """

        training_info = self.load_training_info()

        model_info = self.model_information()

        return {

            **training_info,

            **model_info,

        }
    # ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    trainer = ModelTrainer()

    result = trainer.train()

    summary = trainer.summary()

    print()

    print("=" * 60)

    print("TRAINING RESULT")

    print("=" * 60)

    print(f"Status        : {result.status}")

    print(f"Started At    : {result.started_at}")

    print(f"Finished At   : {result.finished_at}")

    print(f"Duration      : {result.duration:.2f} sec")

    print(f"Message       : {result.message}")

    print()

    print("=" * 60)

    print("MODEL INFORMATION")

    print("=" * 60)

    for key, value in summary.items():

        print(f"{key:<25}: {value}")

    print("=" * 60)
