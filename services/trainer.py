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
        Menjalankan train.py.
        """

        started_at = datetime.now()

        self._emit_progress(
            progress_callback,
            "Preparing",
            5,
        )

        logger.info("=" * 60)
        logger.info("MODEL TRAINING")
        logger.info("=" * 60)

        if not self.dataset_path.exists():

            raise FileNotFoundError(
                f"Dataset tidak ditemukan: {self.dataset_path}"
            )

        self._emit_progress(
            progress_callback,
            "Training",
            20,
        )

        command = [

            "python",

            str(self.train_script),

            "--csv",

            str(self.dataset_path),

        ]

        logger.info(" ".join(command))

        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(ROOT_DIR)
        )
        
        if process.returncode != 0:
            error_message = (
                f"Training gagal dengan exit code {process.returncode}\n\n"
                f"STDOUT:\n{process.stdout}\n\n"
                f"STDERR:\n{process.stderr}"
            )
        
            logger.error(error_message)
        
            raise RuntimeError(error_message)

        self._emit_progress(
            progress_callback,
            "Saving",
            90,
        )

        finished_at = datetime.now()

        duration = (

            finished_at - started_at

        ).total_seconds()

        info = {

            "status": "success",

            "last_training": finished_at.isoformat(),

            "duration": duration,

            "dataset_size": len(
                pd.read_csv(self.dataset_path)
            ),

            "model_path": str(
                MODEL_DIR / "model.pkl"
            ),

            "vectorizer_path": str(
                MODEL_DIR / "vectorizer.pkl"
            ),

            "label_encoder_path": str(
                MODEL_DIR / "label_encoder.pkl"
            ),

        }

        self.save_training_info(
            info
        )

        self._emit_progress(
            progress_callback,
            "Finished",
            100,
        )

        logger.info(
            "Training selesai."
        )

        return TrainingResult(

            status="success",

            started_at=started_at,

            finished_at=finished_at,

            duration=duration,

            message="Training berhasil.",

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
