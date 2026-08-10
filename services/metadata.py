"""
=========================================================
NAZALYST
Metadata Service
=========================================================
"""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# PATH
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

METADATA_PATH = DATA_DIR / "metadata.json"


# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class MetadataResult:

    status: str

    metadata: dict


# ==========================================================
# METADATA SERVICE
# ==========================================================

class MetadataService:

    """
    Service untuk membuat metadata dataset.
    """

    def __init__(
        self,
        metadata_path: Path = METADATA_PATH,
    ):

        self.metadata_path = metadata_path

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        # ======================================================
    # BUILD METADATA
    # ======================================================

    def build_metadata(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        if dataframe.empty:

            return {

                "total_reviews": 0,

                "positive_reviews": 0,

                "neutral_reviews": 0,

                "negative_reviews": 0,

                "average_rating": 0,

                "latest_review": None,

                "last_updated": datetime.now().isoformat(),

            }

        sentiment = dataframe["label"].value_counts()

        metadata = {

            "total_reviews": int(len(dataframe)),

            "positive_reviews": int(
                sentiment.get("positif", 0)
            ),

            "neutral_reviews": int(
                sentiment.get("netral", 0)
            ),

            "negative_reviews": int(
                sentiment.get("negatif", 0)
            ),

            "average_rating": round(
                dataframe["score"].mean(),
                2,
            ),

            "latest_review": str(
                dataframe["created_at"].max()
            ),

            "last_updated": datetime.now().isoformat(),

        }

        return metadata
        # ======================================================
    # SAVE METADATA
    # ======================================================

    def save_metadata(
        self,
        metadata: dict,
    ) -> None:
        """
        Menyimpan metadata ke metadata.json.
        """

        with open(
            self.metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
                ensure_ascii=False,
            )

        logger.info(
            "Metadata berhasil disimpan."
        )

    # ======================================================
    # LOAD METADATA
    # ======================================================

    def load_metadata(
        self,
    ) -> dict:
        """
        Membaca metadata.json.
        """

        if not self.metadata_path.exists():

            logger.info(
                "Metadata belum tersedia."
            )

            return self.build_metadata(
                pd.DataFrame()
            )

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            metadata = json.load(file)

        logger.info(
            "Metadata berhasil dibaca."
        )

        return metadata

    # ======================================================
    # UPDATE METADATA
    # ======================================================

    def update_metadata(
        self,
        dataframe: pd.DataFrame,
    ) -> MetadataResult:
        """
        Membuat dan menyimpan metadata terbaru.
        """

        logger.info("=" * 60)
        logger.info("UPDATE METADATA")
        logger.info("=" * 60)

        metadata = self.build_metadata(
            dataframe
        )

        self.save_metadata(
            metadata
        )

        logger.info(
            "Metadata berhasil diperbarui."
        )

        return MetadataResult(

            status="success",

            metadata=metadata,

        )
        # ======================================================
    # SUMMARY
    # ======================================================

    @staticmethod
    def summary(
        metadata: dict,
    ) -> dict:
        """
        Mengembalikan ringkasan metadata
        untuk digunakan oleh Dashboard.
        """

        return {

            "total_reviews": metadata.get(
                "total_reviews", 0
            ),

            "positive_reviews": metadata.get(
                "positive_reviews", 0
            ),

            "neutral_reviews": metadata.get(
                "neutral_reviews", 0
            ),

            "negative_reviews": metadata.get(
                "negative_reviews", 0
            ),

            "average_rating": metadata.get(
                "average_rating", 0
            ),

            "latest_review": metadata.get(
                "latest_review"
            ),

            "last_updated": metadata.get(
                "last_updated"
            ),

        }

    # ======================================================
    # EXISTS
    # ======================================================

    def exists(self) -> bool:
        """
        Mengecek apakah metadata.json tersedia.
        """

        return self.metadata_path.exists()

    # ======================================================
    # DELETE
    # ======================================================

    def delete(self) -> None:
        """
        Menghapus metadata.json.
        """

        if self.metadata_path.exists():

            self.metadata_path.unlink()

            logger.info(
                "Metadata berhasil dihapus."
            )
    # ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    service = MetadataService()

    sample = pd.DataFrame(

        {

            "text": [
                "Bagus",
                "Jelek",
                "Biasa"
            ],

            "label": [
                "positif",
                "negatif",
                "netral"
            ],

            "score": [
                5,
                1,
                3
            ],

            "created_at": pd.to_datetime(
                [
                    "2026-07-01",
                    "2026-07-02",
                    "2026-07-03",
                ]
            ),

            "author": [
                "A",
                "B",
                "C",
            ],

            "id": [
                "1",
                "2",
                "3",
            ],

        }

    )

    result = service.update_metadata(
        sample
    )

    summary = service.summary(
        result.metadata
    )

    print()

    print("=" * 60)

    print("METADATA RESULT")

    print("=" * 60)

    for key, value in summary.items():

        print(f"{key:<20}: {value}")

    print("=" * 60)