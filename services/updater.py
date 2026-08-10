"""
=========================================================
NAZALYST 
Dataset Updater Service
=========================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Optional
from typing import Set

import pandas as pd

from services.scraper import GooglePlayScraper
from services.scraper import ScrapingResult


# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# PATH
# ==========================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

DATASET_PATH = DATA_DIR / "dataset.csv"


# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class UpdateResult:
    """
    Hasil proses update dataset.
    """

    status: str

    previous_total: int

    current_total: int

    new_reviews: int

    duplicate_reviews: int

    latest_review: Optional[datetime]

    dataset: pd.DataFrame


# ==========================================================
# DATASET UPDATER
# ==========================================================

class DatasetUpdater:

    """
    Dataset Updater.

    Bertugas memperbarui dataset.csv
    menggunakan review terbaru.
    """

    def __init__(
        self,
        dataset_path: Path = DATASET_PATH,
    ):

        self.dataset_path = dataset_path

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.scraper = GooglePlayScraper()

    # ======================================================
    # LOAD DATASET
    # ======================================================

    def load_dataset(self) -> pd.DataFrame:

        if not self.dataset_path.exists():

            logger.info(
                "Dataset belum tersedia."
            )

            return pd.DataFrame(
                columns=[
                    "text",
                    "label",
                    "created_at",
                    "score",
                    "author",
                    "id",
                ]
            )

        logger.info(
            "Loading dataset..."
        )

        dataframe = pd.read_csv(
            self.dataset_path
        )

        if "created_at" in dataframe.columns:

            dataframe["created_at"] = (
                pd.to_datetime(
                    dataframe["created_at"],
                    utc=True,
                    errors="coerce",
                )
                .dt.tz_localize(None)
            )

        logger.info(
            "Total dataset : %s",
            len(dataframe),
        )

        return dataframe

    # ======================================================
    # SAVE DATASET
    # ======================================================

    def save_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        dataframe.to_csv(
            self.dataset_path,
            index=False,
            encoding="utf-8-sig",
        )

        logger.info(
            "Dataset berhasil disimpan."
        )

    # ======================================================
    # EXISTING REVIEW ID
    # ======================================================

    @staticmethod
    def get_existing_review_ids(
        dataframe: pd.DataFrame,
    ) -> Set[str]:

        if dataframe.empty:

            return set()

        return set(
            dataframe["id"]
            .astype(str)
            .tolist()
        )
        # ======================================================
    # MERGE DATASET
    # ======================================================

    @staticmethod
    def merge_dataset(
        old_dataset: pd.DataFrame,
        new_dataset: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Menggabungkan dataset lama dengan review baru.
        """

        if new_dataset.empty:
            return old_dataset.copy()

        if old_dataset.empty:
            merged = new_dataset.copy()
        else:
            merged = pd.concat(
                [new_dataset, old_dataset],
                ignore_index=True,
            )

        merged = merged.drop_duplicates(
            subset="id",
            keep="first",
        )

        merged = merged.sort_values(
            by="created_at",
            ascending=False,
        )

        merged.reset_index(
            drop=True,
            inplace=True,
        )

        return merged

    # ======================================================
    # UPDATE DATASET
    # ======================================================

    def update_dataset(
        self,
        progress_callback: Optional[Callable] = None,
    ) -> UpdateResult:
        """
        Memperbarui dataset menggunakan review terbaru
        dari Google Play Store.
        """

        logger.info("=" * 60)
        logger.info("UPDATE DATASET")
        logger.info("=" * 60)

        print("=== UPDATE DATASET START ===")
        
        # ---------------------------------------------
        # Load dataset lama
        # ---------------------------------------------

        old_dataset = self.load_dataset()

        previous_total = len(old_dataset)

        existing_review_ids = self.get_existing_review_ids(
            old_dataset
        )

        # ---------------------------------------------
        # Scraping review baru
        # ---------------------------------------------

        scraping_result: ScrapingResult = self.scraper.scrape(
            existing_review_ids=existing_review_ids,
            progress_callback=progress_callback,
        )

        # ---------------------------------------------
        # Tidak ada review baru
        # ---------------------------------------------

        if scraping_result.reviews.empty:

            logger.info(
                "Tidak ada review baru."
            )

            return UpdateResult(
                status="no_update",
                previous_total=previous_total,
                current_total=previous_total,
                new_reviews=0,
                duplicate_reviews=scraping_result.duplicate_count,
                latest_review=scraping_result.latest_review,
                dataset=old_dataset,
            )

        # ---------------------------------------------
        # Merge dataset
        # ---------------------------------------------

        merged_dataset = self.merge_dataset(
            old_dataset=old_dataset,
            new_dataset=scraping_result.reviews,
        )

        # ---------------------------------------------
        # Simpan dataset
        # ---------------------------------------------

        self.save_dataset(
            merged_dataset
        )

        logger.info(
            "Dataset berhasil diperbarui."
        )

        return UpdateResult(

            status="success",

            previous_total=previous_total,

            current_total=len(merged_dataset),

            new_reviews=scraping_result.new_count,

            duplicate_reviews=scraping_result.duplicate_count,

            latest_review=scraping_result.latest_review,

            dataset=merged_dataset,

        )
        # ======================================================
    # DATASET INFORMATION
    # ======================================================

    @staticmethod
    def dataset_information(
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Menghasilkan informasi dataset.

        Parameters
        ----------
        dataframe : pd.DataFrame

        Returns
        -------
        dict
        """

        if dataframe.empty:

            return {

                "total_reviews": 0,

                "positive": 0,

                "neutral": 0,

                "negative": 0,

                "latest_review": None,

            }

        sentiment = dataframe["label"].value_counts()

        latest_review = dataframe["created_at"].max()

        return {

            "total_reviews": len(dataframe),

            "positive": int(sentiment.get("positif", 0)),

            "neutral": int(sentiment.get("netral", 0)),

            "negative": int(sentiment.get("negatif", 0)),

            "latest_review": latest_review,

        }

    # ======================================================
    # SUMMARY
    # ======================================================

    @staticmethod
    def summary(
        result: UpdateResult,
    ) -> dict:
        """
        Ringkasan hasil update dataset.
        """

        return {

            "status": result.status,

            "previous_total": result.previous_total,

            "current_total": result.current_total,

            "new_reviews": result.new_reviews,

            "duplicate_reviews": result.duplicate_reviews,

            "latest_review": result.latest_review,

        }
    # ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(

        level=logging.INFO,

        format="%(asctime)s | %(levelname)s | %(message)s",

    )

    updater = DatasetUpdater()

    result = updater.update_dataset()

    info = updater.dataset_information(

        result.dataset

    )

    print()

    print("=" * 60)

    print("DATASET UPDATE RESULT")

    print("=" * 60)

    print(f"Status              : {result.status}")

    print(f"Previous Total      : {result.previous_total}")

    print(f"Current Total       : {result.current_total}")

    print(f"New Reviews         : {result.new_reviews}")

    print(f"Duplicate Reviews   : {result.duplicate_reviews}")

    print(f"Latest Review       : {result.latest_review}")

    print()

    print("=" * 60)

    print("DATASET INFORMATION")

    print("=" * 60)

    print(f"Total Review        : {info['total_reviews']}")

    print(f"Positive            : {info['positive']}")

    print(f"Neutral             : {info['neutral']}")

    print(f"Negative            : {info['negative']}")

    print(f"Latest Review       : {info['latest_review']}")

    print("=" * 60)