"""
=========================================================
NAZALYST
Google Play Scraper Service
=========================================================
"""

from __future__ import annotations

import logging
import time

from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import pandas as pd

from google_play_scraper import reviews
from google_play_scraper import Sort


# ==========================================================
# CONFIGURATION
# ==========================================================

APP_PACKAGE = "com.dafturn.mypertamina"

LANGUAGE = "id"

COUNTRY = "id"

PAGE_SIZE = 200

MAX_RETRY = 3

RETRY_DELAY = 2


# ==========================================================
# LOGGER
# ==========================================================

logger = logging.getLogger(__name__)


# ==========================================================
# RESULT
# ==========================================================

@dataclass(slots=True)
class ScrapingResult:
    """
    Hasil scraping Google Play.
    """

    status: str

    reviews: pd.DataFrame

    fetched_count: int

    new_count: int

    duplicate_count: int

    latest_review: Optional[datetime]


# ==========================================================
# SCRAPER
# ==========================================================

class GooglePlayScraper:

    """
    Google Play Review Scraper.
    """

    def __init__(
        self,
        package_name: str = APP_PACKAGE,
        language: str = LANGUAGE,
        country: str = COUNTRY,
        page_size: int = PAGE_SIZE,
        max_retry: int = MAX_RETRY,
    ):

        self.package_name = package_name

        self.language = language

        self.country = country

        self.page_size = page_size

        self.max_retry = max_retry

    # ======================================================
    # PRIVATE REQUEST
    # ======================================================

    def _request_reviews(
        self,
        continuation_token=None,
    ):
        """
        Mengambil satu halaman review.
        """

        last_error = None

        for attempt in range(self.max_retry):

            try:

                return reviews(

                    self.package_name,

                    lang=self.language,

                    country=self.country,

                    sort=Sort.NEWEST,

                    count=self.page_size,

                    continuation_token=continuation_token,

                )

            except Exception as exc:

                last_error = exc

                logger.warning(

                    "Retry %s/%s",

                    attempt + 1,

                    self.max_retry,

                )

                time.sleep(RETRY_DELAY)

        raise RuntimeError(

            "Gagal mengambil review Google Play."

        ) from last_error

    # ======================================================
    # CALLBACK
    # ======================================================

    @staticmethod
    def _emit_progress(
        callback: Optional[Callable],
        stage: str,
        page: int,
        fetched: int,
        new_reviews: int,
        duplicate_reviews: int,
    ) -> None:

        if callback is None:
            return

        callback(
            {
                "stage": stage,
                "page": page,
                "fetched": fetched,
                "new_reviews": new_reviews,
                "duplicate_reviews": duplicate_reviews,
            }
        )

    # ======================================================
    # LABEL
    # ======================================================

    @staticmethod
    def map_label(score: int) -> str:

        if score >= 4:
            return "positif"

        if score == 3:
            return "netral"

        return "negatif"

    # ======================================================
    # CLEAN TEXT
    # ======================================================

    @staticmethod
    def clean_text(text: str) -> str:

        if text is None:
            return ""

        text = str(text)

        text = text.replace("\n", " ")

        text = text.replace("\r", " ")

        text = " ".join(text.split())

        return text.strip()

    # ======================================================
    # CLEAN DATAFRAME
    # ======================================================

    def clean_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        if dataframe.empty:
            return dataframe

        dataframe = dataframe.copy()

        dataframe["text"] = (
            dataframe["text"]
            .fillna("")
            .astype(str)
            .apply(self.clean_text)
        )

        dataframe = dataframe[
            dataframe["text"].str.len() > 0
        ]

        dataframe["created_at"] = (
            pd.to_datetime(
                dataframe["created_at"],
                errors="coerce"
            )
            .dt.tz_localize(None)
        )

        dataframe.reset_index(
            drop=True,
            inplace=True,
        )

        return dataframe
        # ======================================================
    # FETCH REVIEWS
    # ======================================================

    def fetch_reviews(
        self,
        existing_review_ids: Optional[Set[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> ScrapingResult:
        """
        Mengambil review terbaru dari Google Play Store.

        Parameters
        ----------
        existing_review_ids : Optional[Set[str]]
            Kumpulan review ID yang sudah ada pada dataset lokal.
            Digunakan untuk menghindari duplicate.

        progress_callback : Optional[Callable]
            Callback untuk menampilkan progress di Streamlit.

        Returns
        -------
        ScrapingResult
        """

        if existing_review_ids is None:
            existing_review_ids = set()

        logger.info("Start scraping Google Play review...")

        continuation_token = None

        page = 0

        fetched_count = 0

        duplicate_count = 0

        latest_review = None

        rows: List[Dict] = []

        while True:

            page += 1

            logger.info("Downloading page %s", page)

            result, continuation_token = self._request_reviews(
                continuation_token
            )

            if not result:

                logger.info("No review returned.")

                break

            duplicate_in_page = 0

            for review in result:

                fetched_count += 1

                review_id = str(review["reviewId"])

                review_date = pd.Timestamp(review["at"]).tz_localize(None)

                if latest_review is None:
                    latest_review = review_date
                elif review_date > latest_review:
                    latest_review = review_date

                # ------------------------------------------
                # Skip review yang sudah ada
                # ------------------------------------------

                if review_id in existing_review_ids:

                    duplicate_count += 1

                    duplicate_in_page += 1

                    continue

                score = int(review["score"])

                rows.append(
                    {
                        "text": self.clean_text(
                            review["content"]
                        ),
                        "label": self.map_label(score),
                        "created_at": review_date,
                        "score": score,
                        "author": review["userName"],
                        "id": review_id,
                    }
                )

            # ------------------------------------------
            # Progress callback
            # ------------------------------------------

            self._emit_progress(

                callback=progress_callback,

                stage="Downloading",

                page=page,

                fetched=fetched_count,

                new_reviews=len(rows),

                duplicate_reviews=duplicate_count,

            )

            logger.info(

                "Page %s | New=%s | Duplicate=%s",

                page,

                len(rows),

                duplicate_count,

            )

            # ------------------------------------------
            # Seluruh halaman duplicate
            # ------------------------------------------

            if duplicate_in_page == len(result):

                logger.info(
                    "Duplicate page detected."
                )

                break

            # ------------------------------------------
            # Halaman terakhir
            # ------------------------------------------

            if continuation_token is None:

                logger.info(
                    "End of pagination."
                )

                break

        dataframe = pd.DataFrame(rows)

        dataframe = self.clean_dataframe(
            dataframe
        )

        logger.info("Scraping finished.")

        logger.info(
            "Fetched=%s | New=%s | Duplicate=%s",
            fetched_count,
            len(dataframe),
            duplicate_count,
        )

        return ScrapingResult(

            status="success",

            reviews=dataframe,

            fetched_count=fetched_count,

            new_count=len(dataframe),

            duplicate_count=duplicate_count,

            latest_review=latest_review,

        )
        # ======================================================
    # SCRAPE
    # ======================================================

    def scrape(
        self,
        existing_review_ids: Optional[Set[str]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> ScrapingResult:
        """
        Menjalankan proses scraping.

        Method ini merupakan entry point utama yang akan
        dipanggil oleh updater.py.

        Parameters
        ----------
        existing_review_ids
            Kumpulan review ID yang sudah dimiliki dataset.

        progress_callback
            Callback progress (opsional).

        Returns
        -------
        ScrapingResult
        """

        logger.info("=" * 60)
        logger.info("NAZALYST GOOGLE PLAY SCRAPER")
        logger.info("=" * 60)

        result = self.fetch_reviews(
            existing_review_ids=existing_review_ids,
            progress_callback=progress_callback,
        )

        logger.info("Status      : %s", result.status)
        logger.info("Fetched     : %s", result.fetched_count)
        logger.info("New Review  : %s", result.new_count)
        logger.info("Duplicate   : %s", result.duplicate_count)
        logger.info("Latest      : %s", result.latest_review)

        logger.info("=" * 60)

        return result
    # ==========================================================
# LOCAL TEST
# ==========================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    scraper = GooglePlayScraper()

    result = scraper.scrape()

    print("\n")
    print("=" * 60)
    print("NAZALYST SCRAPER RESULT")
    print("=" * 60)

    print(f"Status      : {result.status}")
    print(f"Fetched     : {result.fetched_count}")
    print(f"New Review  : {result.new_count}")
    print(f"Duplicate   : {result.duplicate_count}")
    print(f"Latest      : {result.latest_review}")

    if result.reviews.empty:

        print("\nTidak ada review baru.")

    else:

        print("\nPreview Review Baru")
        print(result.reviews.head(10))

    print("=" * 60)