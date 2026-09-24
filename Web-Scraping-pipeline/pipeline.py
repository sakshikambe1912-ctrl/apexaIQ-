"""
Web Scraping Data Extraction Pipeline
======================================

A fault-tolerant pipeline that:
  1. Fetches HTML from multiple pages (live URLs or local fixtures).
  2. Extracts structured fields using CSS selectors.
  3. Cleans and validates each record.
  4. Persists raw + cleaned datasets (JSON / CSV).
  5. Produces an analytics report (counts, timing, price/rating stats).

Design goals
------------
* One bad page must never kill the whole run -> per-page try/except.
* Every stage is a small, independently testable class (single responsibility).
* Retries with exponential backoff on transient network failures.
* Clear, structured logging instead of bare prints.
* Works fully offline against local HTML fixtures (see demo.py) as well as
  against real URLs when network + selectors are supplied.

Author: Python Intern Advanced Assignments - Problem 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def build_logger(name: str = "scraping_pipeline", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ExtractedRecord:
    source_url: str
    title: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    availability: Optional[str] = None
    description: Optional[str] = None
    extracted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PipelineStats:
    total_pages: int = 0
    successful_pages: List[str] = field(default_factory=list)
    failed_pages: List[Dict[str, str]] = field(default_factory=list)
    total_records_extracted: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    skipped_records: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def total_time_seconds(self) -> float:
        return round(self.end_time - self.start_time, 4)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("start_time", None)
        d.pop("end_time", None)
        d["total_time_seconds"] = self.total_time_seconds
        return d


# ---------------------------------------------------------------------------
# Stage 1: Fetching
# ---------------------------------------------------------------------------

class PageFetchError(Exception):
    """Raised when a page cannot be retrieved after all retries."""


class PageFetcher:
    """Fetches raw HTML either from the live web or from local fixture files.

    Passing `local_dir` switches the fetcher into offline/demo mode, where
    each URL is mapped to a fixture file named after its path
    (e.g. https://shop.test/products/1 -> products_1.html).
    """

    def __init__(
        self,
        timeout: float = 10.0,
        retries: int = 2,
        backoff: float = 1.5,
        user_agent: Optional[str] = None,
        local_dir: Optional[Path] = None,
    ):
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.headers = {"User-Agent": user_agent or "Mozilla/5.0 (compatible; InternScraperBot/1.0)"}
        self.local_dir = Path(local_dir) if local_dir else None

    def fetch(self, url: str) -> str:
        if self.local_dir is not None:
            return self._fetch_local(url)
        return self._fetch_remote(url)

    def _fixture_name(self, url: str) -> str:
        path = urlparse(url).path.strip("/").replace("/", "_")
        return f"{path or 'index'}.html"

    def _fetch_local(self, url: str) -> str:
        fixture = self.local_dir / self._fixture_name(url)
        if not fixture.exists():
            raise PageFetchError(f"no local fixture for '{url}' (expected {fixture.name})")
        return fixture.read_text(encoding="utf-8")

    def _fetch_remote(self, url: str) -> str:
        if requests is None:
            raise PageFetchError("the 'requests' package is not installed")
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.retries + 2):
            try:
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry any failure
                last_exc = exc
                if attempt <= self.retries:
                    time.sleep(self.backoff ** attempt)
        raise PageFetchError(str(last_exc)) from last_exc


# ---------------------------------------------------------------------------
# Stage 2: Extraction
# ---------------------------------------------------------------------------

class DataExtractor:
    """Extracts one or more structured records from a page's HTML.

    `selectors` maps output field name -> CSS selector, evaluated relative
    to each matched item. If `item_selector` is None, the whole page is
    treated as a single item (one record per page).
    """

    DEFAULT_SELECTORS = {
        "title": ".title",
        "price": ".price",
        "rating": ".rating",
        "availability": ".availability",
        "description": ".description",
    }

    def __init__(self, selectors: Optional[Dict[str, str]] = None, item_selector: Optional[str] = None):
        if BeautifulSoup is None:
            raise RuntimeError("beautifulsoup4 is required for extraction")
        self.selectors = selectors or self.DEFAULT_SELECTORS
        self.item_selector = item_selector or ".item"

    def extract(self, html: str, source_url: str) -> List[ExtractedRecord]:
        soup = BeautifulSoup(html, "html.parser")
        items = soup.select(self.item_selector)
        if not items:
            return []  # no matching items on this page -> caller treats it as skipped

        records: List[ExtractedRecord] = []
        for item in items:
            raw: Dict[str, Optional[str]] = {}
            for field_name, selector in self.selectors.items():
                el = item.select_one(selector)
                raw[field_name] = el.get_text(strip=True) if el else None

            records.append(ExtractedRecord(
                source_url=source_url,
                title=raw.get("title"),
                price=self._parse_price(raw.get("price")),
                rating=self._parse_number(raw.get("rating")),
                availability=raw.get("availability"),
                description=raw.get("description"),
                raw=raw,
            ))
        return records

    @staticmethod
    def _parse_price(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        cleaned = re.sub(r"[^\d.]", "", value)
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None

    @staticmethod
    def _parse_number(value: Optional[str]) -> Optional[float]:
        if not value:
            return None
        match = re.search(r"[\d.]+", value)
        try:
            return float(match.group()) if match else None
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Stage 3: Cleaning + Validation
# ---------------------------------------------------------------------------

class DataCleaner:
    """Normalizes whitespace/casing so validation and analytics are stable."""

    def clean(self, record: ExtractedRecord) -> ExtractedRecord:
        if record.title:
            record.title = re.sub(r"\s+", " ", record.title).strip()
        if record.description:
            record.description = re.sub(r"\s+", " ", record.description).strip()
        if record.availability:
            record.availability = record.availability.strip().lower()
        return record


class DataValidator:
    """Applies business rules; a record failing any rule is marked invalid
    (but still kept in the raw dataset for auditability)."""

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        price_range: Tuple[float, float] = (0.0, 1_000_000.0),
        rating_range: Tuple[float, float] = (0.0, 5.0),
    ):
        self.required_fields = required_fields or ["title"]
        self.price_range = price_range
        self.rating_range = rating_range

    def validate(self, record: ExtractedRecord) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        for f in self.required_fields:
            if not getattr(record, f, None):
                errors.append(f"missing required field: {f}")
        if record.price is not None and not (self.price_range[0] <= record.price <= self.price_range[1]):
            errors.append(f"price out of range: {record.price}")
        if record.rating is not None and not (self.rating_range[0] <= record.rating <= self.rating_range[1]):
            errors.append(f"rating out of range: {record.rating}")
        return (len(errors) == 0, errors)


# ---------------------------------------------------------------------------
# Stage 4: Storage
# ---------------------------------------------------------------------------

class DataStorage:
    """Persists raw and cleaned datasets plus the final report to disk."""

    CSV_FIELDS = ["source_url", "title", "price", "rating", "availability", "description", "extracted_at"]

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_raw_json(self, records: List[ExtractedRecord], filename: str = "raw_data.json") -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps([r.to_dict() for r in records], indent=2), encoding="utf-8")
        return path

    def save_cleaned_csv(self, records: List[ExtractedRecord], filename: str = "cleaned_data.csv") -> Path:
        path = self.output_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for r in records:
                writer.writerow({k: getattr(r, k) for k in self.CSV_FIELDS})
        return path

    def save_json(self, data: dict, filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path


# ---------------------------------------------------------------------------
# Stage 5: Reporting
# ---------------------------------------------------------------------------

class ReportGenerator:
    def __init__(self, stats: PipelineStats, valid_records: List[ExtractedRecord]):
        self.stats = stats
        self.records = valid_records

    def generate(self) -> dict:
        prices = [r.price for r in self.records if r.price is not None]
        ratings = [r.rating for r in self.records if r.rating is not None]
        return {
            "summary": self.stats.to_dict(),
            "analytics": {
                "records_with_price": len(prices),
                "average_price": round(sum(prices) / len(prices), 2) if prices else None,
                "min_price": min(prices) if prices else None,
                "max_price": max(prices) if prices else None,
                "records_with_rating": len(ratings),
                "average_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            },
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def render_text(self) -> str:
        r = self.generate()
        s, a = r["summary"], r["analytics"]
        lines = [
            "=" * 60,
            "PIPELINE REPORT",
            "=" * 60,
            f"Total pages attempted : {s['total_pages']}",
            f"Successful pages      : {len(s['successful_pages'])}",
            f"Failed pages          : {len(s['failed_pages'])}",
            f"Total records         : {s['total_records_extracted']}",
            f"Valid records         : {s['valid_records']}",
            f"Invalid records       : {s['invalid_records']}",
            f"Skipped records       : {s['skipped_records']}",
            f"Total time (s)        : {s['total_time_seconds']}",
            "-" * 60,
            f"Avg price  : {a['average_price']}",
            f"Price range: {a['min_price']} - {a['max_price']}",
            f"Avg rating : {a['average_rating']}",
        ]
        if s["failed_pages"]:
            lines.append("-" * 60)
            lines.append("Failed pages detail:")
            for fp in s["failed_pages"]:
                lines.append(f"  - {fp['url']}: {fp['error']}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class ScrapingPipeline:
    """Wires all stages together and guarantees per-page fault isolation:
    a raised exception on any single page is caught, logged, recorded in
    `failed_pages`, and processing continues with the next URL."""

    def __init__(
        self,
        urls: List[str],
        fetcher: PageFetcher,
        extractor: DataExtractor,
        cleaner: DataCleaner,
        validator: DataValidator,
        storage: DataStorage,
        logger: Optional[logging.Logger] = None,
    ):
        self.urls = urls
        self.fetcher = fetcher
        self.extractor = extractor
        self.cleaner = cleaner
        self.validator = validator
        self.storage = storage
        self.logger = logger or build_logger()
        self.stats = PipelineStats(total_pages=len(urls))

    def run(self) -> dict:
        self.stats.start_time = time.time()
        all_records: List[ExtractedRecord] = []
        valid_records: List[ExtractedRecord] = []

        for url in self.urls:
            page_start = time.time()
            try:
                html = self.fetcher.fetch(url)
                records = self.extractor.extract(html, url)

                if not records:
                    self.logger.warning("no records found on %s -> skipping", url)
                    self.stats.skipped_records += 1
                    self.stats.successful_pages.append(url)
                    continue

                for record in records:
                    record = self.cleaner.clean(record)
                    all_records.append(record)
                    is_valid, errors = self.validator.validate(record)
                    if is_valid:
                        valid_records.append(record)
                        self.stats.valid_records += 1
                    else:
                        self.stats.invalid_records += 1
                        self.logger.info("invalid record from %s: %s", url, "; ".join(errors))

                self.stats.successful_pages.append(url)
                self.logger.info(
                    "processed %s -> %d record(s) in %.2fs", url, len(records), time.time() - page_start
                )

            except Exception as exc:  # noqa: BLE001 - isolate faults per page, never abort the run
                self.stats.failed_pages.append({"url": url, "error": str(exc)})
                self.logger.error("failed to process %s: %s", url, exc)
                continue

        self.stats.total_records_extracted = len(all_records)
        self.stats.end_time = time.time()

        raw_path = self.storage.save_raw_json(all_records)
        cleaned_path = self.storage.save_cleaned_csv(valid_records)

        report_gen = ReportGenerator(self.stats, valid_records)
        report = report_gen.generate()
        report_path = self.storage.save_json(report, "report.json")

        self.logger.info("\n" + report_gen.render_text())

        return {
            "raw_path": str(raw_path),
            "cleaned_path": str(cleaned_path),
            "report_path": str(report_path),
            "report": report,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    return json.loads(Path(config_path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Web scraping data extraction pipeline")
    parser.add_argument("--config", type=str, help="Path to a JSON config file (see README).")
    parser.add_argument("--urls", nargs="*", help="One or more URLs to scrape (overrides config urls).")
    parser.add_argument("--local-dir", type=str, help="Directory of local HTML fixtures (offline mode).")
    parser.add_argument("--output-dir", type=str, default="output", help="Where to write results.")
    parser.add_argument("--item-selector", type=str, default=".item", help="CSS selector for repeated items.")
    parser.add_argument("--required-fields", nargs="*", default=["title"], help="Fields that must be present.")
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    urls = args.urls or cfg.get("urls", [])
    if not urls:
        parser.error("no URLs provided (use --urls or --config)")

    logger = build_logger()

    fetcher = PageFetcher(local_dir=Path(args.local_dir) if args.local_dir else None)
    extractor = DataExtractor(
        selectors=cfg.get("selectors"),
        item_selector=cfg.get("item_selector", args.item_selector),
    )
    cleaner = DataCleaner()
    validator = DataValidator(required_fields=cfg.get("required_fields", args.required_fields))
    storage = DataStorage(Path(args.output_dir))

    pipeline = ScrapingPipeline(urls, fetcher, extractor, cleaner, validator, storage, logger)
    pipeline.run()


if __name__ == "__main__":
    main()