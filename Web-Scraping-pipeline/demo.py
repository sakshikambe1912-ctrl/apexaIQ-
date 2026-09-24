"""
Demo runner for the web scraping pipeline.

Generates a handful of local HTML fixtures (simulating a small product
catalog site) and runs the full pipeline against them in offline mode, so
the entire flow can be exercised with no network access:

  - products_1.html : two valid product items
  - products_2.html : one valid item + one item missing its title (invalid)
  - products_3.html : a malformed/empty page (extracts zero records -> skipped)
  - products_4.html : intentionally NOT created, so its URL triggers a fetch
                       failure (missing fixture), demonstrating fault isolation

Run:
    python demo.py
"""

from pathlib import Path

from pipeline import (
    DataCleaner,
    DataExtractor,
    DataStorage,
    DataValidator,
    PageFetcher,
    ScrapingPipeline,
    build_logger,
)

FIXTURE_DIR = Path(__file__).parent / "sample_pages"
OUTPUT_DIR = Path(__file__).parent / "output"

BASE_URL = "https://shop.test"

PAGE_1 = """
<html><body>
  <div class="item">
    <span class="title">Wireless Mouse</span>
    <span class="price">$19.99</span>
    <span class="rating">4.3 out of 5</span>
    <span class="availability">In Stock</span>
    <p class="description">  Ergonomic wireless mouse   with USB receiver. </p>
  </div>
  <div class="item">
    <span class="title">Mechanical Keyboard</span>
    <span class="price">$89.50</span>
    <span class="rating">4.7 out of 5</span>
    <span class="availability">In Stock</span>
    <p class="description">Hot-swappable mechanical keyboard, RGB backlight.</p>
  </div>
</body></html>
"""

PAGE_2 = """
<html><body>
  <div class="item">
    <span class="title">USB-C Hub</span>
    <span class="price">$34.00</span>
    <span class="rating">4.1 out of 5</span>
    <span class="availability">Limited Stock</span>
    <p class="description">7-in-1 USB-C hub with HDMI and SD card slots.</p>
  </div>
  <div class="item">
    <!-- title missing on purpose -> fails validation -->
    <span class="price">$999999999</span>
    <span class="rating">7.5 out of 5</span>
    <span class="availability">Unknown</span>
    <p class="description">Corrupted listing with an out-of-range price and rating.</p>
  </div>
</body></html>
"""

PAGE_3 = """
<html><body>
  <p>This page has no product markup at all.</p>
</body></html>
"""


def write_fixtures() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "products_1.html").write_text(PAGE_1, encoding="utf-8")
    (FIXTURE_DIR / "products_2.html").write_text(PAGE_2, encoding="utf-8")
    (FIXTURE_DIR / "products_3.html").write_text(PAGE_3, encoding="utf-8")
    # products_4.html deliberately omitted to simulate a page fetch failure


def main() -> None:
    write_fixtures()
    logger = build_logger()

    urls = [
        f"{BASE_URL}/products_1",
        f"{BASE_URL}/products_2",
        f"{BASE_URL}/products_3",
        f"{BASE_URL}/products_4",  # no fixture -> will fail
    ]

    fetcher = PageFetcher(local_dir=FIXTURE_DIR)
    extractor = DataExtractor(item_selector=".item")
    cleaner = DataCleaner()
    validator = DataValidator(
        required_fields=["title"],
        price_range=(0, 5000),
        rating_range=(0, 5),
    )
    storage = DataStorage(OUTPUT_DIR)

    pipeline = ScrapingPipeline(urls, fetcher, extractor, cleaner, validator, storage, logger)
    result = pipeline.run()

    print(f"\nRaw dataset     -> {result['raw_path']}")
    print(f"Cleaned dataset -> {result['cleaned_path']}")
    print(f"Report          -> {result['report_path']}")


if __name__ == "__main__":
    main()