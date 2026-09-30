"""
Take a screenshot of the Streamlit dashboard from inside the docker
network using Playwright + headless Chromium. Runs against
http://dashboard:8501 (the Docker service name) so we don't need any
host port mapping.

Three screenshots:
  - 00_overview.png    : full-page after initial render
  - 01_district.png    : after clicking the District Detail tab
  - 02_model.png       : after clicking the Model tab
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = "http://dashboard:8501"
OUT_DIR = Path("/opt/airflow/data/dashboard_screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wait_for_streamlit_ready(page, timeout_s: int = 30) -> None:
    """Streamlit renders client-side; wait for the script to mark ready."""
    page.wait_for_load_state("networkidle", timeout=timeout_s * 1000)
    # The 'Running' indicator is on while a script run is in flight
    end = time.time() + timeout_s
    while time.time() < end:
        running = page.locator("[data-testid='stStatusWidget']").count()
        if running == 0:
            return
        time.sleep(0.5)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1480, "height": 900})
        page = context.new_page()

        print(f"[load] {URL}")
        page.goto(URL, wait_until="domcontentloaded", timeout=30_000)
        wait_for_streamlit_ready(page)
        # Give the Folium map an extra moment to fetch tiles
        time.sleep(3.5)

        page.screenshot(path=str(OUT_DIR / "00_overview.png"), full_page=True)
        print("  saved 00_overview.png")

        # Tab #2 — District Detail
        page.locator("[data-baseweb='tab']").nth(1).click()
        wait_for_streamlit_ready(page)
        time.sleep(2.0)
        page.screenshot(path=str(OUT_DIR / "01_district.png"), full_page=True)
        print("  saved 01_district.png")

        # Tab #3 — Model
        page.locator("[data-baseweb='tab']").nth(2).click()
        wait_for_streamlit_ready(page)
        time.sleep(1.5)
        page.screenshot(path=str(OUT_DIR / "02_model.png"), full_page=True)
        print("  saved 02_model.png")

        browser.close()
    print(f"\nAll screenshots saved under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
