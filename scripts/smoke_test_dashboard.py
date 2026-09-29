"""Open the published dashboard in a real browser and check that it works.

    python scripts/smoke_test_dashboard.py https://matthijssloep.github.io/Impact-measurement/ [out_dir]

Waits for the in-browser Streamlit app (stlite) to boot, then checks that the
title and headline numbers render, that every tab opens without a Python
error, and that both languages work. Saves a screenshot per tab to out_dir.
Exits non-zero on any failure.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BOOT_TIMEOUT_MS = 240_000  # first load downloads Python (Pyodide) and packages
TABS_NL = ["Overzicht", "Impact per doel", "KWF × IKNL", "Projecten", "Artikelen", "Methode"]


def main() -> int:
    url = sys.argv[1]
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "smoke-screenshots")
    out.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    console_errors: list[str] = []

    with sync_playwright() as p:
        # CHROMIUM_PATH: use a preinstalled browser instead of Playwright's own download.
        browser = p.chromium.launch(executable_path=os.environ.get("CHROMIUM_PATH") or None)
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        page.on("console", lambda m: m.type == "error" and console_errors.append(m.text))
        page.on("pageerror", lambda e: console_errors.append(str(e)))

        page.goto(url, wait_until="domcontentloaded")
        # stlite renders the app into the same page; wait for the Streamlit title.
        page.get_by_text("Impact van KWF-financiering").first.wait_for(timeout=BOOT_TIMEOUT_MS)
        page.wait_for_timeout(3000)
        print("App booted")

        for text in ("KWF-financiering", "Artikelen", "IKNL-artikelen"):
            if not page.get_by_text(text).first.is_visible():
                failures.append(f"missing headline '{text}'")

        for i, name in enumerate(TABS_NL):
            page.get_by_role("tab", name=name).click()
            page.wait_for_timeout(4000)
            errors = page.locator('[data-testid="stException"]')
            if errors.count():
                failures.append(f"tab '{name}': {errors.first.inner_text()[:300]}")
            charts = page.locator(".js-plotly-plot").count()
            print(f"tab '{name}': {charts} chart(s), {errors.count()} error(s)")
            page.screenshot(path=str(out / f"{i + 1}_{name.replace(' ', '_').replace('×', 'x')}.png"),
                            full_page=True)

        page.get_by_text("English").first.click()
        try:
            page.get_by_text("Impact of KWF funding").first.wait_for(timeout=30_000)
            print("English switch works")
        except Exception:  # noqa: BLE001
            failures.append("language switch to English did not render")
        page.screenshot(path=str(out / "7_english.png"), full_page=True)
        browser.close()

    python_errors = [e for e in console_errors if "Traceback" in e or "Error" in e]
    for e in python_errors[:10]:
        print("console:", e[:300])
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("All dashboard checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
