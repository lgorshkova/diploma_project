from __future__ import annotations

import os
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib.parse import urlencode, urljoin
from urllib3.util.retry import Retry


class AmazonScraper:
    """
    Faster Amazon scraper with:
    - HTTP-only category link scraping
    - Keep-alive session with connection pooling and retries
    - Concurrent product scraping
    - Selenium fallback only when blocked or critical fields missing

    Public API preserved:
    - __init__(domain: str, language: str, driver_path: Optional[str] = None)
    - close()
    - get_bestseller_links(category_url: str, max_products: int = 50) -> List[str]
    - scrape_product(url: str) -> Dict[str, str | List[str]]
    """

    def __init__(self, domain: str, language: str, driver_path: Optional[str] = None):
        self.domain = domain
        self.language = language
        self.driver_path = driver_path or self._detect_driver_path()
        self._driver = None  # lazy init only when needed

        # Compile regex once
        self._dp_re = re.compile(r"/dp/[A-Z0-9]{10}")

        # HTTP session with pooling & retries
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": self.language if self.language else "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Connection": "keep-alive",
            "Referer": f"https://{self.domain}/",
        })

        retry = Retry(
            total=2,
            backoff_factor=0.25,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50, max_retries=retry)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

    # ------------------------- Driver management -------------------------
    def _detect_driver_path(self) -> str:
        system = platform.system()
        if system == "Windows":
            return "chromedriver-win64/chromedriver.exe"
        elif system == "Linux":
            return "chromedriver-linux64/chromedriver"
        elif system == "Darwin":
            arch = platform.machine()
            if arch == "arm64":
                return "chromedriver-mac-arm64/chromedriver"
            else:
                return "chromedriver-mac-x64/chromedriver"
        else:
            raise OSError(f"Unsupported OS: {system}")

    def _ensure_driver(self):
        if self._driver is not None:
            return self._driver

        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-3d-apis")
        options.add_argument("--disable-webgl")
        options.add_argument("--log-level=3")
        options.add_argument(f"--lang={self.language}")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
        )
        options.page_load_strategy = "none"  # even faster than eager
        options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_experimental_option("prefs", {
            "profile.managed_default_content_settings.images": 2
        })
        options.add_experimental_option("excludeSwitches", ["enable-logging"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--disable-blink-features=AutomationControlled")

        service = Service(executable_path=self.driver_path, log_output=os.devnull)
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_window_size(1200, 800)
        driver.set_page_load_timeout(10)
        driver.implicitly_wait(0)

        # Mask webdriver flag
        try:
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined});"
            })
        except Exception:
            pass

        # Block heavy resources
        try:
            driver.execute_cdp_cmd("Network.enable", {})
            driver.execute_cdp_cmd("Network.setBlockedURLs", {
                "urls": [
                    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp",
                    "*.css", "*.woff", "*.woff2", "*.ttf", "*.otf", "*.svg"
                ]
            })
        except Exception:
            pass

        self._driver = driver
        return self._driver

    @property
    def driver(self):
        # kept for backward compatibility; lazily create when accessed
        return self._ensure_driver()

    def close(self):
        if getattr(self, "_session", None):
            try:
                self._session.close()
            except Exception:
                pass
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None

    # ------------------------- Category links -------------------------
    def _extract_dp_links(self, html: str, max_products: int, seen: set, out_links: List[str]) -> bool:
        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a.a-link-normal[href*='/dp/']"):
            href = a.get("href") or ""
            match = self._dp_re.search(href)
            if not match:
                continue
            full = urljoin(f"https://{self.domain}", match.group(0))
            if full in seen:
                continue
            seen.add(full)
            out_links.append(full)
            if len(out_links) >= max_products:
                return True
        return False

    def get_bestseller_links(self, category_url: str, max_products: int = 50) -> List[str]:
        links: List[str] = []
        seen: set = set()

        for pg in range(1, 6):
            sep = "&" if "?" in category_url else "?"
            url = category_url if pg == 1 else f"{category_url}{sep}{urlencode({'pg': pg})}"
            try:
                r = self._session.get(url, timeout=(5, 10))
            except Exception:
                continue
            if not r.ok:
                continue
            done = self._extract_dp_links(r.text, max_products, seen, links)
            if done:
                break
        return links[:max_products]

    # ------------------------- Product scraping -------------------------
    def _is_blocked_html(self, text_lower: str, status_code: int) -> bool:
        if status_code != 200:
            return True
        markers = [
            "robot check", "captcha", "validatecaptcha", "automated access",
            "sorry we just need to make sure", "enter the characters you see",
        ]
        return any(m in text_lower for m in markers)

    def _parse_product(self, soup: BeautifulSoup, url: str) -> Dict:
        title_tag = soup.select_one("#productTitle")
        rating_tag = soup.select_one("span.a-icon-alt")

        review_count_elem = soup.select_one("#acrCustomerReviewText")
        if not review_count_elem:
            review_count_elem = soup.find("span", attrs={"data-hook": "total-review-count"})
        if not review_count_elem:
            review_count_elem = soup.find("span", class_="a-size-base", attrs={"data-asin-reviews-link": True})

        review_count = ""
        if review_count_elem:
            text = review_count_elem.get_text(strip=True)
            digits = ''.join(filter(str.isdigit, text))
            review_count = digits

        price_tag = (
            soup.select_one("#priceblock_ourprice")
            or soup.select_one("#priceblock_dealprice")
            or soup.select_one("#priceblock_saleprice")
            or soup.select_one("span.a-price > span.a-offscreen")
        )

        bullets = [
            li.get_text(strip=True)
            for li in soup.select("#feature-bullets ul li")
            if li.get_text(strip=True)
        ]
        if not bullets:
            alt_bullets = soup.select(
                "div#productFactsDesktopExpander ul.a-unordered-list.a-vertical.a-spacing-small li"
            )
            bullets = [
                li.get_text(strip=True)
                for li in alt_bullets
                if li.get_text(strip=True)
            ]

        description = ""
        desc_tag = soup.select_one("#productDescription")
        if desc_tag:
            description = desc_tag.get_text(strip=True)
        else:
            aplus_tag = soup.select_one("div[data-cel-widget='aplus']")
            if aplus_tag:
                description = aplus_tag.get_text(strip=True)

        return {
            "url": url,
            "title": title_tag.get_text(strip=True) if title_tag else "",
            "rating": (rating_tag.get_text(strip=True).split(" ")[0] if rating_tag else ""),
            "review_count": review_count,
            "price": price_tag.get_text(strip=True) if price_tag else "",
            "bullet_points": bullets,
            "description": description,
        }

    def scrape_product(self, url: str) -> Dict:
        # HTTP fast path
        try:
            r = self._session.get(url, timeout=(5, 12), allow_redirects=True)
            tl = r.text.lower()
            if not self._is_blocked_html(tl, r.status_code):
                soup = BeautifulSoup(r.text, "lxml")
                result = self._parse_product(soup, url)
                if result["title"] or (result["bullet_points"] and result["description"]):
                    return result
        except Exception:
            pass

        # Selenium fallback
        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC

            driver = self._ensure_driver()
            driver.get(url)
            WebDriverWait(driver, 8).until(
                EC.any_of(
                    EC.presence_of_element_located((By.ID, "ppd")),
                    EC.presence_of_element_located((By.ID, "productTitle")),
                )
            )
            try:
                container = driver.find_element(By.ID, "ppd")
                html = container.get_attribute("innerHTML")
                soup = BeautifulSoup(html, "lxml")
            except Exception:
                soup = BeautifulSoup(driver.page_source, "lxml")
            return self._parse_product(soup, url)
        except Exception as e:
            return {"url": url, "error": str(e)}

    # ------------------------- Batch helpers -------------------------
    def scrape_many(self, urls: List[str], max_workers: int = 12) -> List[Dict]:
        results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self.scrape_product, u): u for u in urls}
            for fut in as_completed(future_to_url):
                url = future_to_url[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    results.append({"url": url, "error": str(e)})
        return results


__all__ = ["AmazonScraper"]

