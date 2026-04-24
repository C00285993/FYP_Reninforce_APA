"""
Headless Browser Crawler (Playwright)

Fallback for sites where the static HTML parser finds 0 injection points:
  - JavaScript-rendered forms (React, Vue, Angular SPAs)
  - Forms submitted via fetch() / XMLHttpRequest (no <form> tag)
  - Hash-based SPA routes (/#/search, /#/profile)
  - Inputs revealed only after user interaction (modals, accordions)

Usage:
    crawler = HeadlessCrawler()
    if crawler.available:
        html, api_points = crawler.fetch(url, cookies=cookies)

Install:
    pip install playwright
    playwright install chromium
"""

import logging
import sys
import warnings
from typing import Optional
from urllib.parse import urljoin, urlsplit

from utils.generic_http_client import InjectionPoint

logger = logging.getLogger(__name__)


def _suppress_playwright_cleanup_noise():
    """Silence harmless asyncio/Playwright transport cleanup errors on Windows.

    When Playwright's browser subprocess is torn down, Python's asyncio
    ProactorEventLoop prints noisy 'Event loop is closed' / 'I/O operation
    on closed pipe' tracebacks.  These come from three sources:

    1. sys.unraisablehook  — __del__ exceptions
    2. asyncio logger      — "Task was destroyed but it is pending!"
    3. CPython C internals  — "Exception ignored in:" (bypasses Python hooks,
       so we redirect stderr temporarily during browser.close())

    All are benign but alarming to users.
    """
    if sys.platform != "win32":
        return

    # 1. __del__ errors go through sys.unraisablehook (Python 3.8+)
    _orig_unraisable = getattr(sys, "unraisablehook", None)

    def _quiet_unraisable(unraisable):
        msg = str(unraisable.exc_value) if unraisable.exc_value else ""
        if "Event loop is closed" in msg or "I/O operation on closed pipe" in msg:
            return  # swallow silently
        if _orig_unraisable:
            _orig_unraisable(unraisable)

    sys.unraisablehook = _quiet_unraisable

    # 2. asyncio "Task was destroyed" warnings go through logging
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)

    # 3. ResourceWarning about unclosed transports
    warnings.filterwarnings("ignore", message="unclosed transport", category=ResourceWarning)


def _is_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except ImportError:
        return False


class HeadlessCrawler:
    """
    Fetches a page using a real Chromium browser, waits for JS to settle,
    then returns:
      - The fully rendered HTML (for BeautifulSoup parsing)
      - InjectionPoints synthesised from intercepted POST/PUT/PATCH XHR requests
    """

    available: bool = _is_available()

    def fetch(
        self,
        url: str,
        cookies: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        wait: str = "networkidle",
        timeout: int = 20_000,
    ) -> tuple[str, list[InjectionPoint]]:
        """
        Load *url* in a headless Chromium browser.

        Returns:
            html        — fully JS-rendered page HTML
            api_points  — InjectionPoints from intercepted XHR/fetch POSTs
        """
        if not self.available:
            logger.warning("HeadlessCrawler.fetch() called but playwright is not installed.")
            return "", []

        from playwright.sync_api import sync_playwright

        intercepted: list[dict] = []

        def _on_request(request):
            if request.method in ("POST", "PUT", "PATCH"):
                try:
                    body = request.post_data or ""
                except Exception:
                    body = ""
                intercepted.append({
                    "url": request.url,
                    "method": request.method,
                    "headers": dict(request.headers),
                    "body": body,
                })

        _suppress_playwright_cleanup_noise()
        import io as _io
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(
                    extra_http_headers=extra_headers or {},
                    ignore_https_errors=True,
                )

                # Inject session cookies
                if cookies:
                    parsed = urlsplit(url)
                    domain = parsed.netloc.split(":")[0]
                    pw_cookies = [
                        {"name": k, "value": v, "domain": domain, "path": "/"}
                        for k, v in cookies.items()
                    ]
                    context.add_cookies(pw_cookies)

                page = context.new_page()
                page.on("request", _on_request)

                try:
                    page.goto(url, wait_until=wait, timeout=timeout)
                except Exception:
                    # networkidle can time out on slow apps — grab what we have
                    pass

                # Capture initial rendered HTML
                html = page.content()

                # Click interactive elements to reveal hidden forms, modals,
                # and SPA route transitions.  Collect HTML after each click.
                try:
                    html = self._explore_interactive(page, html, timeout=5000)
                except Exception as e:
                    logger.debug("Interactive exploration failed: %s", e)

                browser.close()

            # Force garbage collection while stderr is suppressed so that
            # any __del__ "Exception ignored" messages from asyncio
            # transports are silently discarded instead of printed.
            import gc
            _real_stderr = sys.stderr
            sys.stderr = _io.StringIO()
            try:
                gc.collect()
            finally:
                sys.stderr = _real_stderr

        except Exception as e:
            logger.warning("Headless fetch failed for %s: %s", url, e)
            return "", []

        api_points = self._points_from_requests(intercepted, url)
        logger.info(
            "[headless] %s — rendered %d chars, intercepted %d POST(s) → %d API point(s)",
            url, len(html), len(intercepted), len(api_points),
        )
        return html, api_points

    # ------------------------------------------------------------------
    # Interactive element exploration (SPA routes, modals, tabs)
    # ------------------------------------------------------------------

    @staticmethod
    def _explore_interactive(page, base_html: str, timeout: int = 5000) -> str:
        """
        Click buttons, nav links, and tab triggers to reveal hidden content.
        Returns the combined HTML from all discovered DOM states.
        """
        combined_html = base_html
        seen_states = {len(base_html)}

        # Selectors for interactive elements that might reveal forms
        interactive_selectors = [
            "button:not([type='submit'])",           # UI buttons (modals, accordions)
            "a[href^='#']",                          # Hash links (tabs, anchors)
            "a[href^='/#']",                         # SPA hash routes
            "[role='tab']",                          # ARIA tab triggers
            "[data-toggle='modal']",                 # Bootstrap modals
            "[data-bs-toggle='modal']",              # Bootstrap 5 modals
            "[data-toggle='collapse']",              # Bootstrap collapse
            "[data-bs-toggle='collapse']",           # Bootstrap 5 collapse
            ".accordion-header, .accordion-button",  # Accordion patterns
            "nav a",                                 # Navigation links (SPA routes)
        ]

        max_clicks = 15  # safety limit
        clicks = 0

        for selector in interactive_selectors:
            if clicks >= max_clicks:
                break
            try:
                elements = page.query_selector_all(selector)
            except Exception:
                continue

            for el in elements:
                if clicks >= max_clicks:
                    break
                try:
                    # Skip if element is not visible or not clickable
                    if not el.is_visible():
                        continue
                    el.click(timeout=timeout)
                    # Wait briefly for DOM to update
                    page.wait_for_timeout(500)
                    new_html = page.content()
                    state_key = len(new_html)
                    if state_key not in seen_states:
                        seen_states.add(state_key)
                        combined_html += "\n" + new_html
                        logger.debug("[headless] New DOM state after clicking %s", selector)
                    clicks += 1
                except Exception:
                    continue

        if clicks > 0:
            logger.info("[headless] Explored %d interactive elements, found %d DOM states",
                       clicks, len(seen_states))

        return combined_html

    # ------------------------------------------------------------------
    # Synthesise InjectionPoints from intercepted XHR/fetch requests
    # ------------------------------------------------------------------

    def _points_from_requests(
        self, intercepted: list[dict], base_url: str
    ) -> list[InjectionPoint]:
        """
        Turn intercepted POST/PUT requests into testable InjectionPoints.
        Handles both application/x-www-form-urlencoded and application/json bodies.
        """
        import json as _json
        from urllib.parse import parse_qs

        points: list[InjectionPoint] = []
        seen: set[tuple] = set()

        for req in intercepted:
            req_url = req["url"]
            method = req["method"]
            body = req.get("body", "") or ""
            content_type = req.get("headers", {}).get("content-type", "")

            # Resolve relative URLs
            if not req_url.startswith("http"):
                req_url = urljoin(base_url, req_url)

            # Skip asset/analytics requests
            if any(skip in req_url for skip in (
                "google-analytics", "facebook.com", "hotjar",
                ".css", ".js", ".png", ".jpg", ".woff",
            )):
                continue

            params: dict[str, str] = {}

            if "application/json" in content_type:
                try:
                    data = _json.loads(body)
                    if isinstance(data, dict):
                        params = {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float))}
                except Exception:
                    pass
                input_type = "json_field"
            else:
                # form-encoded or raw body
                try:
                    parsed = parse_qs(body, keep_blank_values=True)
                    params = {k: v[0] for k, v in parsed.items()}
                except Exception:
                    pass
                input_type = "form_field"

            for param in params:
                key = (req_url, method, param)
                if key in seen:
                    continue
                seen.add(key)
                other = {k: v for k, v in params.items() if k != param}
                points.append(InjectionPoint(
                    url=req_url,
                    method=method,
                    parameter=param,
                    input_type=input_type,
                    description=f"XHR/fetch {method} to {req_url} — param '{param}'",
                    default_form_values=other,
                    response_format="json" if "application/json" in content_type else "html",
                ))

        return points
