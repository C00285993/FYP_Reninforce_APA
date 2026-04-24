"""
Playwright-based Juice Shop Client
Extends JuiceShopClient with a real Chromium browser for detecting
client-side DOM XSS vulnerabilities that HTTP responses alone cannot reveal.

Problem:
    DOM XSS vulnerabilities execute inside the browser via JavaScript
    (e.g. Angular's innerHTML binding).  When we send a payload to
    /rest/products/search?q=<payload> the *server* response is plain JSON;
    script execution only happens when the Angular SPA renders the results.
    A pure-HTTP agent therefore cannot observe whether XSS fired.

Solution:
    PlaywrightJuiceShopClient keeps a persistent Chromium browser alongside
    the normal HTTP session.  For each XSS action it navigates the browser to
    the Juice Shop SPA search route and intercepts any alert() call triggered
    by the injected payload.
"""

import logging
import urllib.parse

from utils.juiceshop_client import JuiceShopClient

logger = logging.getLogger(__name__)


class PlaywrightJuiceShopClient(JuiceShopClient):
    """
    JuiceShopClient extended with a persistent Playwright browser.

    The HTTP session (inherited from JuiceShopClient) handles authentication,
    product-search API calls, and the challenge scoreboard.
    The Playwright browser adds actual DOM execution for XSS detection.

    Usage:
        client = PlaywrightJuiceShopClient(base_url="http://localhost:3000")
        client.ensure_ready()                       # HTTP login + browser launch
        result = client.navigate_xss_payload("<script>alert(1)</script>")
        print(result["alert_fired"])                # True if XSS executed
        client.close_browser()                      # Clean up
    """

    def __init__(self, base_url: str = "http://localhost:3000",
                 headless: bool = True):
        super().__init__(base_url=base_url)
        self._headless = headless
        self._playwright = None
        self._browser = None
        self._page = None
        # Per-navigation state
        self._alert_fired = False
        self._alert_message = ""

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def ensure_ready(self, security_level: str = "low") -> bool:
        """HTTP login + browser launch."""
        ok = super().ensure_ready(security_level)
        if ok:
            self._launch_browser()
        return ok

    def _launch_browser(self):
        """Start Playwright Chromium and authenticate via localStorage injection."""
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. "
                "Run: pip install playwright && playwright install chromium"
            )

        logger.info(
            f"Launching Playwright browser (headless={self._headless})..."
        )
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless
        )
        context = self._browser.new_context(
            # Silence certificate warnings on localhost
            ignore_https_errors=True,
        )
        self._page = context.new_page()

        # Intercept any JavaScript dialog (alert / confirm / prompt).
        # This is the primary DOM XSS detection signal.
        self._page.on("dialog", self._on_dialog)

        # Navigate to the SPA so Angular bootstraps, then inject the
        # JWT token the HTTP session already obtained.
        self._page.goto(self.base_url, wait_until="networkidle", timeout=30_000)
        if self._token:
            self._page.evaluate(
                f"window.localStorage.setItem('token', '{self._token}')"
            )
            logger.debug("JWT token injected into browser localStorage.")

        logger.info("Playwright browser ready.")

    def _on_dialog(self, dialog):
        """Handle browser dialogs — alert() means XSS fired."""
        self._alert_fired = True
        self._alert_message = dialog.message
        logger.info(
            f"[PLAYWRIGHT] XSS alert intercepted! "
            f"type={dialog.type} message='{dialog.message}'"
        )
        dialog.dismiss()

    # ------------------------------------------------------------------ #
    #  DOM XSS detection
    # ------------------------------------------------------------------ #

    def navigate_xss_payload(self, payload: str) -> dict:
        """
        Navigate the browser to Juice Shop's search SPA route with the payload.

        Juice Shop's Angular component renders search results via innerHTML,
        so DOM XSS payloads are executed by the browser at render time.

        Args:
            payload: The XSS payload string to inject as the search query.

        Returns:
            dict with:
                alert_fired   – True if an alert() / XSS dialog fired
                alert_message – The dialog message (e.g. "1" for alert(1))
                url           – The full URL navigated to
        """
        if self._page is None:
            logger.warning("Browser not initialised — call ensure_ready() first.")
            return {"alert_fired": False, "alert_message": "", "url": ""}

        self._alert_fired = False
        self._alert_message = ""

        encoded = urllib.parse.quote(payload, safe="")
        url = f"{self.base_url}/#/search?q={encoded}"

        try:
            # Navigate; allow partial timeouts — the dialog may interrupt load
            self._page.goto(url, wait_until="networkidle", timeout=8_000)
        except Exception as e:
            # Navigation timeout is normal when an alert fires mid-load
            logger.debug(f"Navigation completed with exception (may be OK): {e}")

        # Extra dwell time for Angular's change-detection cycle
        try:
            self._page.wait_for_timeout(1_200)
        except Exception:
            pass

        return {
            "alert_fired": self._alert_fired,
            "alert_message": self._alert_message,
            "url": url,
        }

    # ------------------------------------------------------------------ #
    #  Episode lifecycle
    # ------------------------------------------------------------------ #

    def reset_for_episode(self) -> bool:
        """HTTP snapshot reset + clear per-episode browser state."""
        self._alert_fired = False
        self._alert_message = ""
        return super().reset_for_episode()

    # ------------------------------------------------------------------ #
    #  Cleanup
    # ------------------------------------------------------------------ #

    def close_browser(self):
        """Release Playwright browser and API resources."""
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._playwright:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None
        logger.info("Playwright browser closed.")
