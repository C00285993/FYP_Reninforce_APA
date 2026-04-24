"""
WebGoat HTTP Client
Handles authentication and vulnerability interactions with OWASP WebGoat.

WebGoat is a deliberately insecure Java/Spring Boot application.  This client
targets lessons in the A1 Injection chapter.  The combined
``webgoat/goat-and-wolf`` Docker image runs WebGoat at port 8080 (mapped to
8081 on the host by default).

Endpoint notes
--------------
Modern WebGoat (8.x / 2023.x / 2025.x) uses Spring Security CSRF protection.
The CSRF token is embedded as ``_csrf`` in each lesson page.

SQL Injection lesson endpoint (WebGoat 2025):
  POST /WebGoat/SqlInjection/attack3
  Form fields: {account: <payload>}   ← String SQL Injection exercise
  Response: JSON {lessonCompleted, feedback, output (HTML)}

Command injection lesson endpoint (WebGoat 8.x):
  POST /WebGoat/attack?Screen=72&menu=900
  Form fields: {ipAddress: <payload>}
  (Actual Screen/menu IDs may vary; inspect /WebGoat/start.mvc at runtime)
"""

import requests
import logging
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Default WebGoat base path within the app server
_WEBGOAT_PREFIX = "/WebGoat"


class WebGoatClient:
    """HTTP client for interacting with OWASP WebGoat (Spring Boot / Java)."""

    # These endpoints may need adjusting for the specific WebGoat version used.
    _REGISTER_PATH = "/WebGoat/register.mvc"
    _LOGIN_PATH = "/WebGoat/login"
    _START_PATH = "/WebGoat/start.mvc"
    # SQL Injection lesson — String SQL Injection exercise (WebGoat 2025)
    # Endpoint: POST /WebGoat/SqlInjection/assignment5a
    # Parameters: account (fixed="Smith"), operator (fixed="="), injection (payload)
    # Query formed: SELECT * FROM user_data WHERE first_name='John' AND last_name='{account} {operator} {injection}'
    _SQLI_PATH = "/WebGoat/SqlInjection/assignment5a"
    # Command injection lesson — Screen/menu IDs for WebGoat 8.x
    # (Some builds use /WebGoat/attack; others expose a dedicated REST path)
    _CMDI_PATH = "/WebGoat/attack"
    _CMDI_SCREEN = "72"
    _CMDI_MENU = "900"
    # XSS lesson — Reflected XSS Shopping Cart exercise (WebGoat 2025)
    # Endpoint: GET /WebGoat/CrossSiteScripting/attack5a
    # Parameters: QTY1-4, field1 (card number), field2 (payload target)
    _XSS_PATH = "/WebGoat/CrossSiteScripting/attack5a"

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        username: str = "pentest",
        password: str = "Agent123",   # WebGoat 2025: password must be 6–10 chars
    ):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Pentest-Assistant/1.0 (Educational FYP)",
        })
        self._logged_in = False

    def _url(self, path: str) -> str:
        return urljoin(self.base_url, path)

    def _extract_csrf_token(self, soup: BeautifulSoup) -> str | None:
        """Extract Spring Security CSRF token from a page."""
        # Spring CSRF token in a hidden input named "_csrf"
        token_input = soup.find("input", {"name": "_csrf"})
        if token_input:
            return token_input.get("value")
        # Also check meta tag (some Spring setups)
        meta = soup.find("meta", {"name": "_csrf"})
        if meta:
            return meta.get("content")
        return None

    def wait_for_webgoat(self, timeout: int = 90, interval: int = 3) -> bool:
        """Wait for WebGoat to become available after Docker start."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.session.get(
                    self._url(self._START_PATH), timeout=5,
                    allow_redirects=True,
                )
                if resp.status_code in (200, 302):
                    logger.info(f"WebGoat is available at {self.base_url}")
                    return True
            except requests.ConnectionError:
                pass
            logger.debug(
                f"Waiting for WebGoat... ({int(time.time() - start)}s)"
            )
            time.sleep(interval)
        logger.error(f"WebGoat not available after {timeout}s")
        return False

    def register(self) -> bool:
        """Register a new user account (idempotent — handles 'already exists').

        WebGoat 2025 notes:
        - /WebGoat/register.mvc does NOT support GET; skip the prefetch.
        - No CSRF token required for the registration form.
        - Password must be 6–10 characters (WebGoat validation).
        - Success → HTTP 302 redirect to /WebGoat/attack?username=... + JSESSIONID cookie.
        - Username taken → HTTP 200 back to registration form (no cookie).
        """
        try:
            data = {
                "username": self.username,
                "password": self.password,
                "matchingPassword": self.password,
                "agree": "agree",
            }
            # Use allow_redirects=False so we reliably capture the JSESSIONID
            # from the Set-Cookie header on the 302 response before it is
            # overwritten by any further redirect.
            resp = self.session.post(
                self._url(self._REGISTER_PATH), data=data,
                allow_redirects=False, timeout=10,
            )
            if resp.status_code == 302:
                # Registration succeeded; follow redirect to land on the lesson page
                location = resp.headers.get("Location", self._url(self._START_PATH))
                if not location.startswith("http"):
                    location = self.base_url + location
                self.session.get(location, allow_redirects=True, timeout=10)
                logger.info(f"WebGoat: registered new user '{self.username}'")
                return True
            elif resp.status_code == 200:
                # 200 = validation error (username taken or password invalid)
                if "exists" in resp.text.lower() or "already" in resp.text.lower():
                    logger.info(
                        f"WebGoat: user '{self.username}' already exists — will log in"
                    )
                else:
                    logger.warning(
                        "WebGoat registration returned 200 — possible validation error "
                        "(check password length 6–10 chars)"
                    )
                return False  # Caller should fall back to login()
            logger.warning(f"WebGoat registration returned unexpected {resp.status_code}")
            return False
        except Exception:
            logger.error("WebGoat registration failed — connection error.")
            return False

    def login(self) -> bool:
        """Authenticate with WebGoat and establish a session.

        WebGoat 2025 login flow:
        - POST /WebGoat/login with username + password (no CSRF token needed).
        - Success → 302 to /WebGoat/welcome.mvc (then to start.mvc).
        - Failure → 302 to /WebGoat/login?error.
        """
        try:
            data = {
                "username": self.username,
                "password": self.password,
            }
            resp = self.session.post(
                self._url(self._LOGIN_PATH), data=data,
                allow_redirects=False, timeout=10,
            )
            location = resp.headers.get("Location", "")
            if resp.status_code == 302 and "error" not in location:
                # Follow redirect to complete login
                if not location.startswith("http"):
                    location = self.base_url + location
                self.session.get(location, allow_redirects=True, timeout=10)
                self._logged_in = True
                logger.info(f"WebGoat: logged in as '{self.username}'")
                return True
            logger.warning("WebGoat login failed — check credentials and that WebGoat is running.")
            return False
        except Exception:
            logger.error("WebGoat login failed — connection error or timeout.")
            return False

    def ensure_ready(self, security_level: str = None) -> bool:
        """Full init: wait → login (or register then login).

        security_level is accepted for API compatibility but is a no-op
        (WebGoat difficulty is fixed per exercise).

        Strategy:
          1. Wait for WebGoat to be reachable.
          2. Try to log in with existing credentials.
          3. If login fails (user doesn't exist yet), register then log in.
        """
        if not self.wait_for_webgoat():
            return False
        if self._logged_in:
            return True
        # Try login first (handles repeat calls with same account)
        if self.login():
            return True
        # Login failed → register (first run) then login
        self.register()
        return self.login()

    def set_security_level(self, level: str = "low") -> bool:
        """No-op: WebGoat doesn't have a global security level knob."""
        return True

    def submit_sqli(self, payload: str) -> tuple[str, int, float]:
        """
        Submit a SQL injection payload to WebGoat's String SQL Injection exercise.

        Targets: POST /WebGoat/SqlInjection/attack3
        Parameter: ``account`` (the user account name field)

        Args:
            payload: The SQL injection string (e.g. "Smith' OR '1'='1")

        Returns:
            Tuple of (response_json_text, status_code, elapsed_seconds)
        """
        # Fetch the lesson page to get a fresh CSRF token
        try:
            page_resp = self.session.get(
                self._url(self._SQLI_PATH), timeout=10
            )
            soup = BeautifulSoup(page_resp.text, "lxml")
            csrf = self._extract_csrf_token(soup)
        except Exception:
            csrf = None

        # The injection goes into the 'injection' field.
        # 'account' and 'operator' are fixed scaffolding that WebGoat uses to
        # construct: SELECT * FROM user_data WHERE ... last_name = '{account} {operator} {injection}'
        data = {
            "account": "Smith",
            "operator": "=",
            "injection": payload,
        }
        if csrf:
            data["_csrf"] = csrf

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest",
        }
        start = time.time()
        try:
            resp = self.session.post(
                self._url(self._SQLI_PATH),
                data=data,
                headers=headers,
                timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception:
            elapsed = time.time() - start
            logger.error("WebGoat submit_sqli failed — connection error.")
            return "", 0, elapsed

    def submit_cmdi(self, payload: str) -> tuple[str, int, float]:
        """
        Submit a command injection payload to WebGoat's OS Command Injection exercise.

        Fetches a fresh CSRF token before each POST.  Returns the response HTML,
        HTTP status code, and elapsed time.

        Args:
            payload: The injection string (e.g. "127.0.0.1 | id")

        Returns:
            Tuple of (response_html, status_code, response_time_seconds)
        """
        # Fetch the lesson page to get a fresh CSRF token
        params = {"Screen": self._CMDI_SCREEN, "menu": self._CMDI_MENU}
        try:
            page_resp = self.session.get(
                self._url(self._CMDI_PATH), params=params, timeout=10
            )
            soup = BeautifulSoup(page_resp.text, "lxml")
            csrf = self._extract_csrf_token(soup)
        except Exception:
            csrf = None

        data = {
            "Screen": self._CMDI_SCREEN,
            "menu": self._CMDI_MENU,
            "ipAddress": payload,
        }
        if csrf:
            data["_csrf"] = csrf

        start = time.time()
        try:
            resp = self.session.post(
                self._url(self._CMDI_PATH),
                data=data,
                params=params,
                timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception:
            elapsed = time.time() - start
            logger.error("WebGoat submit_cmdi failed — connection error.")
            return "", 0, elapsed

    def submit_xss(self, payload: str) -> tuple[str, int, float]:
        """
        Submit an XSS payload to WebGoat's Reflected XSS Shopping Cart exercise.

        Targets: GET /WebGoat/CrossSiteScripting/attack5a
        Parameter: ``field1`` (credit card number field — reflected unsanitised in receipt)

        WebGoat lesson 5a checks XSS_PATTERN against field1, NOT field2.
        Injecting into field2 returns "xss-reflected-5a-failed-wrong-field" (lessonCompleted=False).
        The XSS_PATTERN is: ``.*<script>(console\\.log|alert)\\(.*\\);?</script>.*``
        so payloads like ``<script>alert(1)</script>`` are required for lessonCompleted=True.

        Args:
            payload: The XSS string to inject (e.g. "<script>alert(1)</script>")

        Returns:
            Tuple of (response_json_text, status_code, elapsed_seconds)
        """
        params = {
            "QTY1": "1", "QTY2": "1", "QTY3": "1", "QTY4": "1",
            "field1": payload,
            "field2": "John Doe",
        }
        headers = {"X-Requested-With": "XMLHttpRequest"}
        start = time.time()
        try:
            resp = self.session.get(
                self._url(self._XSS_PATH), params=params,
                headers=headers, timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception:
            elapsed = time.time() - start
            logger.error("WebGoat submit_xss failed — connection error.")
            return "", 0, elapsed

    def get_page(self, vuln_type: str) -> tuple[str, int]:
        """Fetch a lesson page by vuln type."""
        if vuln_type == "sqli":
            path = self._SQLI_PATH
            params = {}
        elif vuln_type == "xss":
            path = self._XSS_PATH
            params = {}
        else:
            path = self._CMDI_PATH
            params = {"Screen": self._CMDI_SCREEN, "menu": self._CMDI_MENU}
        try:
            resp = self.session.get(
                self._url(path), params=params, timeout=10
            )
            return resp.text, resp.status_code
        except Exception:
            logger.error("WebGoat get_page failed — connection error.")
            return "", 0

    def reset_for_episode(self) -> bool:
        """Lightweight reset: verify session is still active."""
        try:
            resp = self.session.get(
                self._url(self._START_PATH), timeout=5, allow_redirects=True
            )
            if resp.status_code == 200:
                return True
            # Session expired — re-login
            return self.login()
        except Exception:
            logger.error("WebGoat episode reset failed — connection error.")
            return False

    # ------------------------------------------------------------------ #
    #  DVWAClient API compatibility shims
    # ------------------------------------------------------------------ #

    @property
    def PAGES(self) -> dict:
        """Compatibility shim so env-style code can call client.PAGES."""
        return {
            "sqli": self._SQLI_PATH,
            "cmdi": self._CMDI_PATH,
            "xss": self._XSS_PATH,
        }

    def _extract_csrf_token_from_html(self, html: str) -> str | None:
        """Helper: parse CSRF from raw HTML string."""
        soup = BeautifulSoup(html, "lxml")
        return self._extract_csrf_token(soup)
