"""
DVWA HTTP Client
Handles authentication, session management, security level configuration,
and page-specific interactions with DVWA.

This is the "hands" of the agent — it performs actual HTTP requests
against the DVWA target running in Docker.
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlencode
import logging
import re
import time

logger = logging.getLogger(__name__)


class DVWAClient:
    """HTTP client for interacting with DVWA (Damn Vulnerable Web Application)."""

    # DVWA page paths for each vulnerability module
    PAGES = {
        "sqli": "/vulnerabilities/sqli/",
        "xss_reflected": "/vulnerabilities/xss_r/",
        "xss_stored": "/vulnerabilities/xss_s/",
        "cmdi": "/vulnerabilities/exec/",
        "ssti": "/vulnerabilities/ssti/",
        "setup": "/setup.php",
        "login": "/login.php",
        "security": "/security.php",
    }

    def __init__(self, base_url: str = "http://localhost:8080",
                 username: str = "admin", password: str = "password"):
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Pentest-Assistant/1.0 (Educational FYP)"
        })
        self._logged_in = False

    def _url(self, path: str) -> str:
        """Build full URL from relative path."""
        return urljoin(self.base_url, path)

    def wait_for_dvwa(self, timeout: int = 60, interval: int = 3) -> bool:
        """Wait for DVWA to become available (e.g., after Docker start)."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.session.get(self._url("/login.php"), timeout=5)
                if resp.status_code == 200:
                    logger.info("DVWA is available")
                    return True
            except requests.ConnectionError:
                pass
            logger.debug(f"Waiting for DVWA... ({int(time.time() - start)}s)")
            time.sleep(interval)
        logger.error(f"DVWA not available after {timeout}s")
        return False

    def setup_database(self) -> bool:
        """Initialize/reset DVWA database via the setup page."""
        try:
            # Get the setup page to extract CSRF token
            resp = self.session.get(self._url(self.PAGES["setup"]))
            soup = BeautifulSoup(resp.text, "lxml")
            token = self._extract_csrf_token(soup)

            # Submit the "Create / Reset Database" form
            data = {"create_db": "Create / Reset Database"}
            if token:
                data["user_token"] = token

            resp = self.session.post(self._url(self.PAGES["setup"]), data=data)
            if resp.status_code == 200:
                logger.info("DVWA database reset successfully")
                return True
        except Exception:
            logger.error("Failed to setup DVWA database — check target is running.")
        return False

    def login(self) -> bool:
        """Authenticate with DVWA and establish a session."""
        try:
            # Get login page for CSRF token
            resp = self.session.get(self._url(self.PAGES["login"]))
            soup = BeautifulSoup(resp.text, "lxml")
            token = self._extract_csrf_token(soup)

            # Submit login
            data = {
                "username": self.username,
                "password": self.password,
                "Login": "Login",
            }
            if token:
                data["user_token"] = token

            resp = self.session.post(self._url(self.PAGES["login"]),
                                     data=data, allow_redirects=True)

            # Check if login succeeded (redirects to index.php or contains welcome)
            if "index.php" in resp.url or "Welcome" in resp.text:
                self._logged_in = True
                logger.info(f"Logged into DVWA as '{self.username}'")
                return True
            else:
                logger.warning("DVWA login failed — check credentials")
                return False
        except Exception:
            logger.error("DVWA login failed — connection error or timeout.")
            return False

    def set_security_level(self, level: str = "low") -> bool:
        """Set DVWA security level: low, medium, high, impossible."""
        level = level.lower()
        assert level in ("low", "medium", "high", "impossible"), \
            f"Invalid security level: {level}"

        try:
            # Get the security page for CSRF token
            resp = self.session.get(self._url(self.PAGES["security"]))
            soup = BeautifulSoup(resp.text, "lxml")
            token = self._extract_csrf_token(soup)

            data = {"security": level, "seclev_submit": "Submit"}
            if token:
                data["user_token"] = token

            resp = self.session.post(self._url(self.PAGES["security"]), data=data)

            # Also set security cookie directly (some DVWA versions need this)
            self.session.cookies.set("security", level)

            logger.info(f"DVWA security level set to '{level}'")
            return True
        except Exception:
            logger.error("Failed to set DVWA security level — connection error.")
            return False

    def get_page(self, vuln_type: str) -> tuple[str, int]:
        """
        Fetch a DVWA vulnerability page.

        Args:
            vuln_type: One of 'sqli', 'xss_reflected', 'xss_stored'

        Returns:
            Tuple of (page_html, status_code)
        """
        path = self.PAGES.get(vuln_type)
        if not path:
            raise ValueError(f"Unknown vulnerability type: {vuln_type}")

        resp = self.session.get(self._url(path))
        return resp.text, resp.status_code

    def submit_sqli(self, payload: str) -> tuple[str, int, float]:
        """
        Submit a SQL injection payload to DVWA's SQLi page.

        Automatically detects the form method:
        - Low security: GET with id parameter on main page
        - Medium security: POST with id parameter on main page
        - High security: POST to session-input.php, then GET main page

        Args:
            payload: The injection string to submit as the 'id' parameter.

        Returns:
            Tuple of (response_html, status_code, response_time_seconds)
        """
        path = self.PAGES["sqli"]

        # Get the page first to extract CSRF token and detect form method
        resp = self.session.get(self._url(path))
        soup = BeautifulSoup(resp.text, "lxml")
        token = self._extract_csrf_token(soup)

        # Detect form method — check if main page has a form
        form = soup.find("form")

        if form is None:
            # High security: no form on main page, use session-input.php
            return self._submit_sqli_high(payload)

        method = form.get("method", "GET").upper()

        params = {"id": payload, "Submit": "Submit"}
        if token:
            params["user_token"] = token

        start = time.time()
        if method == "POST":
            resp = self.session.post(self._url(path), data=params)
        else:
            resp = self.session.get(self._url(path), params=params)
        elapsed = time.time() - start

        return resp.text, resp.status_code, elapsed

    def _submit_sqli_high(self, payload: str) -> tuple[str, int, float]:
        """
        Handle high-security SQLi where input is via session-input.php popup.

        1. POST the payload to session-input.php (sets session variable)
        2. GET the main SQLi page to see query results
        """
        session_input_path = "/vulnerabilities/sqli/session-input.php"
        main_path = self.PAGES["sqli"]

        data = {"id": payload, "Submit": "Submit"}

        start = time.time()
        # Step 1: Submit payload to session-input page
        self.session.post(self._url(session_input_path), data=data)
        # Step 2: Fetch main page which reads from session
        resp = self.session.get(self._url(main_path))
        elapsed = time.time() - start

        return resp.text, resp.status_code, elapsed

    def submit_cmdi(self, payload: str) -> tuple[str, int, float]:
        """
        Submit a command injection payload to DVWA's Command Execution page.

        DVWA's exec page uses a POST form with the 'ip' parameter and
        a CSRF token. All three security levels use the same form structure
        (the server-side filtering differs, not the form).

        Args:
            payload: The injection string to submit as the 'ip' parameter.

        Returns:
            Tuple of (response_html, status_code, response_time_seconds)
        """
        path = self.PAGES["cmdi"]

        # Get the page first to extract CSRF token
        resp = self.session.get(self._url(path))
        soup = BeautifulSoup(resp.text, "lxml")
        token = self._extract_csrf_token(soup)

        data = {"ip": payload, "Submit": "Submit"}
        if token:
            data["user_token"] = token

        start = time.time()
        resp = self.session.post(self._url(path), data=data)
        elapsed = time.time() - start

        return resp.text, resp.status_code, elapsed

    def submit_xss_reflected(self, payload: str) -> tuple[str, int, float]:
        """
        Submit an XSS payload to DVWA's Reflected XSS page.

        Args:
            payload: The XSS string to submit as the 'name' parameter.

        Returns:
            Tuple of (response_html, status_code, response_time_seconds)
        """
        path = self.PAGES["xss_reflected"]

        # Get CSRF token
        resp = self.session.get(self._url(path))
        soup = BeautifulSoup(resp.text, "lxml")
        token = self._extract_csrf_token(soup)

        # DVWA Reflected XSS uses GET with 'name' parameter
        params = {"name": payload}
        if token:
            params["user_token"] = token

        start = time.time()
        resp = self.session.get(self._url(path), params=params)
        elapsed = time.time() - start

        return resp.text, resp.status_code, elapsed

    def submit_xss_stored(self, name: str, message: str) -> tuple[str, int, float]:
        """
        Submit to DVWA's Stored XSS page (guestbook).

        Args:
            name: The 'txtName' field value.
            message: The 'mtxMessage' field value (main injection target).

        Returns:
            Tuple of (response_html, status_code, response_time_seconds)
        """
        path = self.PAGES["xss_stored"]

        # Get CSRF token
        resp = self.session.get(self._url(path))
        soup = BeautifulSoup(resp.text, "lxml")
        token = self._extract_csrf_token(soup)

        data = {
            "txtName": name,
            "mtxMessage": message,
            "btnSign": "Sign Guestbook",
        }
        if token:
            data["user_token"] = token

        start = time.time()
        resp = self.session.post(self._url(path), data=data)
        elapsed = time.time() - start

        return resp.text, resp.status_code, elapsed

    def reset_xss_stored_db(self) -> bool:
        """Clear the guestbook entries in stored XSS (reset between episodes)."""
        try:
            path = self.PAGES["xss_stored"]
            # DVWA stored XSS has a "Clear Guestbook" button
            # We can also directly reset by accessing the setup
            self.setup_database()
            return True
        except Exception:
            logger.error("Failed to reset stored XSS state — connection error.")
            return False

    def get_page_forms(self, html: str) -> list[dict]:
        """
        Extract form information from an HTML page.

        Returns:
            List of dicts with keys: action, method, inputs
            Each input has: name, type, value
        """
        soup = BeautifulSoup(html, "lxml")
        forms = []

        for form in soup.find_all("form"):
            form_info = {
                "action": form.get("action", ""),
                "method": form.get("method", "GET").upper(),
                "inputs": [],
            }

            for inp in form.find_all(["input", "textarea", "select"]):
                input_info = {
                    "name": inp.get("name", ""),
                    "type": inp.get("type", "text"),
                    "value": inp.get("value", ""),
                    "tag": inp.name,
                }
                if inp.name == "textarea":
                    input_info["type"] = "textarea"
                    input_info["value"] = inp.get_text()
                form_info["inputs"].append(input_info)

            forms.append(form_info)

        return forms

    @staticmethod
    def _extract_csrf_token(soup: BeautifulSoup) -> str | None:
        """Extract DVWA's CSRF token from a page."""
        token_input = soup.find("input", {"name": "user_token"})
        if token_input:
            return token_input.get("value")
        return None

    def ensure_ready(self, security_level: str = "low") -> bool:
        """
        Full initialization sequence: wait -> setup -> login -> set security.
        Call this once before training starts.
        """
        if not self.wait_for_dvwa():
            return False
        if not self.login():
            # Might need to setup DB first
            self.setup_database()
            time.sleep(2)
            if not self.login():
                return False
        if not self.set_security_level(security_level):
            return False
        return True

    def reset_for_episode(self) -> bool:
        """
        Lightweight reset between training episodes.
        Re-establishes session state without full DB reset.
        """
        try:
            # Verify session is still valid
            resp = self.session.get(self._url("/index.php"))
            if "login.php" in resp.url:
                # Session expired, re-login
                return self.login()
            return True
        except Exception:
            logger.error("Episode reset failed — connection error or session expired.")
            return False
