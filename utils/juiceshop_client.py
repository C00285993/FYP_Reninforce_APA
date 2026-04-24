"""
Juice Shop HTTP Client
Handles authentication, REST API interactions, and challenge scoreboard
monitoring for OWASP Juice Shop.

Juice Shop is a modern Node.js/Angular SPA that uses REST APIs + JSON
(unlike DVWA's HTML forms + MySQL). SQLi targets its SQLite backend.
"""

import requests
import logging
import time

logger = logging.getLogger(__name__)


class JuiceShopClient:
    """HTTP client for interacting with OWASP Juice Shop."""

    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "AI-Pentest-Assistant/1.0 (Educational FYP)",
            "Content-Type": "application/json",
        })
        self._token = None
        self._solved_challenges: set[str] = set()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    # ------------------------------------------------------------------ #
    #  Startup & readiness
    # ------------------------------------------------------------------ #

    def wait_for_juiceshop(self, timeout: int = 90, interval: int = 3) -> bool:
        """Wait for Juice Shop to become available."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self.session.get(
                    self._url("/rest/admin/application-version"),
                    timeout=5,
                )
                if resp.status_code == 200:
                    version = resp.json().get("version", "unknown")
                    logger.info(f"Juice Shop is available (v{version})")
                    return True
            except requests.ConnectionError:
                pass
            logger.debug(
                f"Waiting for Juice Shop... ({int(time.time() - start)}s)"
            )
            time.sleep(interval)
        logger.error(f"Juice Shop not available after {timeout}s")
        return False

    def login(self, email: str = "admin@juice-sh.op",
              password: str = "admin123") -> bool:
        """Authenticate and store the Bearer token."""
        try:
            resp = self.session.post(
                self._url("/rest/user/login"),
                json={"email": email, "password": password},
            )
            if resp.status_code == 200:
                data = resp.json().get("authentication", {})
                self._token = data.get("token")
                self.session.headers["Authorization"] = f"Bearer {self._token}"
                # /profile (Pug-rendered) uses cookie-based auth, not Bearer header
                self.session.cookies.set("token", self._token)
                logger.info(f"Logged into Juice Shop as '{email}'")
                return True
            logger.warning(f"Juice Shop login failed ({resp.status_code})")
            return False
        except Exception:
            logger.error("Juice Shop login failed — connection error or timeout.")
            return False

    def ensure_ready(self, security_level: str = "low") -> bool:
        """
        Full initialization: wait -> login -> snapshot challenges.
        security_level is accepted for interface compatibility but Juice Shop
        does not have adjustable difficulty.
        """
        if not self.wait_for_juiceshop():
            return False
        if not self.login():
            return False
        self._snapshot_solved_challenges()
        return True

    # ------------------------------------------------------------------ #
    #  Challenge scoreboard (success detection)
    # ------------------------------------------------------------------ #

    def get_challenges(self) -> list[dict]:
        """Fetch all challenges from the API."""
        try:
            resp = self.session.get(self._url("/api/Challenges/"))
            if resp.status_code == 200:
                return resp.json().get("data", [])
        except Exception:
            logger.error("Failed to fetch Juice Shop challenges — connection error.")
        return []

    def _snapshot_solved_challenges(self):
        """Store currently-solved challenge names so we can detect new ones."""
        challenges = self.get_challenges()
        self._solved_challenges = {
            c["name"] for c in challenges if c.get("solved")
        }
        logger.debug(
            f"Challenge snapshot: {len(self._solved_challenges)} already solved"
        )

    def check_new_solves(self) -> list[str]:
        """Return challenge names solved since the last snapshot.

        Updates the internal snapshot so repeated calls within the same
        episode don't report the same challenge as 'new' again.
        """
        challenges = self.get_challenges()
        currently_solved = {
            c["name"] for c in challenges if c.get("solved")
        }
        new = currently_solved - self._solved_challenges
        if new:
            logger.info(f"New challenge(s) solved: {new}")
            self._solved_challenges = currently_solved  # advance snapshot
        return list(new)

    # ------------------------------------------------------------------ #
    #  SQLi attack surfaces
    # ------------------------------------------------------------------ #

    def search_products(self, query: str) -> tuple[str, int, float]:
        """
        Search products via the REST API (SQLi target).

        GET /rest/products/search?q=<query>

        Returns:
            (response_text, status_code, elapsed_seconds)
        """
        start = time.time()
        try:
            resp = self.session.get(
                self._url("/rest/products/search"),
                params={"q": query},
                timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception as e:
            elapsed = time.time() - start
            logger.error("search_products request failed — target may be unreachable.")
            return "", 0, elapsed

    def submit_login(self, email: str, password: str) -> tuple[str, int, float]:
        """
        Attempt login with given credentials (SQLi on login form).

        POST /rest/user/login  {email, password}

        Returns:
            (response_text, status_code, elapsed_seconds)
        """
        start = time.time()
        try:
            resp = self.session.post(
                self._url("/rest/user/login"),
                json={"email": email, "password": password},
                timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception as e:
            elapsed = time.time() - start
            logger.error("submit_login request failed — target may be unreachable.")
            return "", 0, elapsed

    # ------------------------------------------------------------------ #
    #  Episode lifecycle
    # ------------------------------------------------------------------ #

    def reset_for_episode(self) -> bool:
        """Lightweight per-episode reset: re-snapshot challenge state."""
        self._snapshot_solved_challenges()
        return True

    def set_security_level(self, level: str) -> bool:
        """No-op: Juice Shop has no adjustable security level."""
        return True

    # ------------------------------------------------------------------ #
    #  SSTI attack surface
    # ------------------------------------------------------------------ #

    def update_username(self, username: str) -> tuple[str, int, float]:
        """
        Update the logged-in user's username via POST /profile.

        Juice Shop renders the username server-side using Pug templates, making
        this endpoint the primary SSTI injection surface.

        Returns:
            (response_text, status_code, elapsed_seconds)
        """
        start = time.time()
        try:
            # Send as application/x-www-form-urlencoded — Form 3 on the profile page
            # has no enctype, so it submits URL-encoded by default.  We must clear the
            # session-level Content-Type: application/json so requests can set the correct
            # header automatically from data={}.
            # The token cookie (set at login) authenticates this Pug-rendered route.
            resp = self.session.post(
                self._url("/profile"),
                data={"username": username},
                headers={"Content-Type": None},
                timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception as e:
            elapsed = time.time() - start
            logger.error("update_username request failed — target may be unreachable.")
            return "", 0, elapsed

    def get_profile(self) -> tuple[str, int, float]:
        """
        Fetch the rendered profile page via GET /profile.

        Returns the server-rendered HTML where Pug has evaluated any #{...}
        expressions injected via update_username().

        Returns:
            (response_text, status_code, elapsed_seconds)
        """
        start = time.time()
        try:
            resp = self.session.get(
                self._url("/profile"),
                headers={"Accept": "text/html,application/xhtml+xml,*/*"},
                timeout=15,
            )
            elapsed = time.time() - start
            return resp.text, resp.status_code, elapsed
        except Exception as e:
            elapsed = time.time() - start
            logger.error("get_profile request failed — target may be unreachable.")
            return "", 0, elapsed

    def get_page(self, vuln_type: str) -> tuple[str, int]:
        """Fetch a page for compatibility with BasePentestEnv."""
        if vuln_type == "juiceshop_sqli":
            text, status, _ = self.search_products("")
            return text, status
        if vuln_type == "juiceshop_xss":
            text, status, _ = self.search_products("")
            return text, status
        if vuln_type == "juiceshop_ssti":
            text, status, _ = self.get_profile()
            return text, status
        return "", 200
