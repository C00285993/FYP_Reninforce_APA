"""
Generic HTTP Client
A target-agnostic HTTP client that can send payloads to any web application.
Replaces the DVWA-specific DVWAClient for use with the dynamic scanner.

Also defines the InjectionPoint dataclass used throughout the generic pipeline.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import re

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class InjectionPoint:
    """
    Represents a single testable input discovered on a target web application.
    Produced by the LLMCrawler and consumed by DynamicXSSEnv / DynamicSQLiEnv.
    """
    url: str                        # Full URL to send the request to
    method: str                     # "GET" or "POST"
    parameter: str                  # Parameter name to inject into
    input_type: str                 # "url_param", "form_field", "json_field"
    response_format: str = "html"   # "html" or "json" — detected from Content-Type
    description: str = ""           # Human-readable description of this point
    context_html: str = ""          # Surrounding HTML snippet for reference
    auth_required: bool = False     # Whether the endpoint requires authentication
    # Stored XSS support:
    verify_url: str = ""            # GET this URL after POST to check for stored reflection
    form_page_url: str = ""         # Page that hosts the form (used to refresh CSRF tokens)
    default_form_values: dict = field(default_factory=dict)  # Default values for other form fields (required to make POST succeed)
    enctype: str = ""           # Form encoding type ("multipart/form-data" if set)
    nav_hint: str = ""          # Step-by-step instructions to navigate to this page in a browser

    def __str__(self):
        return (
            f"InjectionPoint({self.method} {self.url} "
            f"param={self.parameter!r} type={self.input_type})"
        )


class GenericHttpClient:
    """
    Generic HTTP client compatible with BasePentestEnv's client contract.

    Sends payloads to any URL/parameter combination without any
    application-specific logic. Authentication, if needed, must be handled
    before the env is created (cookies can be passed in via the session).

    Interface contract (mirrors DVWAClient):
        ensure_ready(security_level) -> bool
        reset_for_episode()
        (payload sending is done via send_payload())
    """

    DEFAULT_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json,*/*",
    }

    # Patterns that indicate a session has expired and the user needs to re-login
    _SESSION_EXPIRED_PATTERNS = [
        re.compile(r'<form[^>]*action=["\'][^"\']*login', re.I),
        re.compile(r'(session|login)\s*(has\s*)?(expired|timed?\s*out)', re.I),
        re.compile(r'please\s+(log\s*in|sign\s*in)', re.I),
        re.compile(r'(unauthorized|not\s+authenticated)', re.I),
    ]

    def __init__(
        self,
        base_url: str,
        timeout: int = 15,
        cookies: Optional[dict] = None,
        headers: Optional[dict] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(self.DEFAULT_HEADERS)
        if headers:
            self.session.headers.update(headers)
        if cookies:
            self.session.cookies.update(cookies)

        self._baseline_body: str = ""
        self._baseline_time: float = 0.0    # Baseline response time for time-based SQLi comparison
        self._last_redirect_location: str = ""  # Location header from last 3xx response
        self._last_response_headers: dict = {}  # Headers from last response
        self._crawled_pages: list[str] = []     # All URLs discovered by crawler (for stored XSS checks)
        self._header_display_cache: dict[tuple[str, str], str] = {}  # (url, header) → display page

        # Re-authentication support
        self._auth_credentials: Optional[tuple[str, str, str]] = None  # (login_url, username, password)
        self._reauth_attempts: int = 0
        self._max_reauth_attempts: int = 3

    @property
    def last_response_headers(self) -> dict:
        """Headers from the most recent HTTP response."""
        return self._last_response_headers

    def set_auth_credentials(self, login_url: str, username: str, password: str):
        """Store credentials so the client can re-authenticate if the session expires."""
        self._auth_credentials = (login_url, username, password)

    def _is_session_expired(self, status: int, body: str) -> bool:
        """Detect whether a response indicates the session has expired."""
        if status in (401, 403):
            return True
        # Check for login redirect (302 to login page)
        if status in (301, 302, 303) and self._last_redirect_location:
            loc = self._last_redirect_location.lower()
            if any(kw in loc for kw in ("login", "signin", "auth", "session")):
                return True
        # Check body for login form / session expired messages
        if body and len(body) < 50000:  # only check reasonably-sized responses
            for pattern in self._SESSION_EXPIRED_PATTERNS:
                if pattern.search(body):
                    return True
        return False

    def _try_reauth(self) -> bool:
        """Attempt to re-authenticate using stored credentials. Returns True on success."""
        if not self._auth_credentials or self._reauth_attempts >= self._max_reauth_attempts:
            return False
        self._reauth_attempts += 1
        login_url, username, password = self._auth_credentials
        try:
            from utils.auth_helper import authenticate
            cookies, success, msg, _sess = authenticate(login_url, username, password, session=self.session)
            if success:
                self.session.cookies.update(cookies)
                logger.info("Session re-authenticated successfully (%s)", msg)
                self._reauth_attempts = 0  # reset counter on success
                return True
            logger.debug("Re-authentication failed: %s", msg)
        except Exception as e:
            logger.debug("Re-authentication error: %s", e)
        return False

    # ------------------------------------------------------------------
    # BasePentestEnv contract
    # ------------------------------------------------------------------

    def ensure_ready(self, security_level: str = "low") -> bool:
        """Check connectivity to the target. security_level is ignored."""
        try:
            resp = self.session.get(self.base_url, timeout=self.timeout)
            reachable = resp.status_code < 500
            if reachable:
                logger.info(f"Target reachable: {self.base_url} ({resp.status_code})")
            else:
                logger.warning(
                    f"Target returned {resp.status_code}: {self.base_url}"
                )
            return reachable
        except Exception:
            logger.error("Cannot reach target — connection failed or timed out.")
            return False

    def reset_for_episode(self):
        """Called between episodes. No state to reset for a generic client."""
        pass

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    _RATE_LIMIT_MAX_RETRIES = 3
    _RATE_LIMIT_BASE_DELAY = 2  # seconds; doubles on each retry

    def _handle_rate_limit(self, resp, attempt: int) -> bool:
        """If response is 429, sleep with exponential backoff. Returns True if should retry."""
        if resp.status_code != 429 or attempt >= self._RATE_LIMIT_MAX_RETRIES:
            return False
        retry_after = resp.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            delay = int(retry_after)
        else:
            delay = self._RATE_LIMIT_BASE_DELAY * (2 ** attempt)
        logger.info(f"Rate limited (429) — waiting {delay}s before retry {attempt + 1}")
        time.sleep(delay)
        return True

    def get(
        self, url: str, params: Optional[dict] = None
    ) -> tuple[str, int, float, str]:
        """
        Send GET request.
        Returns (body, status_code, response_time_secs, content_type).
        """
        t0 = time.time()
        try:
            for attempt in range(self._RATE_LIMIT_MAX_RETRIES + 1):
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if not self._handle_rate_limit(resp, attempt):
                    break
            elapsed = time.time() - t0
            self._last_response_headers = dict(resp.headers)
            ct = resp.headers.get("Content-Type", "text/html")
            return resp.text, resp.status_code, elapsed, ct
        except Exception:
            logger.warning("GET request failed — target may be unreachable.")
            return "", 0, time.time() - t0, "text/html"

    def post(
        self,
        url: str,
        data: Optional[dict] = None,
        json_body: Optional[dict] = None,
        allow_redirects: bool = True,
        multipart: bool = False,
    ) -> tuple[str, int, float, str]:
        """
        Send POST request with form data, JSON body, or multipart form data.
        Returns (body, status_code, response_time_secs, content_type).
        """
        t0 = time.time()
        try:
            for attempt in range(self._RATE_LIMIT_MAX_RETRIES + 1):
                if json_body is not None:
                    resp = self.session.post(
                        url, json=json_body, timeout=self.timeout,
                        allow_redirects=allow_redirects,
                    )
                elif multipart and data:
                    # Send as multipart/form-data — wrap each value in a tuple
                    files = {k: (None, str(v)) for k, v in data.items()}
                    resp = self.session.post(
                        url, files=files, timeout=self.timeout,
                        allow_redirects=allow_redirects,
                    )
                else:
                    resp = self.session.post(
                        url, data=data, timeout=self.timeout,
                        allow_redirects=allow_redirects,
                    )
                if not self._handle_rate_limit(resp, attempt):
                    break
            elapsed = time.time() - t0
            ct = resp.headers.get("Content-Type", "text/html")
            # Capture Location header for redirect-aware send_payload()
            self._last_redirect_location = resp.headers.get("Location", "")
            return resp.text, resp.status_code, elapsed, ct
        except Exception:
            logger.warning("POST request failed — target may be unreachable.")
            self._last_redirect_location = ""
            return "", 0, time.time() - t0, "text/html"

    def _extract_hidden_fields(self, html: str, target_action: str) -> dict:
        """
        Parse HTML and extract hidden input fields from the form whose action
        matches *target_action* (used to pick up fresh CSRF tokens).
        Also checks <meta> tags for CSRF tokens (common in Rails, Laravel, etc.).
        Returns a dict of {name: value} for all hidden inputs + meta CSRF tokens.
        """
        hidden = {}
        try:
            soup = BeautifulSoup(html, "lxml")

            # 1) <meta> CSRF tokens (e.g. <meta name="csrf-token" content="...">)
            csrf_meta_names = ["csrf-token", "csrf_token", "_csrf", "_token",
                               "csrfmiddlewaretoken", "X-CSRF-Token"]
            for meta_name in csrf_meta_names:
                meta = soup.find("meta", attrs={"name": meta_name})
                if meta and meta.get("content"):
                    hidden[meta_name] = meta["content"]
                    break

            # 2) Hidden inputs from the matching form
            for form in soup.find_all("form"):
                action = form.get("action", "")
                # Match if target_action ends with the form's action path
                if action and not target_action.endswith(action.lstrip("/")):
                    continue
                for inp in form.find_all("input", type="hidden"):
                    name = inp.get("name", "").strip()
                    value = inp.get("value", "")
                    if name:
                        hidden[name] = value
                return hidden
        except Exception as e:
            logger.debug(f"Hidden-field extraction failed: {e}")
        return hidden

    def _extract_csrf_from_headers(self, response_headers: dict) -> dict:
        """
        Extract CSRF tokens from response headers.
        Some frameworks send tokens via headers (e.g. X-CSRF-Token, X-XSRF-TOKEN).
        """
        csrf_headers = ["X-CSRF-Token", "X-XSRF-TOKEN", "CSRF-Token", "X-CSRFToken"]
        for header in csrf_headers:
            value = response_headers.get(header, "")
            if value:
                return {header: value}
        return {}

    # Maximum crawled pages to sweep on the *first* header probe (before
    # the display page is cached).  Keeps initial discovery fast.
    _HEADER_SWEEP_LIMIT = 10

    def send_header_payload(
        self, url: str, header_name: str, payload: str, method: str = "GET",
    ) -> tuple[str, int, float, str]:
        """
        Send a request with a payload injected into an HTTP header.
        Returns (body, status_code, response_time_secs, content_type).

        Header-injected payloads are almost always *stored* (logged IPs,
        Referer values, User-Agent strings) and rendered on a different page
        (admin panel, log viewer, dashboard).  After sending, this method
        checks for the payload on other pages.

        Performance: on the first hit the display page is cached so
        subsequent calls only need a single GET instead of a full sweep.
        """
        extra_headers = {header_name: payload}
        try:
            t0 = time.time()
            if method.upper() == "POST":
                resp = self.session.post(
                    url, headers=extra_headers, timeout=self.timeout,
                )
            else:
                resp = self.session.get(
                    url, headers=extra_headers, timeout=self.timeout,
                )
            elapsed = time.time() - t0
            ct = resp.headers.get("Content-Type", "")
            immediate_result = (resp.text, resp.status_code, elapsed, ct)
        except Exception as e:
            logger.debug("Header payload request failed: %s", e)
            return "", 0, 0.0, ""

        # Check immediate response for reflection (rare but possible)
        payload_lower = payload.lower()
        if payload_lower and payload_lower in resp.text.lower():
            return immediate_result

        # --- Stored header sweep ---
        # Fast path: if we already know which page displays this header's
        # value, just check that one page (single GET).
        cache_key = (url.rstrip("/"), header_name)
        cached_page = self._header_display_cache.get(cache_key)
        if cached_page:
            try:
                sweep_result = self.get(cached_page)
                if payload_lower and payload_lower in sweep_result[0].lower():
                    return sweep_result
            except Exception:
                pass
            # Cache miss (page no longer shows it) — fall through to sweep
            # but only if we haven't already swept many times
            return immediate_result

        # Slow path (first time only): sweep crawled pages + common admin
        # pages to find the display page.  Capped to _HEADER_SWEEP_LIMIT.
        if payload_lower:
            from urllib.parse import urljoin as _urljoin, urlsplit as _urlsplit
            sweep_pages = list(self._crawled_pages[:self._HEADER_SWEEP_LIMIT])
            # Add common admin/log pages where headers are often displayed
            base_parsed = _urlsplit(url)
            scope = f"{base_parsed.scheme}://{base_parsed.netloc}"
            # Derive app scope from URL path (e.g. /Project26/)
            path_parts = base_parsed.path.rstrip("/").rsplit("/", 1)
            app_scope = path_parts[0] + "/" if len(path_parts) > 1 else "/"
            for suffix in ("admin.php", "admin", "admin/logs", "logs.php",
                           "dashboard", "panel", "index.php", "index.html"):
                candidate = scope + app_scope + suffix
                if candidate not in sweep_pages:
                    sweep_pages.append(candidate)

            url_norm = url.rstrip("/")
            for page_url in sweep_pages[:self._HEADER_SWEEP_LIMIT + 8]:
                if page_url.rstrip("/") == url_norm:
                    continue
                try:
                    sweep_body, sweep_status, sweep_elapsed, sweep_ct = self.get(page_url)
                    if payload_lower in sweep_body.lower():
                        logger.info(
                            "Stored header payload (%s) found on %s (sent to %s)",
                            header_name, page_url, url,
                        )
                        self._header_display_cache[cache_key] = page_url
                        return sweep_body, sweep_status, sweep_elapsed, sweep_ct
                except Exception:
                    continue

        return immediate_result

    def send_payload(
        self, injection_point: InjectionPoint, payload: str
    ) -> tuple[str, int, float, str]:
        """
        Send a payload to the given injection point.
        Returns (body, status_code, response_time_secs, content_type).

        Automatically detects session expiry and re-authenticates once before
        retrying the request if credentials were stored via set_auth_credentials().

        For stored XSS (POST form with verify_url set):
          1. Optionally re-fetch form_page_url to grab a fresh CSRF token.
          2. POST the payload (plus any hidden fields like csrf/postId).
          3. GET verify_url and return THAT response — this is where the
             stored payload will be reflected.
        """
        # Header injection: payload goes into an HTTP header, not a form param
        if injection_point.input_type == "header":
            return self.send_header_payload(
                injection_point.url, injection_point.parameter, payload,
                method=injection_point.method,
            )

        if injection_point.method.upper() == "GET":
            # Include default values for sibling params (e.g. other GET params
            # from the same link/form) so the server receives a complete request.
            params = dict(injection_point.default_form_values) if injection_point.default_form_values else {}
            params[injection_point.parameter] = payload
            result = self.get(injection_point.url, params=params)
            # Session expiry check for GET — re-authenticate and retry once
            if self._is_session_expired(result[1], result[0]) and self._try_reauth():
                logger.info("Retrying GET payload after session re-authentication")
                result = self.get(injection_point.url, params=params)
            return result

        # --- POST path ---
        post_data: dict = {}

        # Start with default values for all other required form fields
        # (e.g. name="test", email="test@test.com" when injecting into comment)
        if injection_point.default_form_values:
            post_data.update(injection_point.default_form_values)

        # If form_page_url is set, fetch it to capture fresh hidden fields
        # (CSRF tokens, postId, etc.) before every POST.
        if injection_point.form_page_url:
            form_html, _, _, _ = self.get(injection_point.form_page_url)
            hidden = self._extract_hidden_fields(form_html, injection_point.url)
            post_data.update(hidden)
            # Also check response headers for CSRF tokens
            header_csrf = self._extract_csrf_from_headers(self._last_response_headers)
            if header_csrf:
                self.session.headers.update(header_csrf)
                logger.debug(f"Captured CSRF header: {list(header_csrf.keys())}")
            logger.debug(f"Captured hidden fields from form page: {list(hidden.keys())}")

        # Inject the payload into the target parameter (overrides any default)
        post_data[injection_point.parameter] = payload

        # --- POST without following redirects so we can inspect the
        # immediate response body.  Many server-side frameworks process
        # form input in one handler and redirect to a display page;
        # the reflected content may appear in any of these locations. ---
        is_multipart = injection_point.enctype == "multipart/form-data"
        if injection_point.input_type == "json_field":
            json_data = dict(injection_point.default_form_values) if injection_point.default_form_values else {}
            json_data[injection_point.parameter] = payload
            post_result = self.post(
                injection_point.url,
                json_body=json_data,
                allow_redirects=False,
            )
        else:
            post_result = self.post(
                injection_point.url, data=post_data,
                allow_redirects=False,
                multipart=is_multipart,
            )

        body, status, elapsed, ct = post_result

        # Session expiry check — re-authenticate and retry once if needed
        if self._is_session_expired(status, body) and self._try_reauth():
            logger.info("Retrying payload after session re-authentication")
            if injection_point.form_page_url:
                form_html, _, _, _ = self.get(injection_point.form_page_url)
                hidden = self._extract_hidden_fields(form_html, injection_point.url)
                post_data.update(hidden)
                post_data[injection_point.parameter] = payload
            if injection_point.input_type == "json_field":
                json_data = dict(injection_point.default_form_values) if injection_point.default_form_values else {}
                json_data[injection_point.parameter] = payload
                post_result = self.post(injection_point.url, json_body=json_data, allow_redirects=False)
            else:
                post_result = self.post(injection_point.url, data=post_data, allow_redirects=False)
            body, status, elapsed, ct = post_result

        payload_lower = payload.lower()

        def _has_reflection(text: str) -> bool:
            """Check if the payload appears in the response body."""
            return bool(payload_lower) and payload_lower in text.lower()

        # 1) Immediate POST response contains the payload → reflected
        if _has_reflection(body):
            return post_result

        # 2) Server returned a redirect (3xx) → follow with GET and check
        best_fallback = post_result
        if 300 <= status < 400:
            location = self._last_redirect_location
            if location:
                redirect_url = urljoin(injection_point.url, location)
                redirect_result = self.get(redirect_url)
                if _has_reflection(redirect_result[0]):
                    return redirect_result
                best_fallback = redirect_result
                # Post-redirect page discovery: extract links from the
                # redirect target and add them to the sweep list.  After
                # signup/login the server often redirects to a dashboard or
                # index page that links to additional authenticated pages
                # where stored payloads may render.
                self._discover_pages_from_html(redirect_result[0], redirect_url)

        # 3) Check the verify_url (stored XSS display page)
        if injection_point.verify_url:
            verify_result = self.get(injection_point.verify_url)
            if _has_reflection(verify_result[0]):
                return verify_result
            if best_fallback[0] == "":
                best_fallback = verify_result

        # 4) Stored payload sweep — the payload may have been persisted
        # and only rendered on a different page (e.g. log viewer,
        # profile page, dashboard).  Check all pages discovered by the
        # crawler, skipping URLs already tested above.
        if payload_lower and self._crawled_pages:
            already_checked = {injection_point.url, injection_point.verify_url}
            for page_url in self._crawled_pages:
                if page_url in already_checked:
                    continue
                try:
                    sweep_result = self.get(page_url)
                    if _has_reflection(sweep_result[0]):
                        logger.info(
                            "Stored payload found on %s (submitted to %s)",
                            page_url, injection_point.url,
                        )
                        return sweep_result
                except Exception:
                    continue

        # 5) Nothing reflected anywhere — return the richest response
        return best_fallback

    def capture_baseline(self, injection_point: InjectionPoint) -> str:
        """
        Capture a baseline response for comparison.
        For stored XSS (verify_url set): just GET the display page without posting.
        For GET params / reflected POST: send a benign probe and record the response.
        Also records baseline response time for time-based SQLi false-positive prevention.
        """
        t0 = time.time()
        if injection_point.verify_url:
            body, _, _, _ = self.get(injection_point.verify_url)
        else:
            body, _, _, _ = self.send_payload(injection_point, "baseline_test_input")
        self._baseline_time = time.time() - t0
        self._baseline_body = body
        logger.debug(f"Baseline response time: {self._baseline_time:.2f}s")
        return body

    def _discover_pages_from_html(self, html: str, base_url: str) -> None:
        """
        Extract same-origin links from *html* and add them to ``_crawled_pages``.

        Called after following a redirect (e.g. post-signup landing page) so
        that the stored-XSS sweep can check newly accessible pages such as
        dashboards, profiles, or index pages where stored payloads render.
        """
        if not html or not self._crawled_pages:
            return
        try:
            from urllib.parse import urlsplit
            base_parsed = urlsplit(base_url)
            soup = BeautifulSoup(html, "lxml")
            existing = set(self._crawled_pages)
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith(("#", "javascript:", "mailto:")):
                    continue
                full = urljoin(base_url, href)
                parsed = urlsplit(full)
                # Same origin only
                if parsed.netloc != base_parsed.netloc:
                    continue
                # Skip static assets
                path_lower = parsed.path.lower()
                if any(path_lower.endswith(ext) for ext in (
                    ".css", ".js", ".png", ".jpg", ".gif", ".svg", ".ico",
                    ".woff", ".pdf", ".zip", ".xml", ".json",
                )):
                    continue
                # Skip logout links
                segments = set(path_lower.strip("/").split("/"))
                if segments & {"logout", "logoff", "signout", "sign-out", "log-out"}:
                    continue
                clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                if parsed.query:
                    clean += f"?{parsed.query}"
                if clean not in existing:
                    self._crawled_pages.append(clean)
                    existing.add(clean)
            added = len(self._crawled_pages) - len(existing - set(self._crawled_pages))
        except Exception:
            pass  # non-fatal — sweep continues with existing pages

    def probe_post_auth_pages(self, base_url: str) -> None:
        """
        Probe common post-authentication landing pages and add any that
        respond with 200 to ``_crawled_pages``.

        Called after submitting to a registration/login form so that the
        stored-XSS sweep can find pages where the stored username/email
        is displayed (e.g. index.php, dashboard, profile, home).
        """
        if not self._crawled_pages:
            return
        from urllib.parse import urlsplit
        parsed = urlsplit(base_url)
        # Derive the app scope prefix (e.g. /project26/)
        path = parsed.path
        scope = path[:path.rfind("/") + 1] or "/"
        origin = f"{parsed.scheme}://{parsed.netloc}"
        existing = set(self._crawled_pages)

        _LANDING_PAGES = [
            "index.php", "index.html", "index", "",
            "home.php", "home", "main.php", "main",
            "dashboard.php", "dashboard", "portal.php", "portal",
            "profile.php", "profile", "account.php", "account",
            "admin.php", "admin", "panel.php", "panel",
            "user.php", "users.php", "members.php",
            "welcome.php", "welcome",
        ]
        for page_name in _LANDING_PAGES:
            probe_url = f"{origin}{scope}{page_name}"
            if probe_url in existing:
                continue
            try:
                resp = self.session.get(
                    probe_url, timeout=5, allow_redirects=False,
                )
                if resp.status_code == 200 and len(resp.text) > 100:
                    self._crawled_pages.append(probe_url)
                    existing.add(probe_url)
            except Exception:
                continue

    def fetch_page(self, url: str) -> str:
        """Simple page fetch (used by the crawler)."""
        body, _, _, _ = self.get(url)
        return body

    @staticmethod
    def detect_response_format(content_type: str, body: str) -> str:
        """
        Determine whether a response is HTML or JSON.
        Returns "json" or "html".
        """
        if "application/json" in content_type:
            return "json"
        try:
            json.loads(body)
            return "json"
        except (ValueError, TypeError):
            return "html"
