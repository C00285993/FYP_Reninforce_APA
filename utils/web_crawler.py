"""
LLM-Powered Web Crawler
Uses Claude to analyse a web page and discover injection points
(forms, URL parameters, JSON API fields) that can be tested for
SQL injection and XSS vulnerabilities.

Usage:
    crawler = LLMCrawler(api_key="sk-...")
    points  = crawler.crawl("http://target.com/search")
    # points -> list[InjectionPoint]
"""

import os
import re
import json
import logging
import time
from collections import deque
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs, urlsplit

import requests
from bs4 import BeautifulSoup

from utils.generic_http_client import InjectionPoint
from utils.headless_crawler import HeadlessCrawler

logger = logging.getLogger(__name__)

# Maximum HTML characters sent to the LLM (keep costs low)
_HTML_TRUNCATE = 8000

# Field names that suggest stored/persistent user content (stored XSS detection).
# Covers: user-generated content fields, auth/registration fields (stored in DB),
# and profile fields. Matched against form field names to set verify_url.
_PERSISTENCE_HINTS = re.compile(
    r"(?i)(comment|message|feedback|review|note|post|reply|"
    r"bio|about|description|title|name|profile|content|text|"
    r"subject|body|entry|article|story|memo|"
    # Auth / registration fields — stored in user table
    r"user|uid|login|email|mail|handle|nick|display|account|"
    r"first.?name|last.?name|full.?name|screen.?name|"
    # Address / contact fields often displayed on profiles
    r"address|city|phone|company|website|url|homepage|signature)"
)

# Extensions that are never HTML pages worth crawling
_SKIP_EXTENSIONS = frozenset({
    ".css", ".js", ".map", ".png", ".jpg", ".jpeg", ".gif", ".svg",
    ".ico", ".woff", ".woff2", ".ttf", ".eot", ".pdf", ".zip", ".gz",
    ".tar", ".xml", ".json", ".txt",
})

# URL path segments that indicate logout / session destruction
_LOGOUT_HINTS = frozenset({
    "logout", "logoff", "signout", "sign-out", "log-out",
    "disconnect", "deauthenticate",
})


class LLMCrawler:
    """
    Discovers injection points on a target URL using:
      1. Static HTML parsing  (fast, free — catches most forms/params)
      2. Claude LLM analysis  (catches tricky/JS-rendered inputs)
    """

    SYSTEM_PROMPT = """\
You are a web security analyst. Given the HTML of a web page and its URL,
identify all input points that could be tested for SQL injection or XSS.

Return ONLY a valid JSON array. Each element must have these fields:
  url         - full URL to send the request to (string)
  method      - "GET" or "POST" (string)
  parameter   - name of the parameter to inject into (string)
  input_type  - one of: "url_param", "form_field", "json_field" (string)
  description - short human-readable description (string)

Rules:
- Include URL query parameters, HTML form fields, and any visible API endpoints.
- If a form has action="", use the page URL.
- Exclude CSRF tokens, hidden fields that are not user-editable, file uploads.
- Resolve relative URLs against the page URL.
- Return [] if no injection points are found.
- Do NOT include any text outside the JSON array.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
        request_delay: float = 0.0,
        http_session: Optional[requests.Session] = None,
    ):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            try:
                import anthropic as _anthropic
                self._client = _anthropic.Anthropic(api_key=key)
                self._model = model
                self._llm_available = True
            except Exception:
                self._llm_available = False
        else:
            self._llm_available = False

        self._request_delay = request_delay
        self._headless = HeadlessCrawler()

        # Reuse an existing session (e.g. from authenticate()) so cookies and
        # server-side session state are preserved exactly.
        if http_session is not None:
            self._http = http_session
        else:
            self._http = requests.Session()
            self._http.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crawl(
        self,
        url: str,
        cookies: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
    ) -> list[InjectionPoint]:
        """
        Fetch *url* and return all discovered injection points.

        Args:
            url:           Target page URL.
            cookies:       Session cookies if auth is required.
            extra_headers: Additional HTTP headers (e.g. Authorization).

        Returns:
            List of InjectionPoint objects, deduplicated.
        """
        if cookies:
            # Clear existing cookies for the same names to avoid duplicates
            # (e.g. when reusing an auth session that already has PHPSESSID)
            for k in cookies:
                self._http.cookies.set(k, None)
            self._http.cookies.update(cookies)
        if extra_headers:
            self._http.headers.update(extra_headers)

        logger.info(f"Crawling: {url}")
        html, final_url = self._fetch_with_meta(url)
        if not html:
            logger.warning("Empty response from target — check URL and auth.")
            return []

        # Step 1: static parsing (always runs)
        static_points = self._parse_static(final_url, html)
        logger.info(f"  Static parser found {len(static_points)} injection point(s)")

        # Step 2: LLM analysis (enriches / catches missed points)
        llm_points: list[InjectionPoint] = []
        if self._llm_available:
            llm_points = self._analyse_with_llm(final_url, html)
            logger.info(f"  LLM analysis found {len(llm_points)} injection point(s)")
        else:
            logger.info("  Skipping LLM analysis (no API key). Set ANTHROPIC_API_KEY.")

        all_points = self._deduplicate(static_points + llm_points)

        # Step 3: Headless browser — always run alongside static to catch
        # JS-rendered forms, SPA routes, and AJAX-only inputs that the
        # static parser misses (not just when static finds nothing).
        headless_points: list[InjectionPoint] = []
        rendered_html = ""
        if self._headless.available:
            logger.info("  Running headless browser to discover JS-rendered inputs...")
            hcookies: dict = {}
            try:
                hcookies = dict(self._http.cookies)
            except Exception:
                try:
                    hcookies = {c.name: c.value for c in self._http.cookies}
                except Exception:
                    hcookies = {}
            rendered_html, api_points = self._headless.fetch(url, cookies=hcookies)
            if rendered_html:
                headless_points = self._parse_static(final_url, rendered_html)
                if self._llm_available:
                    headless_points += self._analyse_with_llm(final_url, rendered_html)
                headless_points += api_points
                headless_points = self._deduplicate(headless_points)
                logger.info(f"  Headless browser found {len(headless_points)} injection point(s)")

        all_points = self._deduplicate(all_points + headless_points)

        # Remove third-party/CDN injection points (e.g. YouTube API calls
        # intercepted by the headless browser from embedded iframes).
        all_points = self._filter_same_domain(all_points, url)

        logger.info(f"  Total unique injection points: {len(all_points)}")

        # Generate navigation hints for each injection point
        all_points = self._generate_nav_hints(url, (html or rendered_html) if not static_points else html, all_points)

        # Auto-generate header injection points for each unique endpoint
        all_points = self._add_header_points(all_points)

        for p in all_points:
            logger.info(f"    {p}")

        # Store the single-page URL for stored XSS sweep compatibility
        self.crawled_pages = [final_url]

        return all_points

    # ------------------------------------------------------------------
    # Domain scoping helper
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_same_domain(points: list, target_url: str) -> list:
        """
        Remove injection points whose URL belongs to a different domain
        than *target_url*.

        This prevents the headless browser's XHR interception from adding
        third-party API calls (e.g. YouTube analytics, Google fonts, CDNs)
        as injection points. Only the scheme+netloc (host:port) must match.
        """
        def _registrable(host: str) -> str:
            """Naive eTLD+1 (last two labels). Imperfect for .co.uk but fine for .com/.org/.net."""
            host = (host or "").lower().split(":")[0]
            parts = host.split(".")
            return ".".join(parts[-2:]) if len(parts) >= 2 else host

        target_root = _registrable(urlsplit(target_url).hostname or "")
        if not target_root:
            return points  # can't determine domain — pass everything through

        filtered = []
        removed = 0
        for pt in points:
            pt_root = _registrable(urlsplit(pt.url).hostname or "")
            if pt_root == target_root or not pt_root:
                filtered.append(pt)
            else:
                removed += 1
                logger.debug(
                    "Filtered off-domain injection point: %s (root=%s, target=%s)",
                    pt.url, pt_root, target_root,
                )
        if removed:
            logger.info(
                "  Removed %d off-domain injection point(s) (third-party APIs/CDNs)",
                removed,
            )
        return filtered

    # ------------------------------------------------------------------
    # Static HTML parsing
    # ------------------------------------------------------------------

    def _fetch(self, url: str) -> str:
        """Fetch *url*, returning HTML. Retries once on HTTP 429 with backoff."""
        return self._fetch_with_meta(url)[0]

    def _fetch_with_meta(self, url: str) -> tuple[str, str]:
        """
        Fetch *url* and return (html, final_url).
        final_url differs from url when the server issued a redirect.
        Handles HTTP 429 with exponential backoff (up to 3 retries).
        """
        if self._request_delay > 0:
            time.sleep(self._request_delay)

        backoff = 2.0
        for attempt in range(3):
            try:
                resp = self._http.get(url, timeout=15, allow_redirects=True)
                if resp.status_code == 429:
                    wait = backoff * (2 ** attempt)
                    logger.warning("HTTP 429 on %s — waiting %.1fs before retry", url, wait)
                    time.sleep(wait)
                    continue
                return resp.text, resp.url
            except Exception as e:
                if attempt == 2:
                    logger.error("Failed to fetch %s — %s", url, e)
                    return "", url
                time.sleep(backoff)
        return "", url

    def _parse_static(self, page_url: str, html: str) -> list[InjectionPoint]:
        """Extract injection points using BeautifulSoup without the LLM."""
        points: list[InjectionPoint] = []
        soup = BeautifulSoup(html, "lxml")

        # --- URL query parameters ---
        parsed = urlparse(page_url)
        if parsed.query:
            # Build lightweight page context for URL params (title + h1)
            title_tag = soup.find("title")
            h1_tag = soup.find("h1")
            page_ctx_parts = []
            if title_tag:
                page_ctx_parts.append(f"<title>{title_tag.get_text(strip=True)}</title>")
            if h1_tag:
                page_ctx_parts.append(f"<h1>{h1_tag.get_text(strip=True)}</h1>")
            page_ctx = " ".join(page_ctx_parts)[:300]

            for param in parse_qs(parsed.query).keys():
                clean_url = page_url.split("?")[0]
                points.append(InjectionPoint(
                    url=clean_url,
                    method="GET",
                    parameter=param,
                    input_type="url_param",
                    description=f"URL query parameter '{param}'",
                    context_html=page_ctx,
                ))

        # --- HTML forms ---
        for form in soup.find_all("form"):
            action = form.get("action", "") or ""
            method = form.get("method", "GET").upper()
            enctype = form.get("enctype", "").lower()
            form_url = urljoin(page_url, action) if action else page_url

            # Stored XSS pattern: form POSTs to a different URL than the page.
            # Set verify_url so we GET the display page after each POST to check
            # for stored reflection. Also set form_page_url so we can refresh
            # CSRF tokens before each POST.
            page_base = page_url.split("?")[0].rstrip("/")
            form_base = form_url.split("?")[0].rstrip("/")
            is_stored_xss_pattern = (method == "POST" and form_base != page_base)

            # Same-page stored XSS: POST form submits to the SAME page but
            # field names suggest persistent user content (comment, message,
            # feedback, name, bio, etc.). Re-GET the page after POST to check
            # if the payload was stored and rendered.
            is_same_page_stored = False
            if method == "POST" and not is_stored_xss_pattern:
                # Check injectable field names for persistence hints
                for inp in form.find_all(["input", "textarea", "select"]):
                    field_name = inp.get("name", "").strip()
                    if field_name and _PERSISTENCE_HINTS.search(field_name):
                        is_same_page_stored = True
                        break
                # Also check <textarea> presence (almost always stored content)
                if not is_same_page_stored and form.find("textarea"):
                    is_same_page_stored = True

            # Registration / signup / login form pattern: the form action URL
            # or the page URL hints at auth, and the form has user-identity
            # fields.  Payloads stored in user records / sessions render on
            # post-auth pages (profile, dashboard, index).
            is_auth_form = False
            if method == "POST":
                _form_action_lower = form_url.lower()
                _page_lower = page_url.lower()
                _auth_action_hints = (
                    "signup", "sign-up", "register", "registration",
                    "create_account", "create-account", "new_user",
                    "enroll", "join",
                    # Login forms too — username stored in session/DB
                    "login", "log-in", "signin", "sign-in", "auth",
                )
                if any(h in _form_action_lower or h in _page_lower
                       for h in _auth_action_hints):
                    # Confirm it has at least one user-identity field
                    _identity_pat = re.compile(
                        r"(?i)(user|uid|login|email|name|handle|nick|account)"
                    )
                    for inp in form.find_all(["input", "textarea"]):
                        if _identity_pat.search(inp.get("name", "")):
                            is_auth_form = True
                            break

            # For auth forms, verify_url should point to the post-auth
            # landing page (index.php), not the login page itself.
            if is_auth_form:
                from urllib.parse import urljoin as _urljoin_v
                _scope = page_url.rsplit("/", 1)[0] + "/"
                # Try index.php as the likely post-auth landing page
                verify = _urljoin_v(_scope, "index.php")
            elif is_stored_xss_pattern or is_same_page_stored:
                verify = page_url
            else:
                verify = ""
            # Also set form_page_url for same-page POST forms that have hidden
            # fields (e.g. CSRF tokens like user_token) so send_payload()
            # re-fetches them before every POST.
            has_hidden = bool(form.find_all("input", type="hidden"))
            form_page = page_url if (is_stored_xss_pattern or is_same_page_stored
                                     or is_auth_form
                                     or has_hidden) else ""

            # Collect all injectable field names and assign sensible defaults.
            # These are included in every POST so required sibling fields are
            # satisfied (e.g. injecting into 'comment' still sends name/email).
            injectable_fields: dict[str, str] = {}
            for inp in form.find_all(["input", "textarea", "select"]):
                n = inp.get("name", "").strip()
                if not n:
                    continue

                tag = inp.name  # "input", "textarea", or "select"
                t = inp.get("type", "text").lower() if tag == "input" else tag

                if t in ("hidden", "submit", "button", "reset",
                         "file", "image", "checkbox", "radio"):
                    continue
                if n.lower() in ("csrf_token", "user_token", "_token",
                                 "__requestverificationtoken"):
                    continue

                # Pick a safe default based on element type and field name
                if tag == "select":
                    # Use the first <option> value so the server accepts it
                    first_opt = inp.find("option")
                    default = (first_opt.get("value", first_opt.get_text(strip=True))
                               if first_opt else "1")
                elif t == "email" or "email" in n.lower():
                    default = "pentest@test.com"
                elif "url" in n.lower() or "website" in n.lower():
                    default = "http://test.test"
                elif t == "number" or "phone" in n.lower() or "tel" in n.lower():
                    default = "1"
                else:
                    default = "pentest_test"
                injectable_fields[n] = default

            # Collect submit button name=value pairs — many apps gate
            # form processing on isset($_POST['Submit']), so we must
            # include them.  Check <input type="submit"> and all
            # <button> elements (a <button> inside a form defaults to
            # type="submit" when no type attribute is set).
            submit_defaults: dict[str, str] = {}
            for inp in form.find_all("input", type="submit"):
                n = inp.get("name", "").strip()
                v = inp.get("value", "Submit")
                if n:
                    submit_defaults[n] = v
            for btn in form.find_all("button"):
                btn_type = (btn.get("type") or "submit").lower()
                if btn_type != "submit":
                    continue
                n = btn.get("name", "").strip()
                v = btn.get("value", "") or btn.get_text(strip=True) or "Submit"
                if n:
                    submit_defaults[n] = v

            for name, default_val in injectable_fields.items():
                # default_form_values = all other fields + submit button values
                other_defaults = {k: v for k, v in injectable_fields.items() if k != name}
                other_defaults.update(submit_defaults)

                points.append(InjectionPoint(
                    url=form_url,
                    method=method,
                    parameter=name,
                    input_type="form_field",
                    description=f"Form field '{name}' ({method} {form_url})",
                    context_html=str(form)[:500],
                    verify_url=verify,
                    form_page_url=form_page,
                    default_form_values=other_defaults,
                    enctype=enctype if enctype == "multipart/form-data" else "",
                ))

        # --- Orphaned inputs (outside any <form> tag) ---
        # e.g. navbar login fields, inline search bars submitted via JS
        all_inputs = soup.find_all("input")
        orphan_inputs = [
            i for i in all_inputs
            if not i.find_parent("form")
            and i.get("type", "text").lower() not in (
                "hidden", "submit", "button", "reset", "image",
                "checkbox", "radio", "file",
            )
            and i.get("name", "").strip()
        ]
        if orphan_inputs:
            # Group them all into a single synthetic POST to the current page
            injectable = {}
            for inp in orphan_inputs:
                name = inp.get("name", "").strip()
                itype = inp.get("type", "text").lower()
                if itype == "email" or "email" in name.lower():
                    injectable[name] = "pentest@test.com"
                else:
                    injectable[name] = inp.get("value", "pentest_test")

            for name in injectable:
                other = {k: v for k, v in injectable.items() if k != name}
                # Build context snippet from the orphaned input's parent container
                inp_tag = next((i for i in orphan_inputs if i.get("name") == name), None)
                ctx = ""
                if inp_tag:
                    parent = inp_tag.find_parent(["div", "section", "nav", "li", "td", "span"])
                    ctx = str(parent)[:500] if parent else str(inp_tag)[:300]
                points.append(InjectionPoint(
                    url=page_url,
                    method="POST",
                    parameter=name,
                    input_type="form_field",
                    description=f"Orphaned input '{name}' (outside <form>, POST to page)",
                    context_html=ctx,
                    default_form_values=other,
                ))

        return points

    # ------------------------------------------------------------------
    # LLM analysis
    # ------------------------------------------------------------------

    def _analyse_with_llm(self, page_url: str, html: str) -> list[InjectionPoint]:
        """Send HTML to Claude and parse the returned injection points."""
        truncated_html = html[:_HTML_TRUNCATE]

        user_message = (
            f"Page URL: {page_url}\n\n"
            f"HTML:\n```html\n{truncated_html}\n```"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw = response.content[0].text.strip()
        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="LLM crawl", once_flag_obj=self)
            self._llm_available = False
            return []

        return self._parse_llm_response(raw, page_url)

    def _parse_llm_response(
        self, raw: str, page_url: str
    ) -> list[InjectionPoint]:
        """Parse the JSON array returned by the LLM."""
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw.strip())
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON — falling back to static parser results.")
            return []

        if not isinstance(data, list):
            logger.warning("LLM response was not a JSON array.")
            return []

        points = []
        for item in data:
            try:
                # Resolve relative URLs
                url = item.get("url", page_url)
                if not url.startswith("http"):
                    url = urljoin(page_url, url)

                points.append(InjectionPoint(
                    url=url,
                    method=item.get("method", "GET").upper(),
                    parameter=item.get("parameter", ""),
                    input_type=item.get("input_type", "url_param"),
                    description=item.get("description", ""),
                ))
            except Exception:
                logger.warning("Skipping malformed item in LLM response.")

        return [p for p in points if p.parameter]

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(points: list[InjectionPoint]) -> list[InjectionPoint]:
        """Remove duplicate injection points (same URL + method + parameter)."""
        seen: set[tuple] = set()
        unique: list[InjectionPoint] = []
        for p in points:
            key = (p.url.rstrip("/"), p.method.upper(), p.parameter)
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # ------------------------------------------------------------------
    # Header injection point generation
    # ------------------------------------------------------------------

    _HEADER_PARAMS = ["X-Forwarded-For", "Referer", "X-Real-IP"]

    @staticmethod
    def _add_header_points(points: list[InjectionPoint]) -> list[InjectionPoint]:
        """
        For each unique endpoint URL, generate header-type injection points
        for common headers (X-Forwarded-For, Referer, X-Real-IP).
        These test for header injection / stored XSS via logged headers.
        """
        seen_urls: set[str] = set()
        header_points: list[InjectionPoint] = []
        for p in points:
            url_key = p.url.rstrip("/")
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)
            for hdr in LLMCrawler._HEADER_PARAMS:
                header_points.append(InjectionPoint(
                    url=p.url,
                    method=p.method,
                    parameter=hdr,
                    input_type="header",
                    nav_hint=f"HTTP header '{hdr}' on {p.url}",
                ))
        return points + header_points

    # ------------------------------------------------------------------
    # Navigation hints
    # ------------------------------------------------------------------

    def _generate_nav_hints(
        self, base_url: str, html: str, points: list[InjectionPoint]
    ) -> list[InjectionPoint]:
        """
        Assign step-by-step browser navigation hints to each injection point.

        Groups points by their page URL (form_page_url if set, else url), then
        either asks the LLM for a concise numbered guide or generates a static
        fallback from the URL path alone.
        """
        if not points:
            return points

        # Extract <title> from the crawled page HTML once
        try:
            page_title = (BeautifulSoup(html, "lxml").title.string or "").strip()
        except Exception:
            page_title = ""

        # Derive the application root URL (scheme + netloc only) for the hint generator
        _parsed_root = urlsplit(base_url)
        root_url = f"{_parsed_root.scheme}://{_parsed_root.netloc}"

        # Group injection points by their page URL
        page_to_points: dict[str, list[InjectionPoint]] = {}
        for p in points:
            page_url = p.form_page_url or p.url
            page_to_points.setdefault(page_url, []).append(p)

        # Build one hint per unique page URL
        for page_url, group in page_to_points.items():
            hint = self._hint_for_page(root_url, page_url, page_title)
            for p in group:
                p.nav_hint = hint

        return points

    def _hint_for_page(self, base_url: str, page_url: str, title: str) -> str:
        """Return a navigation hint string for a single page URL.
        Always uses static hints during crawl to save API calls.
        Use upgrade_nav_hints() post-scan for LLM hints on vulnerable points."""
        return self._static_nav_hint(base_url, page_url)

    def upgrade_nav_hints(self, points: list[InjectionPoint], base_url: str, html: str = ""):
        """Generate LLM nav hints for specific injection points (call post-scan for vulnerable ones only)."""
        if not self._llm_available or not points:
            return
        try:
            page_title = (BeautifulSoup(html, "lxml").title.string or "").strip() if html else ""
        except Exception:
            page_title = ""
        _parsed_root = urlsplit(base_url)
        root_url = f"{_parsed_root.scheme}://{_parsed_root.netloc}"
        seen_pages: dict[str, str] = {}
        for p in points:
            page_url = p.form_page_url or p.url
            if page_url not in seen_pages:
                seen_pages[page_url] = self._llm_nav_hint(root_url, page_url, page_title)
            p.nav_hint = seen_pages[page_url]

    def _llm_nav_hint(self, base_url: str, page_url: str, title: str) -> str:
        """Ask Claude Haiku for 2-4 numbered navigation steps."""
        prompt = (
            f"Given a web application rooted at {base_url}, write 2-4 concise numbered steps\n"
            f"telling a tester how to navigate to this specific page in a browser:\n\n"
            f"  Page URL : {page_url}\n"
            f"  Page title: {title}\n\n"
            f"Start from the home page. Be specific about what to click or type.\n"
            f"Return only the numbered steps, no other text.\n"
            f'Example: "1. Open the home page. 2. Click \'Products\' in the nav bar. 3. Select any product."'
        )
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )
            hint = response.content[0].text.strip()
            return hint
        except Exception as e:
            logger.debug(f"LLM nav hint failed: {e}")
            return self._static_nav_hint(base_url, page_url)

    @staticmethod
    def _static_nav_hint(base_url: str, page_url: str) -> str:
        """Generate a basic hint from the URL path segments."""
        try:
            parsed = urlsplit(page_url)
            path = parsed.path.rstrip("/")
            # Strip the base URL path prefix
            base_path = urlsplit(base_url).path.rstrip("/")
            if path.startswith(base_path):
                path = path[len(base_path):]
            segments = [s for s in path.split("/") if s]

            if not segments:
                return "1. Open the home page."

            if len(segments) == 1:
                label = segments[0].replace("-", " ").replace("_", " ").title()
                return f"1. Go to the home page. 2. Navigate to /{segments[0]}."

            # Two or more segments: describe each level
            steps = ["1. Go to the home page."]
            for idx, seg in enumerate(segments, start=2):
                label = seg.replace("-", " ").replace("_", " ").title()
                steps.append(f"{idx}. Navigate to the {label} section.")
            return " ".join(steps)
        except Exception:
            return f"1. Go to the home page. 2. Navigate to {page_url}."

    # ------------------------------------------------------------------
    # Multi-page spidering
    # ------------------------------------------------------------------

    def _extract_links(
        self, base_url: str, html: str, scope_prefix: str = ""
    ) -> list[str]:
        """
        Extract all internal same-origin <a href> links from *html*.
        Filters out static assets, logout URLs, and fragments.

        Args:
            base_url:     The URL of the page being parsed (used for resolving
                          relative links and same-origin check).
            html:         Raw HTML of the page.
            scope_prefix: If set, only links whose path starts with this prefix
                          are returned.  Example: "/project26/" keeps the crawler
                          confined to that sub-application and prevents it from
                          wandering to /phpmyadmin/, /other-app/, etc.
        """
        base_parsed = urlsplit(base_url)
        soup = BeautifulSoup(html, "lxml")
        links: list[str] = []

        # Collect anchor hrefs AND iframe/frame src URLs
        raw_hrefs: list[str] = []
        for a in soup.find_all("a", href=True):
            raw_hrefs.append(a["href"].strip())
        for tag in soup.find_all(["iframe", "frame"], src=True):
            raw_hrefs.append(tag["src"].strip())

        for href in raw_hrefs:
            if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            full_url = urljoin(base_url, href)
            parsed = urlsplit(full_url)

            # Same origin only
            if parsed.netloc != base_parsed.netloc:
                continue

            # Scope restriction: stay within the base path prefix
            # e.g. scope_prefix="/project26/" → drop anything outside that folder
            if scope_prefix and not parsed.path.startswith(scope_prefix):
                continue

            # Skip non-HTML file extensions
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
                continue

            # Skip logout-like paths (preserve the session)
            path_segments = set(path_lower.strip("/").split("/"))
            if path_segments & _LOGOUT_HINTS:
                continue

            # Normalise: drop fragment, keep query string
            clean = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean += f"?{parsed.query}"
            links.append(clean)

        # Preserve order, deduplicate
        return list(dict.fromkeys(links))

    def deep_crawl(
        self,
        start_url: str,
        cookies: Optional[dict] = None,
        extra_headers: Optional[dict] = None,
        max_pages: int = 10,
    ) -> list[InjectionPoint]:
        """
        Spider the application starting from *start_url*, following internal
        links (BFS) up to *max_pages* pages.

        The crawler is automatically scoped to the path prefix of *start_url*.
        For example, starting at ``http://localhost/project26/`` will only
        follow links under ``/project26/`` — it will never wander to
        ``/phpmyadmin/``, ``/dvwa/``, or any other sibling application on the
        same host.

        Useful for authenticated scans where the login landing page links out
        to many other protected pages. Every visited page is analysed for
        injection points just like the single-page :meth:`crawl` method.

        Args:
            start_url:     Entry point URL (e.g. http://localhost/project26/).
            cookies:       Session cookies from a prior authentication step.
            extra_headers: Additional HTTP headers.
            max_pages:     Maximum number of unique pages to visit (default 10).

        Returns:
            Deduplicated list of :class:`InjectionPoint` objects found across
            all visited pages.
        """
        if cookies:
            for k in cookies:
                self._http.cookies.set(k, None)
            self._http.cookies.update(cookies)
        if extra_headers:
            self._http.headers.update(extra_headers)

        # Derive the scope prefix from the start URL path.
        # e.g. "http://localhost/project26/login.php" → scope "/project26/"
        # e.g. "http://localhost/project26/"          → scope "/project26/"
        # e.g. "http://target.com/"                   → scope "/" (whole site)
        _start_path = urlsplit(start_url).path
        # Keep everything up to and including the last "/" in the start path.
        scope_prefix = _start_path[:_start_path.rfind("/") + 1] or "/"
        logger.info(f"[deep_crawl] Scope prefix: {scope_prefix!r}")

        visited: set[str] = set()   # normalised URLs already processed
        queue: deque[str] = deque([start_url])
        all_points: list[InjectionPoint] = []
        _login_redirect_count = 0

        while queue and len(visited) < max_pages:
            url = queue.popleft()
            norm = url.split("#")[0].rstrip("/") or url

            if norm in visited:
                continue
            visited.add(norm)

            logger.info(
                f"[deep_crawl] Visiting ({len(visited)}/{max_pages}): {url}"
            )
            html, final_url = self._fetch_with_meta(url)
            if not html:
                continue

            # --- Session expiry detection ---
            # If the server redirected us to a login page, our session expired.
            # Re-authenticate using stored credentials and retry once.
            # Compare base paths (without query strings) to detect actual redirects.
            _requested_base = urlsplit(url).path.rstrip("/")
            _final_base = urlsplit(final_url).path.rstrip("/")
            if _final_base != _requested_base:
                final_path = _final_base.lower()
                # Only match dedicated login pages — not any page with "auth" in its
                # name (e.g. auth1.php, auth2.php are application pages, not login).
                is_login_redirect = any(
                    hint in final_path
                    for hint in ("login", "signin", "sign-in", "session")
                )
                if is_login_redirect and _login_redirect_count == 0:
                    _login_redirect_count += 1
                    logger.warning(
                        "[deep_crawl] Session expired — redirected to %s. "
                        "Re-authenticating...", final_url
                    )
                    # Re-apply stored cookies to session (caller may have refreshed them)
                    if cookies:
                        self._http.cookies.update(cookies)
                    # Retry the same URL
                    visited.discard(norm)
                    queue.appendleft(url)
                    continue

            # Collect injection points from this page.
            # Parse both the final URL (after any redirect) and the original
            # queued URL — the original may have query params that the redirect
            # stripped (e.g. auth2.php?FileToView=x → auth2.php).
            page_points = self._parse_static(final_url, html)
            if url != final_url:
                page_points += self._parse_static(url, html)
            if self._llm_available:
                page_points += self._analyse_with_llm(final_url, html)

            # Headless browser per-page — always run to catch JS-rendered inputs
            if self._headless.available:
                try:
                    hcookies = dict(self._http.cookies)
                except Exception:
                    hcookies = {c.name: c.value for c in self._http.cookies}
                rendered_html, api_points = self._headless.fetch(url, cookies=hcookies)
                if rendered_html:
                    page_points += self._parse_static(final_url, rendered_html)
                    if self._llm_available:
                        page_points += self._analyse_with_llm(final_url, rendered_html)
                    page_points += api_points
                    html = rendered_html  # use rendered HTML for link extraction too

            page_points = self._deduplicate(page_points)
            # Strip off-domain points (third-party APIs, CDNs, embedded iframes)
            page_points = self._filter_same_domain(page_points, start_url)
            page_points = self._generate_nav_hints(url, html, page_points)
            all_points.extend(page_points)

            logger.info(
                f"[deep_crawl]   -> {len(page_points)} injection point(s) on this page"
            )

            # Enqueue new links (anchor tags + iframes), confined to scope prefix.
            # Also extract URL query parameters from links as injection points —
            # many apps pass testable params via <a href="page.php?id=1&action=view">
            for link in self._extract_links(url, html, scope_prefix=scope_prefix):
                link_norm = link.split("#")[0].rstrip("/") or link
                if link_norm not in visited:
                    queue.append(link)

                # Extract URL params from the link itself as injection points
                link_parsed = urlsplit(link)
                if link_parsed.query:
                    link_base = f"{link_parsed.scheme}://{link_parsed.netloc}{link_parsed.path}"
                    link_params = parse_qs(link_parsed.query)
                    for param_name, param_vals in link_params.items():
                        default_val = param_vals[0] if param_vals else ""
                        # Build default values for all OTHER params in this link
                        other_defaults = {
                            k: v[0] for k, v in link_params.items()
                            if k != param_name and v
                        }
                        all_points.append(InjectionPoint(
                            url=link_base,
                            method="GET",
                            parameter=param_name,
                            input_type="url_param",
                            description=f"URL param '{param_name}' from link on {url}",
                            default_form_values=other_defaults,
                        ))

        # --- Common page discovery ---
        # Probe for pages that exist but aren't linked from navigation.
        # Many apps have admin panels, dashboards, auth pages, etc. that
        # are accessible but not linked. Only check pages within scope.
        _COMMON_PAGES = [
            "admin.php", "admin", "dashboard.php", "dashboard",
            "profile.php", "profile", "settings.php", "settings",
            "config.php", "config", "panel.php", "panel",
            "auth.php", "auth", "auth1.php", "auth2.php",
            "search.php", "search", "contact.php", "contact",
            "reset.php", "reset", "forgot.php", "forgot",
            "upload.php", "upload", "api.php", "api",
            "test.php", "test", "debug.php", "debug",
            "user.php", "users.php", "account.php",
            "home.php", "main.php", "portal.php",
            "login.php", "register.php", "signup.php",
            "logout.php", "log.php", "logs.php",
        ]
        base_parsed = urlsplit(start_url)
        base_origin = f"{base_parsed.scheme}://{base_parsed.netloc}"
        probed = 0
        for page_name in _COMMON_PAGES:
            if len(visited) >= max_pages:
                break
            probe_url = f"{base_origin}{scope_prefix}{page_name}"
            probe_norm = probe_url.split("#")[0].rstrip("/")
            if probe_norm in visited:
                continue
            try:
                resp = self._http.get(probe_url, timeout=5, allow_redirects=False)
                if resp.status_code in (200, 301, 302) and resp.status_code != 404:
                    # Page exists — crawl it for injection points
                    visited.add(probe_norm)
                    probed += 1
                    probe_html = resp.text if resp.status_code == 200 else ""
                    if resp.status_code in (301, 302):
                        loc = resp.headers.get("Location", "")
                        if loc:
                            from urllib.parse import urljoin as _urljoin
                            probe_html, _ = self._fetch_with_meta(_urljoin(probe_url, loc))
                    if probe_html and len(probe_html) < 5_000_000:  # skip massive pages
                        page_points = self._parse_static(probe_url, probe_html)
                        if self._llm_available:
                            page_points += self._analyse_with_llm(probe_url, probe_html)
                        page_points = self._deduplicate(page_points)
                        all_points.extend(page_points)
                        if page_points:
                            logger.info(f"[deep_crawl] Discovered {probe_url} — {len(page_points)} point(s)")
            except Exception:
                continue
            if self._request_delay > 0:
                time.sleep(self._request_delay)
        if probed > 0:
            logger.info(f"[deep_crawl] Common page probe found {probed} additional page(s)")

        logger.info(
            f"[deep_crawl] Done: visited {len(visited)} page(s), "
            f"{len(all_points)} raw injection point(s) before final dedup"
        )
        # Store visited pages so the scan pipeline can pass them to the
        # GenericHttpClient for stored-XSS sweep (checking all known pages
        # after each payload submission).
        self.crawled_pages = sorted(visited)
        all_points = self._deduplicate(all_points)
        all_points = self._add_header_points(all_points)
        return all_points
