"""
Response Analyzer
Analyzes HTTP responses from DVWA to detect evidence of successful attacks,
SQL errors, XSS reflection, data leaks, and authentication bypass.

This module is critical for the reward function — it determines whether
the agent's actions produced meaningful results.
"""

import re
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Structured result of analyzing an HTTP response."""
    # SQL Injection indicators
    has_sql_error: bool = False
    sql_error_type: str = ""           # e.g., "mysql", "syntax", "generic"
    has_data_leak: bool = False
    leaked_data_count: int = 0         # Number of data rows extracted
    leaked_fields: list = field(default_factory=list)  # Field names found

    # XSS indicators
    payload_reflected: bool = False     # Payload appears in response
    script_tag_present: bool = False    # <script> in response body
    event_handler_present: bool = False # onerror=, onload=, etc.
    xss_context: str = ""              # "html", "attribute", "javascript"

    # General indicators
    auth_bypass: bool = False           # Got admin/authenticated content
    response_differs: bool = False      # Response differs from baseline
    response_length: int = 0
    status_code: int = 200
    response_time: float = 0.0
    error_page: bool = False           # 500/error page detected

    # Command Injection indicators
    has_command_execution: bool = False
    command_output_lines: int = 0
    file_operations_detected: bool = False
    shell_error_type: str = ""

    # SSTI indicators
    expression_evaluated: bool = False   # Arithmetic / string result found (e.g. "49")
    nodejs_info_leaked: bool = False     # Node.js version or env var present in response
    waf_blocked: bool = False            # WAF blocked the payload (400/403 / "Blocked")

    # Mitigation indicators (reduce false positives)
    script_blocked_by_csp: bool = False  # CSP header blocks inline scripts
    non_html_content_type: bool = False  # Response isn't text/html — browser won't parse
    response_data_direction: int = 0     # +1 = larger than baseline, -1 = smaller, 0 = same

    # Time-based command injection
    cmdi_time_based: bool = False        # Response time delta suggests blind CMDi

    # Computed severity score (0-100)
    @property
    def severity_score(self) -> int:
        score = 0
        if self.has_sql_error:
            score += 15
        if self.has_data_leak:
            score += 40 + min(self.leaked_data_count * 5, 30)
        if self.auth_bypass:
            score += 50
        if self.payload_reflected:
            score += 5
        if self.script_tag_present or self.event_handler_present:
            score += 40
        if self.has_command_execution:
            score += 60
        if self.cmdi_time_based:
            score += 50
        # Mitigations reduce the score — reflected but not exploitable
        if self.script_blocked_by_csp and not self.has_command_execution:
            score = max(score - 30, 5)
        if self.non_html_content_type:
            score = max(score - 35, 5)
        return min(score, 100)


class ResponseAnalyzer:
    """Analyzes HTTP responses for evidence of successful exploitation."""

    # SQL error patterns (MySQL-focused since DVWA uses MySQL)
    SQL_ERROR_PATTERNS = {
        "mysql": [
            r"you have an error in your sql syntax",
            r"warning.*mysql",
            r"unclosed quotation mark",
            r"mysql_fetch",
            r"mysql_num_rows",
            r"supplied argument is not a valid mysql",
            r"mysqli?_",
        ],
        "sqlite": [
            r"sqlite3?\.operationalerror",
            r"sqliteexception",
            r"sqlite error",
            r"near \".*\": syntax error",
            r"unrecognized token",
            r"sql\.js",
            r"sequelizedatabaseerror",
        ],
        "syntax": [
            r"sql syntax.*error",
            r"syntax error.*sql",
            r"unexpected end of sql",
            r"unterminated string",
        ],
        "generic": [
            r"sql error",
            r"database error",
            r"query failed",
            r"odbc.*error",
            r"db2.*error",
            r"oracle.*error",
        ],
    }

    # Patterns indicating data was successfully extracted
    DATA_LEAK_PATTERNS = [
        r"first_name",
        r"last_name",
        r"surname",
        r"user_?id",
        r"password",
        r"admin",
        r"username",
        r"email",
        r"credit.?card",
        r"phone",
        r"address",
        r"ssn",
        r"token",
        r"secret",
        r"api.?key",
    ]

    # Generic table-like data patterns (rows of structured output)
    GENERIC_DATA_LEAK_INDICATORS = [
        r"\b\d+\s*\|\s*\w+",                     # pipe-delimited rows: "1 | admin"
        r"\b\w+\s*:\s*\w+.*\n\w+\s*:\s*\w+",     # key: value repeated lines
        r"<tr[^>]*>.*?<td[^>]*>.*?</td>.*?</tr>", # HTML table rows with data
    ]

    # H2 database error patterns (WebGoat uses H2 in-memory database)
    H2_ERROR_PATTERNS = [
        r"org\.h2\.jdbc\.Jdbc",
        r"org\.h2\.",
        r"h2 database",
        r"syntax error in sql statement",
        r"column \"[^\"]+\" not found",
        r"table \"[^\"]+\" not found",
        r"unexpected token",
    ]

    # WAF / security appliance signatures
    WAF_SIGNATURES = [
        # Cloudflare
        (r"cloudflare", "Cloudflare"),
        (r"cf-ray", "Cloudflare"),
        (r"attention required.*cloudflare", "Cloudflare"),
        # ModSecurity
        (r"mod_security|modsecurity", "ModSecurity"),
        (r"not acceptable.*security", "ModSecurity"),
        # AWS WAF
        (r"aws.*waf", "AWS WAF"),
        (r"request blocked.*aws", "AWS WAF"),
        # Akamai
        (r"akamai.*ghost|reference.*akamai", "Akamai"),
        # Imperva / Incapsula
        (r"incapsula|imperva", "Imperva"),
        # Sucuri
        (r"sucuri.*firewall|sucuri.*waf", "Sucuri"),
        # Generic WAF patterns
        (r"web application firewall", "Generic WAF"),
        (r"request\s+(blocked|rejected|denied)", "Generic WAF"),
        (r"illegal\s+(request|activity)", "Generic WAF"),
        (r"access\s+denied.*security", "Generic WAF"),
        (r"your\s+request\s+has\s+been\s+blocked", "Generic WAF"),
        (r"suspicious\s+activity", "Generic WAF"),
    ]

    WAF_HEADER_SIGNATURES = [
        ("Server", r"cloudflare", "Cloudflare"),
        ("X-Sucuri-ID", r".", "Sucuri"),
        ("X-CDN", r"imperva|incapsula", "Imperva"),
        ("Server", r"akamaighost", "Akamai"),
    ]

    @classmethod
    def detect_waf(cls, body: str, status_code: int, headers: dict = None) -> tuple[bool, str]:
        """
        Detect if a response indicates a WAF blocked the request.
        Returns (is_blocked, waf_name). waf_name is "" if not blocked.
        """
        # Status code check — 403 with non-HTML body or specific patterns
        if status_code in (403, 406, 429, 501):
            body_lower = body.lower() if body else ""
            for pattern, name in cls.WAF_SIGNATURES:
                if re.search(pattern, body_lower):
                    return True, name
            # 403 with a very short body is often a WAF block
            if status_code == 403 and len(body) < 2000:
                return True, "Unknown WAF (403)"

        # Body pattern check regardless of status code
        if body:
            body_lower = body.lower()
            for pattern, name in cls.WAF_SIGNATURES:
                if re.search(pattern, body_lower):
                    return True, name

        # Header check
        if headers:
            for header_name, pattern, name in cls.WAF_HEADER_SIGNATURES:
                value = headers.get(header_name, "")
                if value and re.search(pattern, value, re.I):
                    return True, name

        return False, ""

    def analyze_webgoat_sqli_response(
        self,
        response_text: str,
        payload: str,
        status_code: int = 200,
        response_time: float = 0.0,
    ) -> AnalysisResult:
        """
        Analyze a JSON response from WebGoat's SQL Injection lesson.

        WebGoat returns JSON from lesson API endpoints:
          {"lessonCompleted": bool, "feedback": str, "output": str (HTML)}

        The ``output`` field contains an HTML table of database rows when the
        injection succeeds.  ``lessonCompleted`` signals the exercise is solved.

        Data leak is detected by counting user rows in the output HTML.
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(response_text),
        )

        text_lower = response_text.lower()

        # Parse outer JSON wrapper
        lesson_completed = False
        output_html = ""
        feedback_text = ""
        try:
            import json as _json
            data = _json.loads(response_text)
            lesson_completed = bool(data.get("lessonCompleted", False))
            output_html = data.get("output", "") or ""
            feedback_text = data.get("feedback", "") or ""
        except (ValueError, TypeError):
            # Fallback: treat full response as HTML
            output_html = response_text
            feedback_text = ""

        # Check for H2 SQL errors in any field
        for pattern in self.H2_ERROR_PATTERNS:
            if re.search(pattern, text_lower, re.I):
                result.has_sql_error = True
                result.sql_error_type = "h2"
                break

        # Also check generic SQL error patterns
        if not result.has_sql_error:
            for error_type, patterns in self.SQL_ERROR_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        result.has_sql_error = True
                        result.sql_error_type = error_type
                        break
                if result.has_sql_error:
                    break

        # assignment5a: success is signalled by lessonCompleted=True.
        # When injection succeeds, feedback contains all column names and
        # output contains the full query + "Explanation: This injection works..."
        # WebGoat's user_data table has ~6 rows (6 users).
        if lesson_completed:
            result.has_data_leak = True
            result.leaked_data_count = 6  # user_data table has 6 rows
            # Extract column names from feedback if present
            # feedback: "You have succeeded: <p>USERID, FIRST_NAME, LAST_NAME, ..."
            if "userid" in feedback_text.lower() or "first_name" in feedback_text.lower():
                # Parse column names from feedback HTML
                feedback_soup = BeautifulSoup(feedback_text, "lxml")
                cols_text = feedback_soup.get_text(strip=True)
                result.leaked_fields = [
                    c.strip().lower()
                    for c in re.split(r"[,\s]+", cols_text)
                    if c.strip() and c.strip()[0].isalpha()
                ][:6]
            result.auth_bypass = True
        elif output_html:
            # Try table-based detection as fallback (other lesson formats)
            soup = BeautifulSoup(output_html, "lxml")
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                data_rows = max(len(rows) - 1, 0)
                if data_rows > 0:
                    result.has_data_leak = True
                    result.leaked_data_count = data_rows

        # Response differs from baseline
        result.response_differs = self._differs_from_baseline(
            "webgoat_sqli", response_text
        )

        result.error_page = status_code >= 500 or status_code == 0

        # Time-based detection (delta vs baseline to avoid false positives)
        if self._is_time_based_hit("webgoat_sqli", payload, response_time):
            result.has_sql_error = True
            result.sql_error_type = "time_based"
            result.response_time = response_time

        return result

    # XSS event handler patterns
    XSS_EVENT_HANDLERS = [
        "onerror", "onload", "onmouseover", "onfocus", "onclick",
        "onsubmit", "onchange", "oninput", "onkeyup", "onkeydown",
    ]

    # Minimum delta (seconds) above baseline to flag time-based injection
    TIME_BASED_MIN_DELTA = 2.0

    def __init__(self):
        self._baseline_responses = {}  # Cache baseline responses per page

    def set_baseline(self, vuln_type: str, html: str, response_time: float = 0.0):
        """Store a baseline (benign) response for comparison."""
        soup = BeautifulSoup(html, "lxml")
        baseline_text = soup.get_text().lower()
        self._baseline_responses[vuln_type] = {
            "length": len(html),
            "text_hash": hash(self._extract_main_content(html)),
            "response_time": response_time,
            "html_lower": html.lower(),
            "full_text": baseline_text,
            "table_row_count": sum(
                max(len(t.find_all("tr")) - 1, 0) for t in soup.find_all("table")
            ),
        }

    def analyze_sqli_response(self, html: str, payload: str,
                               status_code: int = 200,
                               response_time: float = 0.0,
                               response_format: str = "html") -> AnalysisResult:
        """
        Analyze response from a SQL injection attempt.

        Args:
            html: Response HTML body
            payload: The SQLi payload that was submitted
            status_code: HTTP status code
            response_time: Time taken for the response

        Returns:
            AnalysisResult with SQLi-specific findings
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(html),
        )

        html_lower = html.lower()
        soup = BeautifulSoup(html, "lxml")

        # Check for SQL errors
        for error_type, patterns in self.SQL_ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, html_lower):
                    result.has_sql_error = True
                    result.sql_error_type = error_type
                    logger.debug(f"SQL error detected ({error_type}): {pattern}")
                    break
            if result.has_sql_error:
                break

        # Determine the baseline key — training envs use format-specific keys
        # (e.g. "webgoat_sqli"), dynamic envs always use "sqli".
        _baseline_key_map = {
            "html": "sqli", "webgoat": "webgoat_sqli",
            "juiceshop": "juiceshop_sqli",
        }
        baseline_key = _baseline_key_map.get(response_format, "sqli")
        # Fall back to "sqli" if the specific key isn't stored
        if baseline_key not in self._baseline_responses and "sqli" in self._baseline_responses:
            baseline_key = "sqli"

        # Check for data leaks — compare against baseline to avoid false positives
        result.has_data_leak, result.leaked_data_count, result.leaked_fields = \
            self._detect_data_leak(soup, html_lower, baseline_key=baseline_key)

        # Check for auth bypass — compare against baseline
        result.auth_bypass = self._detect_auth_bypass(html_lower, baseline_key=baseline_key)

        # Check if response differs from baseline
        result.response_differs = self._differs_from_baseline(baseline_key, html)

        # Boolean-blind SQLi: compute response direction vs baseline
        # +1 = larger (OR 1=1 returned more rows), -1 = smaller (AND 1=2 filtered out)
        result.response_data_direction = self._compute_response_direction(baseline_key, html)

        # Check for error page
        result.error_page = status_code >= 500 or status_code == 0

        # Time-based detection (delta vs baseline to avoid false positives)
        if self._is_time_based_hit("sqli", payload, response_time):
            result.has_sql_error = True
            result.sql_error_type = "time_based"
            result.response_time = response_time

        return result

    def analyze_xss_response(self, html: str, payload: str,
                              status_code: int = 200,
                              response_time: float = 0.0,
                              headers: dict | None = None) -> AnalysisResult:
        """
        Analyze response from an XSS attempt.

        Only flags script_tag_present / event_handler_present when the payload
        is in an **executable** HTML context — not merely reflected as text.
        This prevents false positives where the payload string appears in an
        error message but would not execute in a browser.

        Args:
            html: Response HTML body
            payload: The XSS payload that was submitted
            status_code: HTTP status code
            response_time: Time taken for the response
            headers: Response headers dict (for CSP / Content-Type checks)

        Returns:
            AnalysisResult with XSS-specific findings
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(html),
        )

        headers = headers or {}

        # ── Content-Type check ──────────────────────────────────────────
        # Non-HTML content types (text/plain, application/json, etc.)
        # will NOT be parsed as HTML by the browser — no script execution.
        content_type = headers.get("Content-Type", "text/html").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            result.non_html_content_type = True

        # ── CSP check ───────────────────────────────────────────────────
        # If Content-Security-Policy blocks inline scripts, injected
        # <script> tags and event handlers won't execute even if reflected.
        csp = headers.get("Content-Security-Policy", "")
        if csp:
            csp_lower = csp.lower()
            # CSP blocks inline if script-src exists but doesn't include unsafe-inline
            has_script_src = "script-src" in csp_lower or "default-src" in csp_lower
            allows_inline = "'unsafe-inline'" in csp_lower or "'unsafe-eval'" in csp_lower
            if has_script_src and not allows_inline:
                result.script_blocked_by_csp = True

        html_lower = html.lower()
        payload_lower = payload.lower()

        # Check if payload is reflected in response (basic reflection check)
        if payload_lower in html_lower:
            result.payload_reflected = True

        # ── URL-decoded reflection check ────────────────────────────────
        # Catches payloads reflected in URL-encoded form (server stores
        # %3Cscript%3E which the browser will decode to <script>), and
        # double-encoded payloads where the server decoded once.
        if not result.payload_reflected and payload_lower:
            from urllib.parse import unquote, quote
            # 1. Payload was URL-encoded by the user/proxy — decode it and
            #    check if the decoded version appears in the response
            decoded_variants = [
                unquote(payload),              # Single URL decode
                unquote(unquote(payload)),     # Double URL decode
            ]
            for variant in decoded_variants:
                if variant != payload and variant.lower() in html_lower:
                    result.payload_reflected = True
                    break
            # 2. Server reflected the payload URL-encoded — check if the
            #    URL-encoded form of our payload appears in the response
            #    (browser will decode it on render in href/src/action attrs)
            if not result.payload_reflected:
                url_encoded = quote(payload, safe='')
                if url_encoded.lower() in html_lower:
                    result.payload_reflected = True
                    result.xss_context = "encoded"
            # 3. Response itself may be URL-encoded — decode response and
            #    re-check for the raw payload
            if not result.payload_reflected:
                decoded_html = unquote(html_lower)
                if decoded_html != html_lower and payload_lower in decoded_html:
                    result.payload_reflected = True

        # Check for HTML-encoded reflection (e.g. &lt;script&gt;alert(1)&lt;/script&gt;)
        # This indicates the server received the payload but encoded it —
        # still useful signal for the agent (reflected but neutralised).
        if not result.payload_reflected and payload_lower:
            import html as html_module
            encoded_variants = [
                payload.replace("<", "&lt;").replace(">", "&gt;"),           # HTML entity
                payload.replace("<", "&#60;").replace(">", "&#62;"),         # Numeric entity
                payload.replace("<", "\\u003c").replace(">", "\\u003e"),     # Unicode escape (JS)
                html_module.escape(payload),                                  # Python html.escape
            ]
            for variant in encoded_variants:
                if variant.lower() in html_lower:
                    result.payload_reflected = True
                    result.xss_context = "encoded"
                    break

        # Check for unescaped script tags in an executable context
        result.script_tag_present = self._detect_injected_script(html, payload)

        # Check for event handlers on real HTML elements
        result.event_handler_present = self._detect_event_handlers(html, payload)

        # If payload is "reflected" but encoded → it's NOT executable.
        # Only set xss_context to "html"/"attribute" when we have proof of
        # execution (script tag or event handler in executable position).
        if result.xss_context == "encoded":
            # Already set above — reflected but neutralised, not a true positive
            pass
        elif result.script_tag_present or result.event_handler_present:
            result.xss_context = self._determine_xss_context(html, payload)
        elif result.payload_reflected:
            # Payload is reflected raw but no script/handler detected.
            # Check whether it's actually in an executable position by
            # verifying the payload created a real DOM element.
            ctx = self._determine_xss_context(html, payload)
            if ctx in ("javascript", "dom"):
                result.xss_context = ctx
            else:
                # Reflected in HTML body text but NOT as executable markup.
                # Mark as reflected only — NOT a confirmed XSS execution.
                result.xss_context = "reflected_only"

        # Check if response differs from baseline
        result.response_differs = self._differs_from_baseline("xss", html)

        result.error_page = status_code >= 500 or status_code == 0

        return result

    def analyze_juiceshop_sqli_response(
        self,
        response_text: str,
        payload: str,
        status_code: int = 200,
        response_time: float = 0.0,
    ) -> AnalysisResult:
        """
        Analyze a JSON response from Juice Shop for SQLi evidence.

        Juice Shop returns JSON from its REST API, not HTML. Success indicators:
        - SQL/SQLite error messages in the response body
        - Product data returned (search endpoint data leak)
        - Login bypass (200 on /rest/user/login with auth token)
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(response_text),
        )

        text_lower = response_text.lower()

        # Check for SQL errors in JSON error messages
        for error_type, patterns in self.SQL_ERROR_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    result.has_sql_error = True
                    result.sql_error_type = error_type
                    break
            if result.has_sql_error:
                break

        # Detect data leak from search endpoint
        # Juice Shop search returns {"status":"success","data":[...products...]}
        try:
            import json
            data = json.loads(response_text)
            if isinstance(data, dict):
                rows = data.get("data", [])
                if isinstance(rows, list) and len(rows) > 0:
                    result.has_data_leak = True
                    result.leaked_data_count = len(rows)
                    # Check what fields we got
                    if rows:
                        result.leaked_fields = list(rows[0].keys())[:5]

                # Login bypass: authentication token present
                auth = data.get("authentication", {})
                if auth.get("token"):
                    result.auth_bypass = True
        except (json.JSONDecodeError, ValueError):
            pass

        # Response differs from baseline
        result.response_differs = self._differs_from_baseline(
            "juiceshop_sqli", response_text
        )

        result.error_page = status_code >= 500 or status_code == 0

        # Time-based detection (delta vs baseline to avoid false positives)
        if self._is_time_based_hit("juiceshop_sqli", payload, response_time):
            result.has_sql_error = True
            result.sql_error_type = "time_based"
            result.response_time = response_time

        return result

    # XSS challenge names on the Juice Shop scoreboard
    JUICESHOP_XSS_CHALLENGES = {"DOM XSS", "Bonus Payload", "Reflected XSS"}

    def analyze_juiceshop_xss_response(
        self,
        response_text: str,
        payload: str,
        status_code: int = 200,
        response_time: float = 0.0,
        challenge_solves: list[str] | None = None,
    ) -> AnalysisResult:
        """
        Analyze a JSON response from Juice Shop for XSS evidence.

        Juice Shop returns JSON from its REST API. XSS execution happens
        client-side in Angular, so we use heuristics for intermediate
        rewards and the challenge scoreboard as the definitive success signal.

        Args:
            response_text: Raw response body (typically JSON)
            payload: The XSS payload that was submitted
            status_code: HTTP status code
            response_time: Time taken for the response
            challenge_solves: List of newly solved challenge names (from scoreboard)

        Returns:
            AnalysisResult with XSS-specific findings
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(response_text),
        )

        text_lower = response_text.lower()
        payload_lower = payload.lower()

        # Check if payload is reflected in JSON response
        if payload_lower in text_lower:
            result.payload_reflected = True

        # Check for unescaped HTML tags in JSON string values
        # If the payload's angle brackets survive JSON encoding, the
        # server is not sanitising input — Angular may render it via innerHTML
        if "<" in payload and "<" in response_text:
            # Look for our tag patterns in the response (not HTML-entity-escaped)
            import json as _json
            try:
                data = _json.loads(response_text)
                json_str = _json.dumps(data)
                if payload_lower in json_str.lower():
                    result.payload_reflected = True
                    # Check for script/event handler patterns surviving in the value
                    if re.search(r"<script", json_str, re.I):
                        result.script_tag_present = True
                    if any(h + "=" in json_str.lower() for h in self.XSS_EVENT_HANDLERS):
                        result.event_handler_present = True
                    # iframe/object with javascript: are DOM XSS vectors
                    if re.search(r"<iframe[^>]+javascript:", json_str, re.I):
                        result.script_tag_present = True
                        result.xss_context = "html"
            except (ValueError, TypeError):
                pass

        # Definitive success: challenge solved on the scoreboard
        if challenge_solves:
            xss_solves = [
                c for c in challenge_solves
                if c in self.JUICESHOP_XSS_CHALLENGES
            ]
            if xss_solves:
                result.script_tag_present = True
                result.xss_context = "html"

        # Determine XSS context from payload characteristics
        if not result.xss_context and result.payload_reflected:
            if any(h in payload_lower for h in self.XSS_EVENT_HANDLERS):
                result.xss_context = "attribute"
            elif "<" in payload:
                result.xss_context = "html"

        # Response differs from baseline
        result.response_differs = self._differs_from_baseline(
            "juiceshop_xss", response_text
        )

        result.error_page = status_code >= 500 or status_code == 0

        return result

    def analyze_webgoat_xss_response(
        self,
        response_text: str,
        payload: str,
        status_code: int = 200,
        response_time: float = 0.0,
    ) -> AnalysisResult:
        """
        Analyze a JSON response from WebGoat's Reflected XSS lesson.

        WebGoat's XSS Shopping Cart exercise returns JSON:
          {"lessonCompleted": bool, "feedback": str, "output": str (HTML)}

        ``lessonCompleted`` is the definitive success signal — WebGoat's
        server-side detector confirmed that the XSS pattern was present.
        The ``output`` HTML may also contain the reflected payload for
        intermediate reward signals.

        Args:
            response_text: Raw JSON response body
            payload: The XSS payload submitted as ``field2``
            status_code: HTTP status code
            response_time: Time taken for the response

        Returns:
            AnalysisResult with XSS-specific findings set.
        """
        result = AnalysisResult()
        result.status_code = status_code
        result.response_time = response_time
        result.response_length = len(response_text)
        result.error_page = status_code >= 500 or status_code == 0

        import json as _json
        try:
            data = _json.loads(response_text)
            lesson_completed = bool(data.get("lessonCompleted", False))
            output_html = data.get("output", "") or ""
            feedback = data.get("feedback", "") or ""
        except (ValueError, TypeError):
            lesson_completed = False
            output_html = response_text
            feedback = ""

        # Definitive success: WebGoat confirmed the XSS pattern was detected
        if lesson_completed:
            result.script_tag_present = True
            result.event_handler_present = True
            result.auth_bypass = True

        # Check the output HTML for payload reflection and unescaped tags
        inner_html = output_html or feedback
        if inner_html:
            result.payload_reflected = payload.lower() in inner_html.lower()
            if not lesson_completed:
                result.script_tag_present = self._detect_injected_script(inner_html, payload)
                result.event_handler_present = self._detect_event_handlers(inner_html, payload)
                result.xss_context = self._determine_xss_context(inner_html, payload)

        result.response_differs = self._differs_from_baseline("webgoat_xss", response_text)
        return result

    # Patterns indicating command output appeared in the response
    CMD_OUTPUT_PATTERNS = [
        r"uid=\d+",              # id / whoami output
        r"gid=\d+",
        r"root:\S*:0:0:",        # /etc/passwd root entry
        r":/bin/bash",           # passwd shell entries
        r":/bin/sh",
        r"total \d+",            # ls -la header line
        r"drwx",                 # directory listing entry
        r"-rwx",                 # file listing entry
        r"Linux \S+ \d+\.\d+",  # uname -a output
        r"GNU/Linux",
        r"x86_64|aarch64|armv",  # architecture in uname
    ]

    # File operation indicators
    FILE_OP_PATTERNS = [
        r"/etc/passwd",
        r"/etc/shadow",
        r"/home/\w+",
        r"/var/www",
        r"/usr/bin",
        r"/tmp/",
    ]

    # Shell error patterns
    SHELL_ERROR_PATTERNS = {
        "bash": [r"bash: .*: command not found", r"bash: .*: No such file"],
        "sh":   [r"sh: .*: not found", r"sh: \d+: "],
        "ping": [r"ping: unknown host", r"ping: cannot resolve"],
    }

    def analyze_cmdi_response(
        self,
        html: str,
        payload: str,
        status_code: int = 200,
        response_time: float = 0.0,
        response_format: str = "html",
    ) -> AnalysisResult:
        """
        Analyze response from a command injection attempt.

        Works on any website — searches <pre>, <code>, output divs, and full
        page text for command output patterns, not just DVWA-specific elements.

        Args:
            html: Response HTML body
            payload: The CMDi payload that was submitted
            status_code: HTTP status code
            response_time: Time taken for the response

        Returns:
            AnalysisResult with CMDi-specific findings
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(html),
        )

        soup = BeautifulSoup(html, "lxml")

        # Collect text from common output containers (generic, not DVWA-only)
        output_text = ""
        for elem in soup.find_all(["pre", "code", "samp", "tt"]):
            output_text += elem.get_text() + "\n"

        # Also check divs that look like output/result areas
        for div in soup.find_all("div"):
            cls = " ".join(div.get("class", []))
            div_id = div.get("id", "")
            if re.search(r"result|output|response|content|body_padded|command",
                         cls + " " + div_id, re.I):
                output_text += div.get_text() + "\n"

        # Full page text as fallback
        full_text = soup.get_text() + "\n" + output_text

        # Count output lines in structured blocks
        if output_text.strip():
            result.command_output_lines = len(
                [l for l in output_text.strip().splitlines() if l.strip()]
            )

        # Get baseline text for comparison — only flag NEW patterns
        baseline = self._baseline_responses.get("cmdi", {})
        baseline_text = baseline.get("full_text", "")

        # Check for command execution output patterns — only NEW ones
        for pattern in self.CMD_OUTPUT_PATTERNS:
            in_response = bool(re.search(pattern, full_text))
            in_baseline = bool(re.search(pattern, baseline_text)) if baseline_text else False
            if in_response and not in_baseline:
                result.has_command_execution = True
                break

        # Check for file operation evidence — only NEW ones
        for pattern in self.FILE_OP_PATTERNS:
            in_response = bool(re.search(pattern, full_text))
            in_baseline = bool(re.search(pattern, baseline_text)) if baseline_text else False
            if in_response and not in_baseline:
                result.file_operations_detected = True
                break

        # Check for shell errors (evidence of partial injection even if cmd failed)
        # Shell errors are always NEW — they wouldn't appear on a normal page
        for shell_name, patterns in self.SHELL_ERROR_PATTERNS.items():
            for p in patterns:
                if re.search(p, full_text, re.I):
                    result.shell_error_type = shell_name
                    break
            if result.shell_error_type:
                break

        # Response differs from baseline
        result.response_differs = self._differs_from_baseline("cmdi", html)

        result.error_page = status_code >= 500 or status_code == 0

        # Time-based blind CMDi: no visible output but response delayed
        # (e.g. "; sleep 5" executed but output suppressed)
        if not result.has_command_execution:
            if self._is_cmdi_time_based_hit("cmdi", payload, response_time):
                result.cmdi_time_based = True
                result.has_command_execution = True
                result.shell_error_type = "time_based_blind"

        return result

    # SSTI challenge names as returned by /api/Challenges/ .name field
    # Actual API name is "SSTi"; keep the longer alias for future compatibility.
    SSTI_CHALLENGE_NAMES = {"SSTi", "Server-side Template Injection"}

    # Patterns that are very distinctive of Node.js runtime info
    # (would not normally appear in a profile page)
    NODEJS_INFO_PATTERNS = [
        r"v\d+\.\d+\.\d+",              # Node.js version string e.g. v18.17.0
        r"/usr/local/(?:bin|sbin|lib)",  # Typical PATH fragment
        r":/usr/bin:/bin",               # PATH value (colon-separated)
        r"\blinux\b",                    # process.platform
        r"\bwin32\b",                    # process.platform Windows
        r"\bdarwin\b",                   # process.platform macOS
        r"\bx64\b|\bx86_64\b|\baarch64\b",  # process.arch
        r"production|development|test",   # NODE_ENV (common values)
    ]

    # PHP/server info patterns that indicate server-side info leak via Twig
    PHP_INFO_PATTERNS = [
        r"PHP/\d+\.\d+",               # PHP version in Server header or phpinfo
        r"phpinfo",                    # phpinfo() output
        r"Apache/\d+\.\d+",            # Apache version
        r"nginx/\d+\.\d+",             # Nginx version
        r"Symfony",                    # Symfony framework
        r"/var/www/html",              # Typical PHP document root
        r"DOCUMENT_ROOT",              # Server var name in Twig dump
        r"SERVER_SOFTWARE",            # Server software env var
        r"PHP_SELF",                   # PHP_SELF env var
    ]

    # Patterns that confirm template expression evaluation
    # Each is a distinctive result of a specific probe payload
    SSTI_EVAL_PATTERNS = [
        r"\b49\b",                 # 7*7 (Pug/Twig)
        r"\b1024\b",               # Math.pow(2,10) / 2**10 (Twig)
        r"\b1787569\b",            # 1337*1337
        r"\b42\b",                 # 2*21
        r"\b81\b",                 # 9*9
        r"\b256\b",                # 2**8 (Twig)
        r"\b25\b",                 # 100/4 (Twig)
        r"\b27\b",                 # 3**3 (Twig)
        r"\bHELLO\b",              # 'hello'.toUpperCase() / 'hello'|upper
        r"\babcabcabc\b",          # 'abc'.repeat(3)
        r"\bjuiceshop\b",          # 'juice'.concat('shop')
        r"\b1-2-3\b",              # [1,2,3].join('-')
        r"\bnumber\b",             # typeof 1337
        r"\bboolean\b",            # typeof true
        r"\bfunction\b",           # typeof Function
        r"3\.14159",               # Math.PI
        r"\bOLLEH\b",              # 'hello'|reverse (Twig)
        r"\bHello World\b",        # 'hello world'|title (Twig)
        r"abcdef",                 # 'abc'~'def' concat (Twig)
        r"\b1,2,3,4,5\b",          # range(1,5)|join(',') (Twig)
    ]

    # Twig sandbox / security error patterns
    TWIG_WAF_PATTERNS = [
        r"Twig.+Sandbox.+Security",
        r"SecurityNotAllowedMethod",
        r"SecurityNotAllowedFunction",
        r"SecurityNotAllowedFilter",
        r"SecurityNotAllowedProperty",
        r"twig.*sandbox.*error",
        r"not allowed.*method.*twig",
    ]

    def analyze_ssti_response(
        self,
        html: str,
        payload: str,
        status_code: int = 200,
        response_time: float = 0.0,
        challenge_solves: list[str] | None = None,
        response_format: str = "pug",
    ) -> AnalysisResult:
        """
        Analyze an SSTI response from Juice Shop (Pug) or DVWA (Twig).

        Supports two template engine contexts:
          - "pug"  (default): Juice Shop /profile page; uses #{...} syntax
          - "twig": DVWA /vulnerabilities/ssti/ page; uses {{...}} syntax

        Detection layers (in order of confidence):
          1. WAF block: status 400/403 or engine-specific error → waf_blocked=True
          2. Expression eval: distinctive arithmetic / string result in rendered page
          3. Server info leak: Node.js (pug) or PHP/server vars (twig) in page
          4. RCE: uid= / /etc/passwd / command output in rendered page
          5. Scoreboard (pug only): sstiChallenge solved → definitive success

        Args:
            html: Response HTML body
            payload: The SSTI payload submitted
            status_code: HTTP status code
            response_time: Time taken for the response
            challenge_solves: Newly solved challenge names (Juice Shop only)
            response_format: "pug" for Juice Shop, "twig" for DVWA

        Returns:
            AnalysisResult with SSTI-specific findings set.
        """
        result = AnalysisResult(
            status_code=status_code,
            response_time=response_time,
            response_length=len(html),
        )

        # Use the same baseline key that the env's _capture_baselines() sets
        if response_format == "twig":
            baseline_key = "dvwa_ssti"
        elif response_format == "pug":
            baseline_key = "juiceshop_ssti"
        else:
            baseline_key = "ssti"  # generic target
        html_lower = html.lower()

        # --- WAF check -------------------------------------------------------
        waf_triggered = False
        if status_code in (400, 403):
            waf_triggered = True
        elif "blocked" in html_lower or "illegal activity" in html_lower:
            waf_triggered = True
        # Check Twig WAF patterns for twig OR generic (might be PHP)
        if not waf_triggered and response_format in ("twig", "html"):
            for pattern in self.TWIG_WAF_PATTERNS:
                if re.search(pattern, html, re.I):
                    waf_triggered = True
                    break

        if waf_triggered:
            result.waf_blocked = True
            result.response_differs = self._differs_from_baseline(baseline_key, html)
            result.error_page = status_code >= 500 or status_code == 0
            return result

        # --- Scoreboard check (Juice Shop / Pug only) ------------------------
        if response_format == "pug" and challenge_solves:
            ssti_solves = [
                c for c in challenge_solves
                if (
                    c in self.SSTI_CHALLENGE_NAMES
                    or "ssti" in c.lower()
                    or "template injection" in c.lower()
                )
            ]
            if ssti_solves:
                result.has_command_execution = True
                result.expression_evaluated = True
                logger.info(f"SSTI challenge solved via scoreboard: {ssti_solves}")

        # --- Extract visible page text for heuristic analysis ----------------
        soup = BeautifulSoup(html, "lxml")
        page_text = soup.get_text(separator=" ", strip=True)

        # --- Expression evaluated heuristic ----------------------------------
        if not result.expression_evaluated:
            # Raw payload reflection means the template engine did NOT evaluate it
            # For generic targets, check both Pug and Twig syntax markers
            if response_format == "pug":
                payload_reflected_raw = "#{" in page_text
            elif response_format == "twig":
                payload_reflected_raw = "{{" in page_text
            else:
                payload_reflected_raw = "#{" in page_text or "{{" in page_text

            if not payload_reflected_raw:
                for pattern in self.SSTI_EVAL_PATTERNS:
                    if re.search(pattern, page_text, re.I):
                        result.expression_evaluated = True
                        break

        # --- Server info leak heuristic --------------------------------------
        if not result.nodejs_info_leaked:
            # For generic targets, check both Node.js and PHP patterns
            if response_format == "pug":
                info_patterns = self.NODEJS_INFO_PATTERNS
            elif response_format == "twig":
                info_patterns = self.PHP_INFO_PATTERNS
            else:
                info_patterns = self.NODEJS_INFO_PATTERNS + self.PHP_INFO_PATTERNS
            for pattern in info_patterns:
                if re.search(pattern, page_text):
                    if self._differs_from_baseline(baseline_key, html):
                        result.nodejs_info_leaked = True
                        break

        # --- Command execution heuristic -------------------------------------
        if not result.has_command_execution:
            for pattern in self.CMD_OUTPUT_PATTERNS:
                if re.search(pattern, page_text):
                    result.has_command_execution = True
                    break

        for pattern in self.FILE_OP_PATTERNS:
            if re.search(pattern, page_text):
                result.file_operations_detected = True
                break

        # --- Response comparison ---------------------------------------------
        result.response_differs = self._differs_from_baseline(baseline_key, html)
        result.error_page = status_code >= 500 or status_code == 0

        return result

    def _detect_data_leak(self, soup: BeautifulSoup,
                          html_lower: str,
                          baseline_key: str = "sqli") -> tuple[bool, int, list]:
        """
        Detect if SQL injection caused data to be leaked in the response.
        Works on any website — searches full page text, tables, <pre> blocks,
        and common output containers (not just DVWA).

        Compares against the stored baseline to avoid false positives from
        patterns that already exist in the normal (uninjected) page.
        """
        leaked = False
        count = 0
        fields = []

        # Get baseline for comparison (if available)
        baseline = self._baseline_responses.get(baseline_key, {})
        baseline_text = baseline.get("full_text", "")
        baseline_table_rows = baseline.get("table_row_count", 0)

        # 1) DVWA-style labelled output (First name: / Surname:)
        first_name_matches = re.findall(
            r"first\s*name[:\s]*([^\n<]+)", html_lower
        )
        surname_matches = re.findall(
            r"surname[:\s]*([^\n<]+)", html_lower
        )
        # Only count if these are NEW (not in baseline)
        baseline_fn = re.findall(r"first\s*name[:\s]*([^\n<]+)", baseline.get("html_lower", ""))
        baseline_sn = re.findall(r"surname[:\s]*([^\n<]+)", baseline.get("html_lower", ""))
        new_fn = len(first_name_matches) - len(baseline_fn)
        new_sn = len(surname_matches) - len(baseline_sn)
        if new_fn > 0 and new_sn > 0:
            leaked = True
            count = max(new_fn, new_sn)
            fields = ["first_name", "surname"]

        # 2) Password hash patterns (32-char hex = MD5)
        password_hashes = re.findall(r"[0-9a-f]{32}", html_lower)
        baseline_hashes = re.findall(r"[0-9a-f]{32}", baseline.get("html_lower", ""))
        new_hashes = len(password_hashes) - len(baseline_hashes)
        if new_hashes > 0:
            leaked = True
            count = max(count, new_hashes)
            if "password" not in fields:
                fields.append("password_hash")

        # 3) Search full page text for data leak keywords — but ONLY if
        #    the pattern is NEW (not already present in the baseline page).
        #    Also require the keyword to appear as a label WITH a value,
        #    not just as a form label or heading on its own.
        full_text = soup.get_text().lower()
        for pattern in self.DATA_LEAK_PATTERNS:
            in_response = bool(re.search(pattern, full_text))
            in_baseline = bool(re.search(pattern, baseline_text)) if baseline_text else False
            if in_response and not in_baseline:
                # Verify the keyword appears with actual data (key: value,
                # key = value, or inside a table cell with content)
                has_value = bool(re.search(
                    pattern + r"[\s:=]+\S", full_text
                ))
                if has_value:
                    leaked = True
                    if pattern not in fields:
                        fields.append(pattern)

        # 4) Count table rows with actual data — only NEW rows beyond baseline
        total_data_rows = 0
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            data_rows = 0
            for row in rows:
                cells = row.find_all(["td", "th"])
                cell_text = " ".join(c.get_text(strip=True) for c in cells)
                if cell_text and len(cell_text) > 2:
                    data_rows += 1
            total_data_rows += max(data_rows - 1, 0)
        new_rows = total_data_rows - baseline_table_rows
        if new_rows > 0:
            leaked = True
            count = max(count, new_rows)

        # 5) Check <pre>, <code>, and common output divs — only NEW patterns
        output_blocks = soup.find_all(["pre", "code"])
        for div in soup.find_all("div"):
            cls = " ".join(div.get("class", []))
            if re.search(r"result|output|response|data|content", cls, re.I):
                output_blocks.append(div)
        for block in output_blocks:
            text = block.get_text().lower()
            for pattern in self.DATA_LEAK_PATTERNS:
                in_block = bool(re.search(pattern, text))
                in_baseline = bool(re.search(pattern, baseline_text)) if baseline_text else False
                if in_block and not in_baseline:
                    # Require keyword + value, not just the label
                    has_value = bool(re.search(pattern + r"[\s:=]+\S", text))
                    if has_value:
                        leaked = True
                        if pattern not in fields:
                            fields.append(pattern)

        # 6) Content volume comparison — if the injected response has
        #    significantly MORE repeating content structures (product cards,
        #    list items, article blocks) than baseline, the injection likely
        #    expanded a WHERE clause to return hidden rows.
        #    This catches "OR 1=1" style data retrieval on e-commerce sites
        #    where extra products appear as div/li blocks (not table rows).
        if not leaked and baseline_text:
            baseline_len = baseline.get("length", 0)
            response_len = len(str(soup))
            if baseline_len > 0:
                ratio = response_len / baseline_len
                # Count repeating content blocks (product cards, list items, etc.)
                _repeating_tags = soup.find_all(
                    ["article", "li", "section", "tr"]
                )
                # Also match div elements with product/card/item classes
                for div in soup.find_all("div"):
                    cls = " ".join(div.get("class", [])).lower()
                    if re.search(r"product|card|item|result|row|entry|listing", cls):
                        _repeating_tags.append(div)

                baseline_soup = BeautifulSoup(baseline.get("html_lower", ""), "lxml")
                _baseline_repeating = baseline_soup.find_all(
                    ["article", "li", "section", "tr"]
                )
                for div in baseline_soup.find_all("div"):
                    cls = " ".join(div.get("class", [])).lower()
                    if re.search(r"product|card|item|result|row|entry|listing", cls):
                        _baseline_repeating.append(div)

                new_blocks = len(_repeating_tags) - len(_baseline_repeating)
                # If response is >30% larger AND has more content blocks,
                # the injection retrieved hidden data
                if ratio > 1.3 and new_blocks >= 2:
                    leaked = True
                    count = max(count, new_blocks)
                    fields.append("hidden_rows")
                    logger.debug(
                        "Content volume leak: %.1fx larger, %d new blocks",
                        ratio, new_blocks,
                    )

        return leaked, count, list(set(fields))

    def _detect_auth_bypass(self, html_lower: str,
                            baseline_key: str = "sqli") -> bool:
        """
        Check if we bypassed authentication (generic — works on any site).

        Compares against the stored baseline: if the same indicator already
        appears on the normal page, it's not evidence of an auth bypass caused
        by injection — it's just the page's normal content.
        """
        baseline = self._baseline_responses.get(baseline_key, {})
        baseline_html = baseline.get("html_lower", "")

        bypass_indicators = [
            "welcome to the password protected area",
            "welcome.*admin",
            "you have logged in as",
            "admin.*panel",
            r"dashboard",
            r"logged\s*in\s*(as|successfully)",
            r"my\s*account",
            r"sign\s*out|log\s*out",
            r"profile\s*settings",
        ]
        for p in bypass_indicators:
            in_response = bool(re.search(p, html_lower))
            in_baseline = bool(re.search(p, baseline_html)) if baseline_html else False
            # Only flag as bypass if the indicator is NEW (not in baseline)
            if in_response and not in_baseline:
                return True
        return False

    # Tags whose text content is NOT rendered as HTML by browsers —
    # a payload reflected inside these is NOT executable.
    _NON_EXECUTABLE_PARENTS = {
        "textarea", "title", "noscript", "style", "template",
        "xmp", "listing", "plaintext", "iframe",  # srcdoc is separate
    }

    def _detect_injected_script(self, html: str, payload: str) -> bool:
        """
        Detect if our XSS payload's script tag made it into the page unescaped
        AND in an executable HTML context.

        Returns False when:
          - The payload is HTML-entity-encoded (&lt;script&gt;)
          - The payload lands inside a non-executable parent element
            (textarea, title, noscript, style, template, etc.)
          - The payload lands inside an HTML comment
          - The Content-Type is not HTML (checked by caller)
        """
        if "<script" not in payload.lower():
            return False

        # Extract the script body from our payload to match against
        payload_script = re.search(r"<script[^>]*>(.*?)</script>", payload, re.I | re.DOTALL)
        if not payload_script:
            return False
        payload_body = payload_script.group(1).strip().lower()

        # Quick check: if the raw HTML contains the HTML-encoded version
        # but NOT the raw version, the server escaped it → not executable
        html_lower = html.lower()
        raw_tag = "<script"
        encoded_variants = ["&lt;script", "&#60;script", "\\u003cscript"]
        has_raw = raw_tag in html_lower
        has_encoded = any(enc in html_lower for enc in encoded_variants)
        if has_encoded and not has_raw:
            return False

        try:
            soup = BeautifulSoup(html, "lxml")
            container = soup.body if soup.body else soup

            for script in container.find_all("script"):
                script_text = (script.string or "").strip().lower()
                if not script_text:
                    continue

                # Check the script content matches our payload
                is_ours = False
                if payload_body and payload_body in script_text:
                    is_ours = True
                elif any(kw in script_text for kw in ("alert(", "prompt(", "confirm(",
                                                       "document.cookie", "eval(")):
                    is_ours = True

                if not is_ours:
                    continue

                # Verify the script is NOT inside a non-executable parent
                in_non_exec = False
                for parent in script.parents:
                    if parent.name and parent.name.lower() in self._NON_EXECUTABLE_PARENTS:
                        in_non_exec = True
                        break
                if in_non_exec:
                    continue

                # Verify it's not inside an HTML comment
                # (BeautifulSoup strips comments, but double-check raw HTML)
                # Find the approximate position in raw HTML
                script_str = str(script)
                pos = html_lower.find(script_str.lower()[:40])
                if pos >= 0:
                    # Check if there's an unclosed <!-- before this position
                    before = html_lower[:pos]
                    open_comments = before.count("<!--")
                    close_comments = before.count("-->")
                    if open_comments > close_comments:
                        continue  # Inside a comment

                return True
        except Exception:
            pass

        return False

    def _detect_event_handlers(self, html: str, payload: str) -> bool:
        """
        Check if our XSS event handler payload is reflected inside an actual
        HTML tag attribute — not just the string appearing in page text.

        Example TRUE positive:
          payload: <img src=x onerror=alert(1)>
          html:    ...<img src=x onerror=alert(1)>...

        Example FALSE positive (text, not attribute):
          html:    ...The username <img src=x onerror=alert(1)> is invalid...
          (only true if BeautifulSoup parses it as a real <img> tag)
        """
        payload_lower = payload.lower()
        handlers_in_payload = [
            h for h in self.XSS_EVENT_HANDLERS if h in payload_lower
        ]
        if not handlers_in_payload:
            return False

        # Use BeautifulSoup to find actual tags with event handler attributes
        try:
            soup = BeautifulSoup(html, "lxml")
            container = soup.body if soup.body else soup
            for tag in container.find_all(True):  # all tags
                # Skip tags inside non-executable parents
                in_non_exec = False
                for parent in tag.parents:
                    if parent.name and parent.name.lower() in self._NON_EXECUTABLE_PARENTS:
                        in_non_exec = True
                        break
                if in_non_exec:
                    continue

                for handler in handlers_in_payload:
                    attr_val = tag.get(handler)
                    if attr_val is not None:
                        # The handler attribute exists on a real HTML element
                        return True
        except Exception:
            pass

        return False

    def _determine_xss_context(self, html: str, payload: str) -> str:
        """
        Determine in what context the XSS payload was reflected.
        Returns one of: "javascript", "attribute", "dom", "html", "".
        """
        if not payload:
            return ""

        payload_lower = payload.lower()
        html_lower = html.lower()

        # Check if reflected inside a <script> block
        script_blocks = re.findall(r"<script[^>]*>(.*?)</script>",
                                    html_lower, re.DOTALL)
        for block in script_blocks:
            if payload_lower in block:
                return "javascript"

        # Check for DOM-based XSS sinks — payload may appear as a JS string
        # assignment that gets written to DOM via innerHTML, document.write, etc.
        dom_sinks = [
            r"\.innerHTML\s*=",
            r"\.outerHTML\s*=",
            r"document\.write\s*\(",
            r"document\.writeln\s*\(",
            r"eval\s*\(",
            r"setTimeout\s*\(",
            r"setInterval\s*\(",
            r"\.insertAdjacentHTML\s*\(",
        ]
        for block in script_blocks:
            for sink in dom_sinks:
                if re.search(sink, block):
                    # Check if payload or a URL param reference is near this sink
                    if payload_lower in block or "location" in block or "document.url" in block:
                        return "dom"

        # Check if reflected inside an HTML attribute
        attr_pattern = re.compile(
            r'(value|href|src|action|data|style|on\w+)\s*=\s*["\'][^"\']*'
            + re.escape(payload_lower)
        )
        if attr_pattern.search(html_lower):
            return "attribute"

        # Default: HTML body context
        if payload_lower in html_lower:
            return "html"

        return ""

    def _differs_from_baseline(self, vuln_type: str, html: str) -> bool:
        """Check if response differs significantly from baseline."""
        baseline = self._baseline_responses.get(vuln_type)
        if not baseline:
            return False

        content = self._extract_main_content(html)
        return hash(content) != baseline["text_hash"]

    def _compute_response_direction(self, baseline_key: str, html: str) -> int:
        """
        Compare response size to baseline and return direction.

        Returns:
            +1 if response is significantly LARGER than baseline (>20%)
            -1 if response is significantly SMALLER than baseline (>20%)
             0 if roughly the same size
        """
        baseline = self._baseline_responses.get(baseline_key, {})
        baseline_len = baseline.get("length", 0)
        if baseline_len == 0:
            return 0
        ratio = len(html) / baseline_len
        if ratio > 1.2:
            return 1   # larger — possible data leak (OR 1=1 returned more rows)
        elif ratio < 0.8:
            return -1  # smaller — possible true condition filtered out data
        return 0

    def _is_cmdi_time_based_hit(self, baseline_key: str, payload: str,
                                response_time: float) -> bool:
        """
        Detect blind command injection via response time delay.

        Checks for CMDi-specific delay commands (sleep, ping, timeout)
        and compares against baseline response time.
        """
        cmdi_delay_keywords = ("sleep", "timeout", "ping -n", "ping -c")
        if not any(kw in payload.lower() for kw in cmdi_delay_keywords):
            return False
        baseline = self._baseline_responses.get(baseline_key, {})
        baseline_time = baseline.get("response_time", 0.0)
        delta = response_time - baseline_time
        # CMDi sleep/ping commands: 4s threshold (sleep 5 minus network jitter)
        return delta >= 4.0

    def _is_time_based_hit(self, baseline_key: str, payload: str,
                           response_time: float) -> bool:
        """Check if response time indicates successful time-based injection.

        Compares against baseline response time to avoid false positives
        on naturally slow servers. Requires both:
          1. Payload contains a time-delay keyword (SLEEP, WAITFOR, etc.)
          2. Response took at least TIME_BASED_MIN_DELTA seconds LONGER than baseline
        """
        time_keywords = ("sleep", "waitfor", "pg_sleep", "benchmark")
        if not any(kw in payload.lower() for kw in time_keywords):
            return False

        baseline = self._baseline_responses.get(baseline_key, {})
        baseline_time = baseline.get("response_time", 0.0)
        delta = response_time - baseline_time
        return delta >= self.TIME_BASED_MIN_DELTA

    @staticmethod
    def _extract_main_content(html: str) -> str:
        """Extract the main content area (ignore headers/footers)."""
        soup = BeautifulSoup(html, "lxml")
        main = soup.find("div", {"class": "body_padded"})
        if main:
            return main.get_text(strip=True)
        return soup.get_text(strip=True)[:2000]
