"""
Heuristic Security Checks
Deterministic vulnerability checks that don't require RL training.
These probe for logic flaws, misconfigurations, and design weaknesses
that injection-based scanning misses.

Each check returns a list of finding dicts compatible with the scanner
report format.
"""

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standard finding format (compatible with scan.py report)
# ---------------------------------------------------------------------------

def _finding(
    vuln_category: str,
    url: str,
    parameter: str,
    method: str,
    severity: str,
    description: str,
    evidence: list[str],
    remediation: str,
    payloads: list[str] | None = None,
) -> dict:
    return {
        "check_type": "heuristic",
        "vuln_category": vuln_category,
        "url": url,
        "parameter": parameter,
        "method": method,
        "success_rate": 1.0,
        "severity": severity,
        "description": description,
        "impact_evidence": evidence,
        "remediation": remediation,
        "successful_payloads": payloads or [],
        "vuln_subtype": vuln_category,
    }


# ---------------------------------------------------------------------------
# Check 1: CSRF Token Not Validated
# ---------------------------------------------------------------------------

def check_csrf_validation(
    session: requests.Session,
    injection_points: list,
    cookies: dict | None = None,
) -> list[dict]:
    """
    For each POST form that has hidden fields (likely CSRF tokens),
    submit the form WITHOUT the token and with a FAKE token.
    If the server accepts either, CSRF protection is broken.
    """
    findings = []
    tested = set()

    for point in injection_points:
        if point.method.upper() != "POST":
            continue
        if point.url in tested:
            continue
        tested.add(point.url)

        # Need a form page to detect CSRF fields
        form_url = point.form_page_url or point.url
        try:
            resp = session.get(form_url, timeout=10, cookies=cookies)
            soup = BeautifulSoup(resp.text, "lxml")
        except Exception:
            continue

        # Find hidden inputs that look like CSRF tokens
        csrf_fields = []
        for inp in soup.find_all("input", {"type": "hidden"}):
            name = (inp.get("name") or "").lower()
            if any(tok in name for tok in ("csrf", "token", "nonce", "_verify", "authenticity")):
                csrf_fields.append((inp.get("name"), inp.get("value", "")))

        if not csrf_fields:
            continue

        # Build base form data with all other fields
        base_data = dict(point.default_form_values) if point.default_form_values else {}
        base_data[point.parameter] = "csrf_test_value"

        # Test 1: Submit WITHOUT CSRF token
        no_token_data = {k: v for k, v in base_data.items()
                         if k.lower() not in [f[0].lower() for f in csrf_fields]}
        try:
            r1 = session.post(point.url, data=no_token_data, timeout=10,
                              cookies=cookies, allow_redirects=True)
            no_token_accepted = (r1.status_code < 400 and
                                 not _has_csrf_rejection(r1.text))
        except Exception:
            no_token_accepted = False

        # Test 2: Submit WITH fake CSRF token
        fake_data = dict(base_data)
        for field_name, _ in csrf_fields:
            fake_data[field_name] = f"fake_{uuid.uuid4().hex[:16]}"
        try:
            r2 = session.post(point.url, data=fake_data, timeout=10,
                              cookies=cookies, allow_redirects=True)
            fake_accepted = (r2.status_code < 400 and
                             not _has_csrf_rejection(r2.text))
        except Exception:
            fake_accepted = False

        if no_token_accepted or fake_accepted:
            evidence = []
            if no_token_accepted:
                evidence.append(f"Server accepted POST without CSRF token (status {r1.status_code})")
            if fake_accepted:
                evidence.append(f"Server accepted POST with fake CSRF token (status {r2.status_code})")
            evidence.append(f"CSRF field(s): {', '.join(f[0] for f in csrf_fields)}")

            findings.append(_finding(
                vuln_category="CSRF Not Validated",
                url=point.url,
                parameter=csrf_fields[0][0],
                method="POST",
                severity="high",
                description="CSRF token is present but not validated server-side. "
                            "An attacker can forge requests on behalf of authenticated users.",
                evidence=evidence,
                remediation="Validate the CSRF token on every state-changing POST request. "
                            "Reject requests with missing or invalid tokens.",
            ))

    return findings


def _has_csrf_rejection(body: str) -> bool:
    """Check if the response indicates CSRF token rejection."""
    lower = body.lower()
    return any(p in lower for p in (
        "csrf", "invalid token", "token mismatch", "forbidden",
        "request verification", "security token",
    ))


# ---------------------------------------------------------------------------
# Check 2: Broken Access Control
# ---------------------------------------------------------------------------

def check_broken_access_control(
    authenticated_urls: list[str],
    base_url: str,
    auth_cookies: dict | None = None,
    unauth_cache: dict[str, tuple[str, int]] | None = None,
) -> list[dict]:
    """
    Request each authenticated URL without any session cookies.
    If the page content is served (not a login redirect), access control is broken.
    Uses unauth_cache to avoid re-fetching pages already fetched by the orchestrator.
    """
    findings = []
    unauth_cache = unauth_cache or {}
    # Also try common admin paths
    admin_paths = ["/admin", "/admin.php", "/dashboard", "/panel",
                   "/settings", "/config", "/users", "/manage"]
    extra_urls = [urljoin(base_url, p) for p in admin_paths]
    all_urls = list(set(authenticated_urls + extra_urls))

    # Get a reference login page to compare against
    login_patterns = re.compile(
        r"(login|sign.?in|log.?in|authenticate|password)",
        re.I,
    )

    unauth_session = requests.Session()  # fresh, no cookies

    for url in all_urls:
        try:
            # Use cache if available, otherwise fetch
            if url in unauth_cache:
                body, status_code = unauth_cache[url]
                final_url = url
            else:
                r = unauth_session.get(url, timeout=10, allow_redirects=True)
                body, status_code, final_url = r.text, r.status_code, r.url
                unauth_cache[url] = (body, status_code)

            if status_code >= 400:
                continue  # Correctly denied

            # Check if we got redirected to login
            if login_patterns.search(final_url) and final_url != url:
                continue  # Correctly redirected to login

            # Check if response looks like a login page
            if login_patterns.search(body[:2000]):
                # Could be the login page itself
                if "password" in body.lower()[:3000] and "<form" in body.lower()[:3000]:
                    continue

            # Got a 200 with real content — check if it has meaningful content
            if len(body) < 100:
                continue  # Too small, likely empty or error

            # Compare: if we have auth cookies, check that auth version has more content
            if auth_cookies:
                auth_session = requests.Session()
                for k, v in auth_cookies.items():
                    auth_session.cookies.set(k, v)
                try:
                    auth_r = auth_session.get(url, timeout=10)
                    # If unauth and auth responses are nearly identical,
                    # the page doesn't require auth (might be public)
                    if abs(len(auth_r.text) - len(body)) < 100:
                        continue  # Public page, not a finding
                except Exception:
                    pass

            findings.append(_finding(
                vuln_category="Broken Access Control",
                url=url,
                parameter="",
                method="GET",
                severity="medium",
                description=f"Page accessible without authentication. "
                            f"Response: {status_code}, {len(body)} bytes.",
                evidence=[
                    f"GET {url} without cookies returned {status_code} ({len(body)} bytes)",
                    "Page should require authentication but serves content to anonymous users",
                ],
                remediation="Enforce authentication checks on all protected pages. "
                            "Redirect unauthenticated users to the login page.",
            ))

        except Exception:
            continue

    return findings


# ---------------------------------------------------------------------------
# Check 3: IP Spoofing / Lockout Bypass
# ---------------------------------------------------------------------------

def check_ip_spoofing(
    session: requests.Session,
    login_url: str,
    username: str = "admin",
    password: str = "wrong_password_intentional",
    cookies: dict | None = None,
) -> list[dict]:
    """
    Test if the server trusts X-Forwarded-For and similar headers
    for IP-based decisions (lockout, rate limiting, logging).
    """
    findings = []
    spoof_headers = [
        "X-Forwarded-For",
        "X-Real-IP",
        "X-Originating-IP",
        "Client-IP",
        "X-Client-IP",
    ]

    # Step 1: Find the login form
    try:
        resp = session.get(login_url, timeout=10, cookies=cookies)
        soup = BeautifulSoup(resp.text, "lxml")
        form = soup.find("form")
        if not form:
            return findings

        # Find username and password field names
        inputs = form.find_all("input")
        uid_field = pwd_field = None
        for inp in inputs:
            name = (inp.get("name") or "").lower()
            itype = (inp.get("type") or "").lower()
            if itype == "password":
                pwd_field = inp.get("name")
            elif itype in ("text", "email", "") and not uid_field:
                uid_field = inp.get("name")

        if not uid_field or not pwd_field:
            return findings

        action = form.get("action", login_url)
        post_url = urljoin(login_url, action)

    except Exception:
        return findings

    # Step 2: Send requests with spoofed headers — each with a unique IP
    for header_name in spoof_headers:
        fake_ip = f"10.{hash(header_name) % 256}.{hash(header_name + 'x') % 256}.1"
        try:
            r = session.post(
                post_url,
                data={uid_field: username, pwd_field: password},
                headers={header_name: fake_ip},
                timeout=10,
                cookies=cookies,
                allow_redirects=True,
            )

            body_lower = r.text.lower()
            # Check if the fake IP appears in the response (logged/displayed)
            if fake_ip in r.text:
                findings.append(_finding(
                    vuln_category="IP Spoofing",
                    url=post_url,
                    parameter=header_name,
                    method="POST",
                    severity="medium",
                    description=f"Server uses {header_name} header value in its response. "
                                f"Attacker can spoof their IP address.",
                    evidence=[
                        f"Sent {header_name}: {fake_ip}",
                        f"Fake IP '{fake_ip}' appeared in the server response",
                        "Server trusts client-supplied IP headers without validation",
                    ],
                    remediation=f"Do not trust {header_name} for security decisions. "
                                "Use the actual TCP source IP (REMOTE_ADDR) instead.",
                    payloads=[f"{header_name}: {fake_ip}"],
                ))
        except Exception:
            continue

    return findings


# ---------------------------------------------------------------------------
# Check 4: Stored XSS via HTTP Headers
# ---------------------------------------------------------------------------

def check_header_stored_xss(
    session: requests.Session,
    target_urls: list[str],
    display_urls: list[str],
    cookies: dict | None = None,
) -> list[dict]:
    """
    Inject XSS payloads via HTTP headers (X-Forwarded-For, Referer, User-Agent)
    then check display pages (admin, logs) for unescaped reflection.
    """
    findings = []
    xss_marker = f"xss_header_{uuid.uuid4().hex[:8]}"
    xss_payloads = {
        "X-Forwarded-For": f"<script>alert('{xss_marker}')</script>",
        "Referer": f"http://evil.com/<script>alert('{xss_marker}')</script>",
    }

    # Step 1: Send requests with XSS in headers
    for url in target_urls[:5]:  # Limit to avoid spam
        for header_name, payload in xss_payloads.items():
            try:
                session.get(
                    url,
                    headers={header_name: payload},
                    timeout=10,
                    cookies=cookies,
                )
            except Exception:
                continue

    # Step 2: Check display pages for the stored XSS
    time.sleep(1)  # Brief wait for server to process/store

    for display_url in display_urls:
        try:
            r = session.get(display_url, timeout=10, cookies=cookies)
            if xss_marker in r.text:
                # Check if it's in an executable context
                if f"<script>" in r.text and xss_marker in r.text:
                    # Find which header caused it
                    for header_name in xss_payloads:
                        if xss_payloads[header_name] in r.text or xss_marker in r.text:
                            findings.append(_finding(
                                vuln_category="Stored XSS (Header)",
                                url=display_url,
                                parameter=header_name,
                                method="GET",
                                severity="high",
                                description=f"XSS payload injected via {header_name} header "
                                            f"is stored and rendered unescaped on {display_url}.",
                                evidence=[
                                    f"Injected XSS via {header_name} header on target pages",
                                    f"Payload found unescaped on {display_url}",
                                    "Browser would execute injected JavaScript on page load",
                                ],
                                remediation="HTML-encode all data before displaying it, "
                                            "including values from HTTP headers. "
                                            "Never trust X-Forwarded-For or Referer values.",
                                payloads=[f"{header_name}: {xss_payloads[header_name]}"],
                            ))
                            break
        except Exception:
            continue

    return findings


# ---------------------------------------------------------------------------
# Check 5: Multi-Step Stored XSS (login/signup username)
# ---------------------------------------------------------------------------

def check_stored_xss_via_login(
    session: requests.Session,
    login_url: str,
    authenticated_urls: list[str],
    username: str = "",
    password: str = "",
    cookies: dict | None = None,
) -> list[dict]:
    """
    Inject XSS as the username during login/signup, then check if it
    appears unescaped on other authenticated pages.
    """
    findings = []
    if not username or not password:
        return findings

    xss_marker = f"xss_stored_{uuid.uuid4().hex[:8]}"
    xss_username = f"<script>alert('{xss_marker}')</script>"

    # Find login form
    try:
        resp = session.get(login_url, timeout=10, cookies=cookies)
        soup = BeautifulSoup(resp.text, "lxml")
        form = soup.find("form")
        if not form:
            return findings

        inputs = form.find_all("input")
        uid_field = pwd_field = None
        hidden_fields = {}
        for inp in inputs:
            name = inp.get("name") or ""
            itype = (inp.get("type") or "").lower()
            if itype == "password":
                pwd_field = name
            elif itype == "hidden":
                hidden_fields[name] = inp.get("value", "")
            elif itype in ("text", "email", "") and not uid_field:
                uid_field = name

        if not uid_field or not pwd_field:
            return findings

        action = form.get("action", login_url)
        post_url = urljoin(login_url, action)

    except Exception:
        return findings

    # Submit login with XSS username (may fail login but still store the username)
    try:
        form_data = dict(hidden_fields)
        form_data[uid_field] = xss_username
        form_data[pwd_field] = password
        session.post(post_url, data=form_data, timeout=10,
                     cookies=cookies, allow_redirects=True)
    except Exception:
        return findings

    # Also try signup forms if they exist
    signup_urls = [u for u in authenticated_urls if "signup" in u.lower() or "register" in u.lower()]
    for signup_url in signup_urls[:2]:
        try:
            resp = session.get(signup_url, timeout=10, cookies=cookies)
            soup = BeautifulSoup(resp.text, "lxml")
            form = soup.find("form")
            if form:
                action = form.get("action", signup_url)
                post_url = urljoin(signup_url, action)
                form_data = {}
                for inp in form.find_all("input"):
                    name = inp.get("name") or ""
                    itype = (inp.get("type") or "").lower()
                    if itype == "password":
                        form_data[name] = "TestPass123!"
                    elif itype == "hidden":
                        form_data[name] = inp.get("value", "")
                    elif itype in ("text", "email", ""):
                        form_data[name] = xss_username
                session.post(post_url, data=form_data, timeout=10,
                             cookies=cookies, allow_redirects=True)
        except Exception:
            continue

    # Now check all authenticated pages for the stored XSS marker
    time.sleep(1)

    # Re-login with real credentials to access authenticated pages
    try:
        form_data = dict(hidden_fields)
        form_data[uid_field] = username
        form_data[pwd_field] = password
        session.post(post_url, data=form_data, timeout=10, allow_redirects=True)
    except Exception:
        pass

    for page_url in authenticated_urls:
        try:
            r = session.get(page_url, timeout=10)
            if xss_marker in r.text:
                escaped = f"&lt;script&gt;" in r.text and xss_marker in r.text
                if not escaped:
                    findings.append(_finding(
                        vuln_category="Stored XSS (Multi-step)",
                        url=page_url,
                        parameter=uid_field,
                        method="POST",
                        severity="high",
                        description=f"XSS payload submitted as username is stored and "
                                    f"rendered unescaped on {page_url}.",
                        evidence=[
                            f"Submitted '{xss_username}' as username via login/signup",
                            f"Payload found unescaped on {page_url}",
                            "Stored XSS: any user visiting this page would execute the script",
                        ],
                        remediation="HTML-encode all user-supplied data before rendering. "
                                    "Sanitize usernames on input to reject HTML/script content.",
                        payloads=[xss_username],
                    ))
        except Exception:
            continue

    return findings


# ---------------------------------------------------------------------------
# Check 6: Passwords in GET Parameters
# ---------------------------------------------------------------------------

def check_passwords_in_get(
    crawled_html: dict[str, str],
    injection_points: list,
) -> list[dict]:
    """
    Flag any form that sends password fields via GET method.
    Also checks injection points for sensitive parameter names in GET requests.

    Args:
        crawled_html: dict of {url: html_content} from crawler
        injection_points: list of InjectionPoint objects
    """
    findings = []
    sensitive_params = re.compile(
        r"(passw|passwd|pwd|secret|token|api.?key|auth|ssn|credit)",
        re.I,
    )

    # Check crawled HTML for GET forms with password fields
    checked_urls = set()
    for url, html in crawled_html.items():
        soup = BeautifulSoup(html, "lxml")
        for form in soup.find_all("form"):
            method = (form.get("method") or "GET").upper()
            if method != "GET":
                continue
            # Check if form has password-type input
            pwd_inputs = form.find_all("input", {"type": "password"})
            if pwd_inputs:
                action = form.get("action", url)
                form_url = urljoin(url, action)
                if form_url in checked_urls:
                    continue
                checked_urls.add(form_url)
                param_names = [inp.get("name", "?") for inp in pwd_inputs]
                findings.append(_finding(
                    vuln_category="Passwords in GET",
                    url=form_url,
                    parameter=", ".join(param_names),
                    method="GET",
                    severity="high",
                    description="Form submits password field(s) via GET method. "
                                "Passwords will appear in URL, browser history, "
                                "server logs, and proxy logs.",
                    evidence=[
                        f"Form at {url} uses method='GET' with password input(s): {', '.join(param_names)}",
                        "Sensitive data exposed in URL query string",
                    ],
                    remediation="Change the form method to POST. "
                                "Never send passwords or sensitive data in URL parameters.",
                ))

    # Check injection points for sensitive GET params
    for point in injection_points:
        if point.method.upper() != "GET":
            continue
        if sensitive_params.search(point.parameter):
            key = f"{point.url}:{point.parameter}"
            if key in checked_urls:
                continue
            checked_urls.add(key)
            findings.append(_finding(
                vuln_category="Sensitive Data in GET",
                url=point.url,
                parameter=point.parameter,
                method="GET",
                severity="high",
                description=f"Sensitive parameter '{point.parameter}' is sent via GET. "
                            "Value will be visible in URL, logs, and browser history.",
                evidence=[
                    f"GET {point.url} uses parameter '{point.parameter}' which appears sensitive",
                    "Sensitive data exposed in URL query string",
                ],
                remediation="Use POST method for sensitive parameters. "
                            "Never send passwords, tokens, or secrets in GET requests.",
            ))

    return findings


# ---------------------------------------------------------------------------
# Runner: execute all heuristic checks
# ---------------------------------------------------------------------------

def run_all_heuristic_checks(
    session: requests.Session,
    injection_points: list,
    crawled_urls: list[str],
    crawled_html: dict[str, str],
    base_url: str,
    cookies: dict | None = None,
    login_url: str = "",
    username: str = "",
    password: str = "",
) -> list[dict]:
    """
    Run all 6 heuristic security checks and return combined findings.
    Uses a shared page cache to avoid redundant HTTP fetches across checks.
    """
    all_findings = []

    # Pre-fetch crawled URLs without auth (shared across access control + stored XSS checks)
    _unauth_cache: dict[str, tuple[str, int]] = {}  # url -> (body, status)
    logger.info("Running heuristic security checks...")
    logger.info("  Pre-fetching pages for heuristic checks...")
    unauth_session = requests.Session()
    for url in crawled_urls[:30]:
        try:
            r = unauth_session.get(url, timeout=8, allow_redirects=True)
            _unauth_cache[url] = (r.text, r.status_code)
        except Exception:
            pass

    # Check 6: Passwords in GET (fastest, no HTTP needed)
    logger.info("  [1/6] Checking for passwords in GET parameters...")
    all_findings.extend(check_passwords_in_get(crawled_html, injection_points))

    # Check 1: CSRF validation
    logger.info("  [2/6] Checking CSRF token validation...")
    all_findings.extend(check_csrf_validation(session, injection_points, cookies))

    # Check 2: Broken access control
    logger.info("  [3/6] Checking broken access control...")
    all_findings.extend(check_broken_access_control(crawled_urls, base_url, cookies,
                                                      unauth_cache=_unauth_cache))

    # Check 3: IP spoofing
    if login_url:
        logger.info("  [4/6] Checking IP spoofing via header injection...")
        all_findings.extend(check_ip_spoofing(session, login_url, username or "admin",
                                               cookies=cookies))
    else:
        logger.info("  [4/6] Skipping IP spoofing check (no login URL)")

    # Check 4: Stored XSS via headers
    display_urls = [u for u in crawled_urls
                    if any(kw in u.lower() for kw in ("admin", "log", "dashboard", "panel", "index"))]
    if display_urls:
        logger.info("  [5/6] Checking stored XSS via HTTP headers...")
        all_findings.extend(check_header_stored_xss(session, crawled_urls[:10],
                                                      display_urls, cookies))
    else:
        logger.info("  [5/6] Skipping header stored XSS (no admin/log pages found)")

    # Check 5: Multi-step stored XSS
    if login_url and username and password:
        logger.info("  [6/6] Checking multi-step stored XSS...")
        all_findings.extend(check_stored_xss_via_login(session, login_url, crawled_urls,
                                                        username, password, cookies))
    else:
        logger.info("  [6/6] Skipping multi-step stored XSS (no credentials provided)")

    logger.info("  Heuristic checks complete: %d finding(s)", len(all_findings))
    return all_findings
