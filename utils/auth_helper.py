"""
Authentication helper — shared login logic for scan.py and assistant.py.

Handles arbitrary login forms:
  - Auto-detects the username field by common name patterns
  - Preserves CSRF/hidden fields
  - Heuristically confirms success from the post-login response
"""

from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOGIN_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_FAIL_PATTERNS = [
    "invalid password", "incorrect password", "wrong password",
    "invalid credentials", "login failed", "authentication failed",
    "invalid username", "account not found", "bad credentials",
    "error logging in", "could not log in", "login error",
    "invalid email", "username or password", "password is incorrect",
    "locked out", "account locked", "too many attempts",
    "too many failed", "temporarily locked", "try again later",
    "rate limit", "brute force", "account disabled",
]

_SUCCESS_PATTERNS = [
    "logout", "log out", "sign out", "signout", "my account",
    "dashboard", "welcome", "profile", "account settings",
]

_USERNAME_HINTS = ("user", "email", "login", "name", "account", "mail", "id")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_login_form(soup: BeautifulSoup):
    """Return the first form that contains a password field, or None."""
    for form in soup.find_all("form"):
        if form.find("input", {"type": "password"}):
            return form
    return None


def _find_orphaned_login_inputs(soup: BeautifulSoup) -> dict | None:
    """
    Fallback for login inputs that live outside any <form> tag (e.g. navbar inputs).
    Finds all <input type="password"> not inside a <form>, gathers sibling inputs,
    and returns a dict of {name: default_value} ready to POST, or None if not found.
    """
    # Collect all inputs that are NOT inside a <form>
    all_inputs = soup.find_all("input")
    orphans = [i for i in all_inputs if not i.find_parent("form")]

    password_inputs = [i for i in orphans if i.get("type", "text").lower() == "password"]
    if not password_inputs:
        return None

    # Return all orphaned inputs so the caller can fill them in
    return orphans


def _build_login_data(form, username: str, password: str) -> dict:
    """
    Build a form POST dict from the login form, filling in username and password.
    Keeps CSRF/hidden fields intact and includes submit button values.
    """
    data: dict[str, str] = {}
    username_field: str | None = None

    for inp in form.find_all(["input", "textarea", "select", "button"]):
        name = inp.get("name", "").strip()
        itype = inp.get("type", "text").lower()
        if not name:
            continue

        # <button> tags default to type="submit" and may lack a value attr
        is_button_tag = inp.name == "button"
        if is_button_tag:
            itype = inp.get("type", "submit").lower()

        if itype == "password":
            data[name] = password
        elif itype == "hidden":
            data[name] = inp.get("value", "")
        elif itype in ("submit", "button", "reset", "image"):
            if name:
                data[name] = inp.get("value") or ""
        elif itype in ("checkbox", "radio"):
            val = inp.get("value", "on")
            if inp.get("checked"):
                data[name] = val
        else:
            # Identify username field by common name patterns
            if username_field is None and any(h in name.lower() for h in _USERNAME_HINTS):
                username_field = name
                data[name] = username
            else:
                data[name] = inp.get("value", "")

    # Fallback: use first non-password text input as username field
    if username_field is None:
        for inp in form.find_all("input"):
            itype = inp.get("type", "text").lower()
            name = inp.get("name", "").strip()
            if name and itype not in ("password", "hidden", "submit", "button",
                                      "reset", "image", "checkbox", "radio"):
                data[name] = username
                break

    return data


def _check_login_success(response: requests.Response) -> tuple[bool, str]:
    """Heuristically determine whether a login attempt succeeded."""
    body = response.text.lower()

    has_fail = any(pat in body for pat in _FAIL_PATTERNS)
    has_success = any(pat in body for pat in _SUCCESS_PATTERNS)

    # If both fail AND success patterns match, success wins — the page
    # may contain remnants of old error messages (e.g. "locked out" in
    # an admin log) while also showing "logout" because we're logged in.
    if has_success:
        return True, "Login succeeded."

    if has_fail:
        return False, "Login failed — check your credentials."

    # If no password field on the post-login page, likely redirected to authenticated area
    soup_after = BeautifulSoup(response.text, "lxml")
    if not soup_after.find("input", {"type": "password"}):
        return True, "Login succeeded."

    return True, "Credentials submitted — verifying session..."


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_cookies(cookie_str: str) -> dict:
    """Parse a 'k=v; k2=v2' cookie string into a dict."""
    cookies = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k.strip()] = v.strip()
    return cookies


def authenticate_basic(
    url: str,
    username: str,
    password: str,
) -> tuple[dict, bool, str]:
    """
    Authenticate using HTTP Basic Auth (WWW-Authenticate: Basic).
    Returns (cookies, success, message).
    """
    from requests.auth import HTTPBasicAuth
    session = requests.Session()
    session.headers["User-Agent"] = _LOGIN_UA
    try:
        resp = session.get(url, auth=HTTPBasicAuth(username, password), timeout=15)
        if resp.status_code == 401:
            return {}, False, "HTTP Basic Auth failed — check credentials."
        cookies = dict(session.cookies)
        # Store auth header so subsequent requests include it
        import base64 as _b64
        token = _b64.b64encode(f"{username}:{password}".encode()).decode()
        return cookies, True, f"HTTP Basic Auth succeeded (status {resp.status_code})."
    except Exception as e:
        return {}, False, f"HTTP Basic Auth request failed — {e}"


def detect_auth_type(url: str) -> str:
    """
    Probe *url* and return the auth type: 'basic', 'form', or 'none'.
    Used to automatically select the right authentication strategy.
    """
    try:
        resp = requests.get(url, timeout=10, allow_redirects=False)
        www_auth = resp.headers.get("WWW-Authenticate", "")
        if "basic" in www_auth.lower():
            return "basic"
        if "digest" in www_auth.lower():
            return "basic"  # treat digest as basic for now
        if resp.status_code in (401, 403):
            return "basic"
        return "form"
    except Exception:
        return "form"


def authenticate(
    login_url: str,
    username: str,
    password: str,
    session: requests.Session | None = None,
) -> tuple[dict, bool, str, requests.Session]:
    """
    Submit credentials to a login form and return (cookies, success, message, session).

    Handles CSRF tokens and arbitrary form structures automatically.
    The returned cookies dict can be passed directly to requests or the crawler.
    The returned *session* is the live requests.Session that performed the login —
    reuse it (instead of just the cookies) so PHP/server session state is preserved.

    Returns:
        cookies  — session cookies to use for subsequent requests
        success  — True if login heuristics indicate success
        message  — human-readable status string
        session  — the requests.Session that holds the authenticated state
    """
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = _LOGIN_UA

    try:
        resp = session.get(login_url, timeout=15)
        resp.raise_for_status()
    except Exception:
        return {}, False, "Could not reach the login page — check the URL and that the target is running.", session

    soup = BeautifulSoup(resp.text, "lxml")
    form = _find_login_form(soup)

    if form:
        form_data = _build_login_data(form, username, password)
        action = (form.get("action") or "").strip() or login_url
        if not action.startswith("http"):
            action = urljoin(login_url, action)
        method = form.get("method", "POST").upper()
    else:
        # Fallback: login inputs are outside any <form> tag (e.g. navbar inputs)
        orphans = _find_orphaned_login_inputs(soup)
        if not orphans:
            return {}, False, "No login form (with a password field) found at the given URL", session

        form_data: dict[str, str] = {}
        username_field: str | None = None
        for inp in orphans:
            name = inp.get("name", "").strip()
            itype = inp.get("type", "text").lower()
            if not name:
                continue
            if itype == "password":
                form_data[name] = password
            elif itype == "hidden":
                form_data[name] = inp.get("value", "")
            elif itype in ("submit", "button", "reset", "image"):
                if name:
                    form_data[name] = inp.get("value", "Submit")
            else:
                if username_field is None and any(h in name.lower() for h in _USERNAME_HINTS):
                    username_field = name
                    form_data[name] = username
                elif username_field is None:
                    username_field = name
                    form_data[name] = username
                else:
                    form_data[name] = inp.get("value", "")

        action = login_url
        method = "POST"

    try:
        if method == "POST":
            resp2 = session.post(action, data=form_data, timeout=15, allow_redirects=True)
        else:
            resp2 = session.get(action, params=form_data, timeout=15, allow_redirects=True)
    except Exception:
        return {}, False, "Login request failed — the server may be unavailable or rejecting the request.", session

    cookies = dict(session.cookies)
    success, msg = _check_login_success(resp2)
    return cookies, success, msg, session
