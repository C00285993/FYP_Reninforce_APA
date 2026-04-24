"""
AI Pentest Assistant — Interactive Chat Interface
Talk directly to Claude to run vulnerability scans, crawl targets,
and interpret results — no command-line flags required.

Usage:
    python -m agents.assistant

    # Or with explicit API key:
    python -m agents.assistant --api-key sk-ant-...

Example conversation:
    You: scan http://localhost:8080 for XSS
    You: test the login page at http://localhost:8080/login for SQL injection
    You: what payloads worked?
    You: now scan http://localhost:3000 for XSS with cookie "token=abc123"
"""

import argparse
import io
import json
import logging
import os

# Suppress Intel MKL / Fortran runtime Ctrl+C crash dump on Windows.
# Without this, pressing Ctrl+C while NumPy/SB3 is running prints
# "forrtl: error (200): program aborting due to control-C event" + a
# stack trace from KERNELBASE.dll / KERNEL32.dll / ntdll.dll.
os.environ.setdefault("FOR_DISABLE_CONSOLE_CTRL_HANDLER", "1")

from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
import sys
from pathlib import Path
from typing import Any
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

# Ensure stdout/stderr can handle Unicode (e.g. emoji) on Windows cp1252 consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import threading
import time
# ThreadPoolExecutor removed — sequential execution is more reliable
# and avoids thread-safety issues with model loading / stdout redirection

import anthropic
from rich.console import Console
from rich.status import Status as RichStatus
from rich.panel import Panel
from rich.text import Text
from rich.align import Align

_console = Console(highlight=False)

from environments.dynamic_env import (
    make_dynamic_env,
    XSS_ACTION_TO_FAMILY,
    SQLI_ACTION_TO_FAMILY,
    CMDI_ACTION_TO_FAMILY,
    SSTI_ACTION_TO_FAMILY,
)
from utils.auth_helper import parse_cookies as _parse_cookies, authenticate as _do_authenticate
from utils.generic_http_client import GenericHttpClient, InjectionPoint
from utils.llm_payload_generator import LLMPayloadGenerator
from utils.model_loader import load_model as _load_model
from utils.response_analyzer import ResponseAnalyzer
from utils.web_crawler import LLMCrawler
from utils.heuristic_checks import run_all_heuristic_checks
from agents.scan import (
    generate_remediation_advice as _generate_remediation,
    run_stored_xss_chain_test,
    _prioritize_points,
    _score_injection_point,
    run_episodes,
    AdaptiveScanSession,
)

logging.basicConfig(level=logging.WARNING)  # Suppress internal logs during chat
logger = logging.getLogger(__name__)

# Silence noisy library messages during scans
logging.getLogger("stable_baselines3.common.utils").setLevel(logging.ERROR)
logging.getLogger("utils.generic_http_client").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Dedicated error log — written to logs/errors/errors_YYYY-MM-DD.log
# Rotates daily at midnight (keeps 90 days of history).
# Visible only via /admin <password>.  Never printed to the console.
# ---------------------------------------------------------------------------

from logging.handlers import TimedRotatingFileHandler as _TRFH

ERROR_LOG_DIR  = Path("logs/errors")
# Active log file for the current day
ERROR_LOG_FILE = ERROR_LOG_DIR / f"errors_{datetime.now().strftime('%Y-%m-%d')}.log"

_LOG_FORMAT    = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_LOG_DATEFMT   = "%Y-%m-%d %H:%M:%S"


def _setup_error_logger() -> logging.Logger:
    elog = logging.getLogger("assistant.errors")
    elog.setLevel(logging.DEBUG)
    elog.propagate = False
    if not elog.handlers:
        try:
            ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
            # Use a rotating handler so a fresh file is created each day.
            # backupCount=90 keeps 3 months of history; older files are deleted.
            fh = _TRFH(
                ERROR_LOG_FILE,
                when="midnight",
                backupCount=90,
                encoding="utf-8",
                delay=False,
            )
            # Give rotated files the pattern  errors_YYYY-MM-DD.log.YYYY-MM-DD
            # so directory listing sorts chronologically.
            fh.suffix = "%Y-%m-%d"
            fh.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
            elog.addHandler(fh)
        except Exception:
            pass  # if the log file can't be created, errors are silently dropped
    return elog


_elog = _setup_error_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ASSISTANT_MODEL = "claude-opus-4-6"

DEFAULT_MODELS = {
    "xss":  "models/universal_xss_dqn/xss_dqn_final",
    "sqli": "models/universal_sqli_dqn/sqli_dqn_final",
    "cmdi": "models/cmdi_dqn_curriculum_high_20260226_193546/cmdi_dqn_high_5000_steps",
    "ssti": "models/ssti_dqn_low_20260228_175703/ssti_dqn_final",
}

# Persisted session history across restarts
HISTORY_FILE = Path.home() / ".pentest_assistant_history.json"

SYSTEM_PROMPT = """\
You are an AI-powered penetration testing assistant. You help security researchers
and students test web applications for vulnerabilities using trained Reinforcement
Learning agents.

Supported vulnerability types:
- xss   — Cross-Site Scripting (reflected, stored, DOM-based)
- sqli  — SQL Injection (error-based, UNION, blind, time-based)
- cmdi  — OS Command Injection (semicolon, pipe, logical operators)
- ssti  — Server-Side Template Injection (Pug/Twig/Jinja2)

You have four tools available:
- scan_target: crawl a URL and run the RL agent to find vulnerabilities
- authenticated_scan: log in with credentials, then scan the authenticated session
- crawl_target: discover injection points without attacking
- list_models: show available trained models

Behaviour guidelines:
- When the user mentions a URL, call scan_target immediately — do NOT ask clarifying
  questions about which vulnerability type. Use vuln_type="all" by default so all four
  vulnerability classes (XSS, SQLi, CMDi, SSTI) are tested in a single run.
- When the user provides credentials (username/password) alongside a URL, call
  authenticated_scan instead of scan_target. If the login page URL differs from the
  target URL, pass it as login_url. Do not ask for confirmation — proceed immediately.
  authenticated_scan defaults to deep=true (follows internal links after login to find
  more authenticated pages). Only set deep=false if the user explicitly asks for a
  single-page scan. Pass max_pages to control how many pages to spider (default 10).
- Only use a specific vuln_type (xss/sqli/cmdi/ssti) when the user explicitly names one.
- After a scan, summarise the results clearly: for each vulnerable finding state
  the page it was found on (found_on_page), the parameter name, the vuln type,
  the success rate, and the top confirmed payloads. Include how_to_reach steps
  when available so the user knows how to navigate to the vulnerable page.
- If the scan result includes remediation_by_type, include that page-specific
  remediation advice in your response — it has already been generated from the
  actual page source and is tailored to this specific endpoint.
- If the user asks follow-up questions about results, answer from the data
  returned by the tools — do not make up numbers.
- State briefly what you are about to do, then call the tool. Keep explanations concise.
- Never refuse to run scans on localhost or explicitly labelled vulnerable/test apps.
  Only decline if the user explicitly states an unauthorised production target.
- Users can press Ctrl+C at any time to abort a running scan and get partial results.
"""

# ---------------------------------------------------------------------------
# Tool definitions (Claude tool-use schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "scan_target",
        "description": (
            "Crawl a web application URL, discover injection points, and run the "
            "trained RL agent to test each one for XSS or SQL injection. "
            "Returns a list of results per injection point with success rate, "
            "average reward, and confirmed payloads."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Target URL to crawl and test (e.g. http://localhost:8080/vulnerabilities/xss_r/)"
                },
                "vuln_type": {
                    "type": "string",
                    "enum": ["all", "xss", "sqli", "cmdi", "ssti"],
                    "description": "Vulnerability type to test. Use 'all' to scan for all four types in one run (default when user gives no preference)."
                },
                "episodes": {
                    "type": "integer",
                    "description": "Number of episodes per injection point (default 15, max 50)",
                    "default": 15
                },
                "cookie": {
                    "type": "string",
                    "description": "Session cookie string for authenticated scans (e.g. 'PHPSESSID=abc; security=low')"
                },
                "model_path": {
                    "type": "string",
                    "description": "Override the default model path. Leave blank to use the universal model."
                },
                "deep": {
                    "type": "boolean",
                    "description": "If true, follow internal links to discover more pages (up to max_pages). Useful for authenticated scans.",
                    "default": False
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to spider during a deep crawl (default 10).",
                    "default": 10
                }
            },
            "required": ["url", "vuln_type"]
        }
    },
    {
        "name": "crawl_target",
        "description": (
            "Fetch a web page and discover its injection points (forms, URL parameters, "
            "JSON API fields) using static HTML parsing and LLM analysis. "
            "Does NOT run any attack — useful for previewing what will be tested."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Target URL to crawl"
                },
                "cookie": {
                    "type": "string",
                    "description": "Session cookie for authenticated pages"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "list_models",
        "description": "List all available trained RL models for XSS and SQL injection testing.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "authenticated_scan",
        "description": (
            "Log in to a web application using a username and password, then run a full "
            "vulnerability scan on the authenticated session. Automatically discovers the "
            "login form, submits credentials, captures session cookies, and scans pages "
            "that are only accessible after login. Use this when the user provides credentials."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Target URL to scan after login (e.g. the main app URL or dashboard)."
                },
                "username": {
                    "type": "string",
                    "description": "Login username or email address."
                },
                "password": {
                    "type": "string",
                    "description": "Login password."
                },
                "login_url": {
                    "type": "string",
                    "description": "URL of the login page if different from the target URL. Leave blank to use the target URL."
                },
                "vuln_type": {
                    "type": "string",
                    "enum": ["all", "xss", "sqli", "cmdi", "ssti"],
                    "description": "Vulnerability type(s) to test. Defaults to 'all'."
                },
                "episodes": {
                    "type": "integer",
                    "description": "Episodes per injection point (default 15, max 50).",
                    "default": 15
                },
                "deep": {
                    "type": "boolean",
                    "description": "Follow internal links after login to discover more authenticated pages. Defaults to true for authenticated scans.",
                    "default": True
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Maximum pages to spider (default 10).",
                    "default": 10
                }
            },
            "required": ["url", "username", "password"]
        }
    }
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def tool_list_models() -> dict:
    """Return available model files."""
    models_dir = Path("models")
    found = {}

    for vuln, default_path in DEFAULT_MODELS.items():
        p = Path(default_path + ".zip")
        found[vuln] = {
            "default_path": default_path,
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1_048_576, 1) if p.exists() else None,
        }

    # Also list any extra model dirs (exclude those already covered by default_models)
    default_dirs = {Path(p).parent.name for p in DEFAULT_MODELS.values()}
    extra = []
    if models_dir.exists():
        for d in sorted(models_dir.iterdir()):
            if d.is_dir() and d.name not in default_dirs:
                extra.append(d.name)

    return {"default_models": found, "other_model_dirs": extra}


def tool_crawl_target(url: str, cookie: str = "", api_key: str = "") -> dict:
    """Run LLMCrawler and return discovered injection points."""
    print(f"  [Crawling] {url} ...", flush=True)
    cookies = _parse_cookies(cookie)

    crawler = LLMCrawler(api_key=api_key or None)
    points = crawler.crawl(url, cookies=cookies)

    return {
        "url": url,
        "injection_points_found": len(points),
        "injection_points": [
            {
                "url": p.url,
                "method": p.method,
                "parameter": p.parameter,
                "input_type": p.input_type,
                "description": p.description,
            }
            for p in points
        ],
    }


_print_lock = threading.Lock()

_ACTION_TO_FAMILY_MAP = {
    "xss": XSS_ACTION_TO_FAMILY,
    "sqli": SQLI_ACTION_TO_FAMILY,
    "cmdi": CMDI_ACTION_TO_FAMILY,
    "ssti": SSTI_ACTION_TO_FAMILY,
}


def _test_single_point(
    point,
    point_episodes: int,
    point_index: int,
    total: int,
    vuln_type: str,
    model_path: str,
    api_key: str,
    cookies: dict | None,
    crawled_pages: list[str] | None,
    action_to_family: dict,
) -> dict:
    """Test one injection point (thread-safe — owns all its resources)."""
    if _cancel_event.is_set():
        return None

    point_score = _score_injection_point(point, vuln_type)
    with _print_lock:
        print(
            f"  [Testing {point_index}/{total}] [{vuln_type.upper()}] "
            f"{point.method} {point.url} param={point.parameter!r} "
            f"(score={point_score}, eps={point_episodes})",
            flush=True,
        )

    # --- WAF pre-check + reflection probe (single request does both) ---
    try:
        waf_client = GenericHttpClient(
            point.url, cookies=cookies if cookies else None,
        )
        # Use a unique canary that doubles as WAF probe and reflection test
        _canary = "zQ7xR3pL9"
        probe_payload = f"<script>{_canary}</script>"
        probe_body, probe_status, _, _ = waf_client.send_payload(
            point, probe_payload
        )
        waf_blocked, waf_name = ResponseAnalyzer.detect_waf(
            probe_body, probe_status,
            waf_client.last_response_headers,
        )
        if waf_blocked:
            with _print_lock:
                print(f"    [WAF] param={point.parameter!r} blocked by {waf_name} -- skipping", flush=True)
            return {
                "_sort_index": point_index,
                "found_on_page": point.form_page_url or point.url,
                "injection_url": point.url,
                "method": point.method,
                "parameter": point.parameter,
                "input_type": point.input_type,
                "waf_blocked": True,
                "waf_name": waf_name,
                "vulnerable": False,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_steps": 0.0,
                "successful_payloads": [],
                "payload_log": [],
            }

        # Reflection pre-check: if the canary doesn't appear in the
        # response (or the verify page for stored XSS), this param
        # almost certainly can't be exploited for XSS.  Skip it to
        # save episodes and API calls.  Only applies to XSS scans and
        # non-stored points (stored XSS has a separate verify URL).
        if vuln_type == "xss" and not point.verify_url:
            canary_reflected = _canary in (probe_body or "")
            if not canary_reflected:
                with _print_lock:
                    print(
                        f"    [Skip] param={point.parameter!r} — canary not reflected, skipping",
                        flush=True,
                    )
                return {
                    "_sort_index": point_index,
                    "found_on_page": point.form_page_url or point.url,
                    "injection_url": point.url,
                    "method": point.method,
                    "parameter": point.parameter,
                    "input_type": point.input_type,
                    "skipped": True,
                    "skip_reason": "canary not reflected",
                    "vulnerable": False,
                    "success_rate": 0.0,
                    "mean_reward": 0.0,
                    "mean_steps": 0.0,
                    "successful_payloads": [],
                    "payload_log": [],
                }
    except Exception:
        pass  # WAF/reflection check failure is non-fatal

    if _cancel_event.is_set():
        return None

    try:
        env = make_dynamic_env(
            injection_point=point,
            vuln_type=vuln_type,
            max_steps=30,
            api_key=api_key or None,
            wrap_monitor=True,
            cookies=cookies if cookies else None,
            crawled_pages=crawled_pages,
        )
        # Each thread loads its own model (thread-safe, OS-cached after first load)
        model, algo = _load_model(model_path, env)

        # --- Adaptive session (per-thread LLM generator) ---
        # Only use LLM-assisted payloads on HIGH-priority points (score >= 60)
        # to conserve API calls.  LOW/MED points use pure RL (free).
        adaptive_session = None
        if api_key and action_to_family and point_score >= 60:
            try:
                llm_gen = LLMPayloadGenerator(api_key=api_key)
                adaptive_session = AdaptiveScanSession(
                    model=model,
                    vuln_type=vuln_type,
                    action_to_family=action_to_family,
                    llm_gen=llm_gen,
                )
            except Exception as e:
                logger.debug("Adaptive session init failed for %s: %s", point.parameter, e)

        # --- Run episodes with early stop/give-up ---
        def _progress(ep_num, total_eps, ep_reward, successes):
            with _print_lock:
                print(
                    f"      ep {ep_num}/{total_eps}  reward={ep_reward:+.1f}  "
                    f"hits={successes}",
                    end="\r", flush=True,
                )

        result = run_episodes(
            model, env, point_episodes, point, adaptive_session,
            time_limit=90.0,
            progress_callback=_progress,
        )
        result["_sort_index"] = point_index
        result["algo"] = algo
        result["found_on_page"] = point.form_page_url or point.url
        result["injection_url"] = point.url
        result["input_type"] = point.input_type
        result["how_to_reach"] = point.nav_hint or ""
        result["vulnerable"] = result.get("success_rate", 0) > 0
        env.close()

        sr = result.get("success_rate", 0)
        status = "VULNERABLE" if sr > 0 else "clean"
        early = " (early-stopped)" if result.get("early_stopped") else ""
        with _print_lock:
            # Clear the progress line, then print final result
            print(f"\r    [{point_index}/{total}] {point.parameter!r} -> {status} ({sr:.0%} success rate){early}          ", flush=True)

        return result

    except KeyboardInterrupt:
        _cancel_event.set()
        with _print_lock:
            print(f"\n  [Scan cancelled — stopping after current point]", flush=True)
        return None

    except Exception as e:
        logger.debug("Agent test error on %s param=%s: %s", point.url, point.parameter, e)
        with _print_lock:
            print(f"    [ERROR] Could not test parameter {point.parameter!r} -- skipping ({type(e).__name__})", flush=True)
        return {
            "_sort_index": point_index,
            "found_on_page": point.form_page_url or point.url,
            "injection_url": point.url,
            "method": point.method,
            "parameter": point.parameter,
            "input_type": point.input_type,
            "error": f"Could not test this injection point ({type(e).__name__}).",
            "vulnerable": False,
            "success_rate": 0.0,
        }


def _run_agent_on_points(
    injection_points: list,
    vuln_type: str,
    model_path: str,
    episodes: int,
    api_key: str,
    cookies: dict | None = None,
    crawled_pages: list[str] | None = None,
) -> tuple[list, bool]:
    """
    Run the RL agent against every injection point for a single vuln type.
    Runs sequentially to avoid thread-safety issues with model loading
    and stdout redirection, and to prevent overwhelming external targets.
    Returns (results_list, aborted).
    """
    prioritized = _prioritize_points(injection_points, episodes, vuln_type=vuln_type)
    total = len(prioritized)
    action_to_family = _ACTION_TO_FAMILY_MAP.get(vuln_type, {})

    all_results = []
    aborted = False

    for i, (point, point_episodes) in enumerate(prioritized, 1):
        if _cancel_event.is_set():
            aborted = True
            break
        result = _test_single_point(
            point, point_episodes, i, total,
            vuln_type, model_path, api_key, cookies,
            crawled_pages, action_to_family,
        )
        if result is not None:
            all_results.append(result)

    # Restore priority order
    all_results.sort(key=lambda r: r.get("_sort_index", 0))
    for r in all_results:
        r.pop("_sort_index", None)
    return all_results, aborted


def tool_scan_target(
    url: str,
    vuln_type: str,
    episodes: int = 15,
    cookie: str = "",
    model_path: str = "",
    api_key: str = "",
    deep: bool = False,
    max_pages: int = 10,
    extra_points: list | None = None,
) -> dict:
    """Full scan: crawl once → run RL agent(s) → return results.
    vuln_type='all' tests all four vulnerability types in one crawl.
    deep=True follows internal links to find more pages (up to max_pages).
    Press Ctrl+C at any time to abort and receive partial results.
    """
    episodes = max(1, min(episodes, 50))
    cookies = _parse_cookies(cookie)

    # Determine which vuln types to run
    types_to_scan = list(DEFAULT_MODELS.keys()) if vuln_type == "all" else [vuln_type]

    # Validate model availability up-front
    missing = []
    for vt in types_to_scan:
        mp = model_path if (model_path and vuln_type != "all") else DEFAULT_MODELS.get(vt, "")
        p = mp + ".zip" if not mp.endswith(".zip") else mp
        if not Path(p).exists() and not Path(mp).exists():
            missing.append(vt)
    if missing:
        return {"error": f"Model(s) not found for: {', '.join(missing)}. Run training first."}

    # Crawl once (shared across all vuln types)
    crawler = LLMCrawler(api_key=api_key or None)
    try:
        if deep:
            print(f"  [Deep-crawling] {url} (max {max_pages} pages) ...", flush=True)
            injection_points = crawler.deep_crawl(url, cookies=cookies, max_pages=max_pages)
        else:
            print(f"  [Crawling] {url} ...", flush=True)
            injection_points = crawler.crawl(url, cookies=cookies)
    except KeyboardInterrupt:
        print("\n  [Crawl aborted — Ctrl+C]\n", flush=True)
        return {"url": url, "vuln_type": vuln_type, "aborted": True, "results": []}

    # Dual crawl: if we're authenticated (cookies present), also crawl
    # unauthenticated to discover login/signup form params that vanish
    # after login (the page redirects to dashboard when already logged in).
    if cookies and not extra_points:
        print(f"  [Dual crawl] Crawling {url} (unauthenticated) to find login form params ...", flush=True)
        try:
            unauth_crawler = LLMCrawler(api_key=api_key or None)
            unauth_points = unauth_crawler.crawl(url, cookies={})
        except Exception:
            unauth_points = []
        if unauth_points:
            existing_keys = {(p.method, p.url, p.parameter) for p in injection_points}
            added = 0
            for pt in unauth_points:
                key = (pt.method, pt.url, pt.parameter)
                if key not in existing_keys:
                    injection_points.append(pt)
                    existing_keys.add(key)
                    added += 1
            if added:
                print(f"  [Dual crawl] +{added} login form injection point(s) added", flush=True)

    # Merge extra injection points (e.g. from authenticated_scan's unauthenticated crawl)
    if extra_points:
        existing_keys = {(p.method, p.url, p.parameter) for p in injection_points}
        added = 0
        for pt in extra_points:
            key = (pt.method, pt.url, pt.parameter)
            if key not in existing_keys:
                injection_points.append(pt)
                existing_keys.add(key)
                added += 1
        if added:
            print(f"  [Merged] +{added} extra injection point(s) (login form params)", flush=True)

    if not injection_points:
        return {
            "url": url,
            "vuln_type": vuln_type,
            "error": "No injection points discovered. Try providing auth cookies or a more specific URL.",
            "results": [],
        }

    # Filter: header injection points are only relevant for XSS (stored
    # header → admin page).  SQLi/CMDi/SSTI via headers is extremely rare
    # and wastes time.  Keep them only for XSS scanning.
    form_points = [p for p in injection_points if p.input_type != "header"]
    header_points = [p for p in injection_points if p.input_type == "header"]

    print(
        f"  [Found] {len(injection_points)} injection point(s) "
        f"({len(form_points)} form + {len(header_points)} header) — "
        f"scanning for {', '.join(t.upper() for t in types_to_scan)} "
        f"({episodes} episodes each)...",
        flush=True,
    )
    print("  [Tip] Press Ctrl+C at any time to abort and get partial results.\n", flush=True)

    # Crawled pages list for stored XSS and env setup
    _crawled = getattr(crawler, "crawled_pages", [])

    # Run each vuln type sequentially (parallel injection points within each).
    # Running vuln types in parallel overwhelms local web servers.
    combined_results: dict[str, list] = {}
    global_aborted = False

    for vt in types_to_scan:
        if global_aborted:
            break
        mp = model_path if (model_path and vuln_type != "all") else DEFAULT_MODELS[vt]
        # XSS gets all points (form + header); others get form-only
        points = injection_points if vt == "xss" else form_points
        if not points:
            combined_results[vt] = []
            continue
        print(f"\n  --- [{vt.upper()}] ({len(points)} points) ---", flush=True)
        results, aborted = _run_agent_on_points(
            points, vt, mp, episodes, api_key,
            cookies=cookies, crawled_pages=_crawled,
        )
        combined_results[vt] = results
        if aborted:
            global_aborted = True

    # Stored XSS chain test (register -> login -> sweep)
    if "xss" in types_to_scan and not global_aborted and _crawled:
        print("\n  ============================================", flush=True)
        print("  STORED XSS CHAIN TEST (register -> login -> sweep)", flush=True)
        print("  ============================================", flush=True)
        chain_findings = run_stored_xss_chain_test(
            injection_points=injection_points,
            crawled_pages=_crawled,
            cookies=cookies,
            base_url=url,
        )
        if chain_findings:
            xss_results = combined_results.get("xss", [])
            xss_results.extend(chain_findings)
            combined_results["xss"] = xss_results
            print(f"  [Chain] Found {len(chain_findings)} stored XSS via registration chain!", flush=True)
        else:
            print("  [Chain] No stored XSS found via registration chain.", flush=True)

    # --- Heuristic security checks (CSRF, access control, headers, etc.) ---
    heuristic_findings = []
    if not global_aborted and _crawled:
        print("\n  ============================================", flush=True)
        print("  HEURISTIC SECURITY CHECKS", flush=True)
        print("  ============================================", flush=True)
        try:
            heuristic_session = requests.Session()
            if cookies:
                for k, v in cookies.items():
                    heuristic_session.cookies.set(k, v)
            # Pre-fetch pages so heuristic checks have HTML to analyze
            page_html_map: dict[str, str] = {}
            for page_url in _crawled[:15]:
                try:
                    resp = heuristic_session.get(page_url, timeout=8, allow_redirects=True)
                    page_html_map[page_url] = resp.text
                except Exception:
                    pass
            heuristic_findings = run_all_heuristic_checks(
                session=heuristic_session,
                injection_points=injection_points,
                crawled_urls=_crawled,
                crawled_html=page_html_map,
                base_url=url,
                cookies=cookies,
            )
            if heuristic_findings:
                h_high = sum(1 for f in heuristic_findings if f.get("severity") == "High")
                h_med = sum(1 for f in heuristic_findings if f.get("severity") == "Medium")
                h_low = len(heuristic_findings) - h_high - h_med
                print(f"  [Heuristic] {len(heuristic_findings)} issue(s) found "
                      f"(High={h_high}, Medium={h_med}, Low={h_low})", flush=True)
                for f in heuristic_findings:
                    sev = f.get("severity", "Info")
                    print(f"    [{sev}] {f.get('title', 'Unknown')} - {f.get('url', '')}", flush=True)
            else:
                print("  [Heuristic] No additional issues found.", flush=True)
        except Exception as e:
            logger.debug("Heuristic checks failed: %s", e)
            print(f"  [Heuristic] Check failed: {e}", flush=True)

    # Build summary
    summary_lines = []
    total_vulnerable = 0
    for vt, results in combined_results.items():
        vuln_count = sum(1 for r in results if r.get("vulnerable"))
        total_vulnerable += vuln_count
        summary_lines.append(
            f"{vt.upper()}: {vuln_count}/{len(results)} injection points vulnerable"
        )
    if heuristic_findings:
        summary_lines.append(f"HEURISTIC: {len(heuristic_findings)} issue(s)")

    # Generate page-specific remediation advice for each vulnerable vuln type
    remediation_by_type: dict[str, str] = {}
    if api_key and total_vulnerable > 0:
        print("\n  [Generating page-specific remediation advice...]", flush=True)
        for vt, results in combined_results.items():
            vuln_results = [r for r in results if r.get("vulnerable")]
            if not vuln_results:
                continue
            mapped = [
                {**r, "url": r.get("injection_url", r.get("url", ""))}
                for r in vuln_results
            ]
            advice = _generate_remediation(vt, mapped, api_key, cookies=cookies)
            remediation_by_type[vt] = advice
            print(f"\n  HOW TO FIX [{vt.upper()}]")
            print("  " + "-" * 48)
            print(advice)
            print()

    return {
        "url": url,
        "vuln_type": vuln_type,
        "aborted": global_aborted,
        "injection_points_tested": len(injection_points),
        "total_vulnerable": total_vulnerable,
        "results_by_type": combined_results,
        "summary": " | ".join(summary_lines) if summary_lines else "No results.",
        "remediation_by_type": remediation_by_type,
        "heuristic_checks": heuristic_findings,
    }


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------


def tool_authenticated_scan(
    url: str,
    username: str,
    password: str,
    login_url: str = "",
    vuln_type: str = "all",
    episodes: int = 15,
    api_key: str = "",
    deep: bool = True,
    max_pages: int = 10,
) -> dict:
    """
    Log in with provided credentials, then run a full vulnerability scan
    on the authenticated session. By default performs a deep crawl (follows
    internal links) to discover pages only accessible after login.
    Returns login status + all scan results.
    """
    target_login = login_url or url
    print(f"  [Auth] Logging in at {target_login} as '{username}' ...", flush=True)

    cookies, success, msg, _session = _do_authenticate(target_login, username, password)
    print(f"  [Auth] {msg}", flush=True)

    if not cookies:
        return {
            "authenticated": False,
            "auth_message": msg,
            "error": "Could not authenticate — scan aborted.",
        }

    print(f"  [Auth] Session established ({len(cookies)} cookie(s) obtained).", flush=True)
    if deep:
        print(
            f"  [Auth] Deep crawl enabled — will follow internal links "
            f"(up to {max_pages} pages) to discover authenticated pages.",
            flush=True,
        )

    # ── Dual crawl: also crawl unauthenticated to discover login form params ──
    # When logged in, the login page typically redirects → login form params
    # are invisible. Crawl the login page without cookies to find them.
    print(f"  [Auth] Crawling {target_login} (unauthenticated) to find login form params ...", flush=True)
    unauth_crawler = LLMCrawler(api_key=api_key or None)
    try:
        unauth_points = unauth_crawler.crawl(target_login, cookies={})
    except Exception:
        unauth_points = []
    if unauth_points:
        print(f"  [Auth] +{len(unauth_points)} login form injection point(s) discovered", flush=True)

    # Encode cookies as a string for the existing scan pipeline
    cookie_str = "; ".join(f"{k}={v}" for k, v in cookies.items())

    scan_result = tool_scan_target(
        url=url,
        vuln_type=vuln_type,
        episodes=episodes,
        cookie=cookie_str,
        api_key=api_key,
        deep=deep,
        max_pages=max_pages,
        extra_points=unauth_points,
    )

    scan_result["authenticated"] = success
    scan_result["auth_message"] = msg
    scan_result["session_cookies"] = list(cookies.keys())  # names only, not values
    return scan_result


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, inputs: dict, api_key: str) -> Any:
    """Route a tool call to the appropriate implementation."""
    if name == "list_models":
        return tool_list_models()

    if name == "crawl_target":
        return tool_crawl_target(
            url=inputs["url"],
            cookie=inputs.get("cookie", ""),
            api_key=api_key,
        )

    if name == "scan_target":
        return tool_scan_target(
            url=inputs["url"],
            vuln_type=inputs["vuln_type"],
            episodes=inputs.get("episodes", 15),
            cookie=inputs.get("cookie", ""),
            model_path=inputs.get("model_path", ""),
            api_key=api_key,
            deep=inputs.get("deep", False),
            max_pages=inputs.get("max_pages", 10),
        )

    if name == "authenticated_scan":
        return tool_authenticated_scan(
            url=inputs["url"],
            username=inputs["username"],
            password=inputs["password"],
            login_url=inputs.get("login_url", ""),
            vuln_type=inputs.get("vuln_type", "all"),
            episodes=inputs.get("episodes", 15),
            api_key=api_key,
            deep=inputs.get("deep", True),
            max_pages=inputs.get("max_pages", 10),
        )

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# Session history persistence
# ---------------------------------------------------------------------------

_BLOCK_ALLOWED_FIELDS = {
    "text":        {"type", "text"},
    "tool_use":    {"type", "id", "name", "input"},
    "tool_result": {"type", "tool_use_id", "content", "is_error"},
    "image":       {"type", "source"},
}


def _clean_block(block: dict) -> dict:
    """Strip API-rejected internal fields (e.g. parsed_output) from a content block."""
    block_type = block.get("type", "")
    allowed = _BLOCK_ALLOWED_FIELDS.get(block_type)
    if allowed:
        return {k: v for k, v in block.items() if k in allowed}
    return block


def _serialize_history(history: list) -> list:
    """Convert Anthropic ContentBlock objects to JSON-serializable dicts."""
    result = []
    for msg in history:
        content = msg["content"]
        if isinstance(content, str):
            result.append({"role": msg["role"], "content": content})
        elif isinstance(content, list):
            serialized = []
            for block in content:
                if isinstance(block, dict):
                    serialized.append(_clean_block(block))
                elif hasattr(block, "model_dump"):
                    serialized.append(_clean_block(block.model_dump()))
                else:
                    serialized.append({"type": "text", "text": str(block)})
            result.append({"role": msg["role"], "content": serialized})
    return result


def _save_history(history: list):
    """Persist conversation history to disk."""
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(_serialize_history(history), f, default=str)
    except Exception as e:
        logger.debug(f"Failed to save history: {e}")


def _load_history() -> list:
    """Load previously saved conversation history from disk."""
    try:
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE) as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"Failed to load history: {e}")
    return []


# ---------------------------------------------------------------------------
# Slash command implementations (no API key required)
# ---------------------------------------------------------------------------

_QUIT_WORDS = frozenset({
    "quit", "exit", "q", "bye", "goodbye", ":q", r"\q",
    "exit()", "q!", "wq",
})
_SCAN_TOOLS = frozenset({"scan_target", "authenticated_scan"})

# Module-level session state — persists last scan result and verbose flag
# across slash commands and the entire chat loop lifetime.
_session: dict = {
    "last_scan":     None,   # dict  — last result from scan tools
    "last_scan_url": "",     # str   — URL of the most recent scan
    "verbose":       False,  # bool  — /verbose toggle
    "scans_run":     0,      # int   — total scans since process start
}

# Set this event to cancel any running scan gracefully (partial results are returned).
# Raised by Ctrl+C or the /cancel command.
_cancel_event = threading.Event()


def _cmd_help(arg: str = ""):
    """
    /help              — full command reference
    /help scan         — detail on /scan
    /help auth         — detail on /auth
    /help crawl        — detail on /crawl
    /help report       — detail on /report and /export
    /help vulns        — detail on /vulns
    /help targets      — detail on /targets
    /help session      — session management commands
    /help admin        — admin / error log commands
    /help ai           — example natural-language prompts for Claude
    /help examples     — complete copy-paste examples for every command
    """
    topic = arg.strip().lower()

    W = 68  # total width

    def _header(title):
        _console.print()
        _console.print(Panel(
            f"[bold bright_cyan]{title}[/bold bright_cyan]",
            border_style="bright_cyan",
            padding=(0, 2),
        ))

    def _section(title):
        _console.print(f"\n  [bold bright_yellow]▸ {title}[/bold bright_yellow]")
        _console.print("  [dim]" + "─" * (W - 2) + "[/dim]")

    def _row(cmd, desc):
        _console.print(f"  [cyan]{cmd:<36}[/cyan] [white]{desc}[/white]")

    def _example(text):
        _console.print(f"    [dim]›[/dim] [green]{text}[/green]")

    # ------------------------------------------------------------------
    # Topic: scan
    # ------------------------------------------------------------------
    if topic == "scan":
        _header("/scan — Direct vulnerability scan")
        print("""
  Crawls a URL, discovers all injection points, and runs the trained
  RL agent against each one. No API key required.

  SYNTAX
    /scan <url> [vuln] [episodes=N]

  ARGUMENTS
    url         Target page to scan (required)
    vuln        xss | sqli | cmdi | ssti | all   (default: all)
    episodes=N  Episodes per injection point      (default: 20)

  EXAMPLES""")
        _example("/scan http://localhost:8080 sqli")
        _example("/scan http://localhost:8080 xss  episodes=30")
        _example("/scan http://localhost:3000 xss")
        _example("/scan http://localhost:8081 sqli episodes=10")
        print("""
  WHAT YOU GET
    A results table showing each parameter tested, success rate,
    average reward, and the payloads that worked.
    Run /report afterwards for a full formatted breakdown.
    Run /export to save results as JSON.\n""")
        return

    # ------------------------------------------------------------------
    # Topic: auth
    # ------------------------------------------------------------------
    if topic == "auth":
        _header("/auth — Authenticated scan (auto-login)")
        print("""
  Logs in with your credentials, captures the session cookie,
  then scans all discovered pages as an authenticated user.

  SYNTAX
    /auth <url> <username> <password> [vuln] [episodes=N]

  ARGUMENTS
    url         Login page or application root (required)
    username    Login username (required)
    password    Login password (required)
    vuln        xss | sqli | cmdi | ssti | all   (default: all)
    episodes=N  Episodes per injection point      (default: 20)

  EXAMPLES""")
        _example("/auth http://localhost:8080 admin password")
        _example("/auth http://localhost:8080 admin password sqli")
        _example("/auth http://localhost:8080 admin password xss episodes=30")
        _example("/auth http://localhost:3000 admin@juice-sh.op password xss")
        print("""
  WHAT YOU GET
    Same results table as /scan, but all injection points found
    behind the login wall are included.\n""")
        return

    # ------------------------------------------------------------------
    # Topic: crawl
    # ------------------------------------------------------------------
    if topic == "crawl":
        _header("/crawl — Discover injection points")
        print("""
  Fetches a page and lists all forms and URL parameters it finds.
  Use this to preview what /scan will test before committing to a run.

  SYNTAX
    /crawl <url> [cookie=<value>]

  ARGUMENTS
    url           Page to crawl (required)
    cookie=<val>  Session cookie for authenticated crawl (optional)

  EXAMPLES""")
        _example("/crawl http://localhost:8080")
        _example("/crawl http://localhost:8080 cookie=PHPSESSID=abc123")
        _example("/crawl http://localhost:3000/rest/products/search")
        print("""
  WHAT YOU GET
    A list of discovered injection points:
      [1] GET  http://localhost:8080/vulnerabilities/sqli/  param=id
      [2] POST http://localhost:8080/login  param=username
      [3] POST http://localhost:8080/login  param=password\n""")
        return

    # ------------------------------------------------------------------
    # Topic: report / export
    # ------------------------------------------------------------------
    if topic in ("report", "export", "payloads", "explain"):
        _header("/report, /explain, /payloads, and /export — View and save results")
        print("""
  /report
    Prints a formatted table of the last scan's results.
    Shows each injection point, success rate, average reward,
    and the top payloads that worked.

  /explain [param]
    Detailed breakdown of each vulnerability found:
      - WHAT was found (type, subtype, parameter, URL)
      - HOW it was found (crawl discovery, priority score, episodes)
      - Successful payload(s) and the full HTTP request
      - Server response snippet showing the actual evidence
      - WHY the payload worked (technical explanation per payload family)
      - Impact evidence (data leaked, auth bypassed, etc.)
      - HOW TO FIX (LLM-generated or static remediation advice)

    Filters:
      /explain               — explain all vulnerable parameters
      /explain <param>       — explain only a specific parameter

  /payloads [filter]
    Shows EVERY payload attempted during the last scan, grouped
    by parameter.  Each entry shows the episode, step, reward,
    success/fail status, whether it was reflected, and the
    response snippet for successful hits.

    Filters:
      /payloads              — all payloads for all parameters
      /payloads success      — only successful payloads
      /payloads fail         — only failed payloads
      /payloads <param>      — only payloads for a specific parameter

  /export [filename]
    Saves the last scan to a JSON file.
    Default filename: results/assistant_scan_YYYYMMDD_HHMMSS.json

  EXAMPLES""")
        _example("/report")
        _example("/explain")
        _example("/explain username")
        _example("/payloads")
        _example("/payloads success")
        _example("/export my_results.json")
        print()
        return

    # ------------------------------------------------------------------
    # Topic: vulns
    # ------------------------------------------------------------------
    if topic == "vulns":
        _header("/vulns — Vulnerability type reference")
        print("""
  Explains each supported vulnerability type and what the agent
  can do against it.

  SYNTAX
    /vulns              Show all four types
    /vulns <type>       Detail on one type: sqli | xss | cmdi | ssti

  EXAMPLES""")
        _example("/vulns")
        _example("/vulns sqli")
        _example("/vulns xss")
        print("""
  SUPPORTED TYPES
    sqli   SQL Injection       — DVWA, Juice Shop, WebGoat
    xss    Cross-Site Scripting — DVWA (low/med/high), Juice Shop, WebGoat
    cmdi   OS Command Injection — DVWA (low/med/high)
    ssti   Template Injection   — Juice Shop\n""")
        return

    # ------------------------------------------------------------------
    # Topic: targets
    # ------------------------------------------------------------------
    if topic == "targets":
        _header("/targets — Known test applications")
        print("""
  Lists the three preconfigured target applications with their URLs,
  supported vulnerability types, and a ready-to-run scan command.

  SYNTAX
    /targets

  WHAT YOU GET
    DVWA       http://localhost:8080  — SQLi, XSS, CMDi
    Juice Shop http://localhost:3000  — SQLi, XSS, SSTI
    WebGoat    http://localhost:8081  — SQLi, XSS

  NOTE
    Containers must be running first:
      docker compose up -d\n""")
        return

    # ------------------------------------------------------------------
    # Topic: session
    # ------------------------------------------------------------------
    if topic == "session":
        _header("Session management commands")
        _section("History")
        _row("/history [n]",     "Show last n conversation turns (default: 5)")
        _row("/history full",    "Show all turns including raw tool call blocks")
        _section("Saving")
        _row("/save [file]",     "Save chat transcript to a text file")
        _row("/export [file]",   "Save last scan results to JSON")
        _section("State")
        _row("/cancel",          "Stop the current scan (returns partial results)")
        _row("/clear  /reset",   "Wipe conversation history")
        _row("/verbose",         "Toggle debug logging on/off")
        _row("/setkey <key>",    "Set Anthropic API key for this session")
        _row("/status",          "Show key status, scans run, verbose mode")
        _row("/models",          "List trained RL models on disk")
        _section("Exit")
        _row("/quit  /exit  bye","Exit the assistant")
        print()
        return

    # ------------------------------------------------------------------
    # Topic: admin
    # ------------------------------------------------------------------
    if topic == "admin":
        _header("/admin — Error log viewer  (password-protected)")
        print("""
  Requires ASSISTANT_ADMIN_PASSWORD set in your .env file.

  SYNTAX
    /admin <password> list
    /admin <password> today [lines=N]
    /admin <password> date=YYYY-MM-DD
    /admin <password> lines=N

  COMMANDS""")
        _row("list",                "List all daily log files with entry count + size")
        _row("today [lines=N]",     "Show today's errors (default: last 50 lines)")
        _row("date=YYYY-MM-DD",     "View errors from a specific past date")
        _row("lines=N",             "Show last N lines of the current log")
        print("""
  EXAMPLES""")
        _example("/admin secret list")
        _example("/admin secret today")
        _example("/admin secret today lines=100")
        _example("/admin secret date=2026-03-14")
        print("""
  LOG LOCATION
    logs/errors/errors_YYYY-MM-DD.log  (rotates daily, 90 days kept)\n""")
        return

    # ------------------------------------------------------------------
    # Topic: examples
    # ------------------------------------------------------------------
    if topic == "examples":
        _header("QUICK-START EXAMPLES — How to use every command")
        print("""
  ── SCAN ALL VULNERABILITIES ────────────────────────────────────────""")
        _example("/scan http://localhost:8080 all")
        _example("  → tests SQLi, XSS, CMDi and SSTI on every injection point")
        print("""
  ── SCAN ONE VULNERABILITY TYPE ─────────────────────────────────────""")
        _example("/scan http://localhost:8080 sqli")
        _example("/scan http://localhost:3000 xss  episodes=30")
        _example("/scan http://localhost:8081 cmdi episodes=10")
        _example("/scan http://localhost:3000 ssti")
        print("""
  ── PREVIEW INJECTION POINTS BEFORE SCANNING ────────────────────────""")
        _example("/crawl http://localhost:8080")
        _example("/crawl http://localhost:3000/rest/products/search")
        _example("/crawl http://localhost:8080 cookie=PHPSESSID=abc123")
        print("""
  ── AUTHENTICATED SCAN (login first, then scan) ──────────────────────""")
        _example("/auth http://localhost:8080 admin password")
        _example("/auth http://localhost:8080 admin password all")
        _example("/auth http://localhost:3000 admin@juice-sh.op password xss episodes=30")
        print("""
  ── VIEW AND SAVE RESULTS ────────────────────────────────────────────""")
        _example("/report                    ← formatted results table")
        _example("/export                    ← save to auto-named JSON file")
        _example("/export my_results.json   ← save to specific file")
        print("""
  ── EXPLORE TARGETS AND VULNERABILITY TYPES ──────────────────────────""")
        _example("/targets                   ← list DVWA, Juice Shop, WebGoat URLs")
        _example("/vulns                     ← explain all four vuln types")
        _example("/vulns sqli                ← detail on SQL Injection only")
        _example("/vulns xss                 ← detail on Cross-Site Scripting only")
        print("""
  ── SESSION AND MODELS ───────────────────────────────────────────────""")
        _example("/models                    ← list trained RL models on disk")
        _example("/status                    ← show API key, scans run, verbose flag")
        _example("/history 10               ← show last 10 conversation turns")
        _example("/save transcript.txt      ← save chat to file")
        _example("/setkey sk-ant-...        ← set API key for this session")
        _example("/verbose                   ← toggle debug logging on/off")
        _example("/cancel                    ← stop the current scan, keep partial results")
        _example("/clear                     ← wipe conversation history")
        print("""
  ── ADMIN (requires ASSISTANT_ADMIN_PASSWORD in .env) ────────────────""")
        _example("/admin secret list")
        _example("/admin secret today")
        _example("/admin secret today lines=100")
        _example("/admin secret date=2026-03-14")
        print("""
  ── AI NATURAL-LANGUAGE PROMPTS (require API key) ────────────────────""")
        _example("scan http://localhost:8080 for all vulnerabilities")
        _example("login to http://localhost:8080 as admin/password and scan for SQLi")
        _example("what payloads worked?")
        _example("which parameters were vulnerable?")
        _example("explain why ' OR '1'='1 works")
        print()
        return

    # ------------------------------------------------------------------
    # Topic: ai
    # ------------------------------------------------------------------
    if topic == "ai":
        _header("AI-powered natural language prompts  (require API key)")
        print("""
  With an API key set, you can describe what you want in plain English.
  Claude will choose the right tool and run it for you.

  SCANNING""")
        _example("scan http://localhost:8080 for SQL injection")
        _example("scan http://localhost:3000 for XSS")
        _example("test http://localhost:8080 for all vulnerability types")
        print("\n  AUTHENTICATED SCANNING")
        _example("login to http://localhost:8080 as admin/password and scan for SQLi")
        _example("scan http://localhost:3000 for XSS using cookie 'token=abc123'")
        print("\n  RESULTS & ANALYSIS")
        _example("what payloads worked?")
        _example("which parameters were vulnerable?")
        _example("explain why ' OR '1'='1 works")
        _example("what could an attacker do with this SQL injection?")
        print("\n  MODELS")
        _example("what models do we have trained?")
        _example("which model should I use for scanning Juice Shop?")
        print("""
  SET KEY
    /setkey sk-ant-...   (or set ANTHROPIC_API_KEY in your .env file)\n""")
        return

    # ------------------------------------------------------------------
    # Default: full overview
    # ------------------------------------------------------------------
    _header("AI PENTEST ASSISTANT — Command Reference  (/help <topic> for detail)")

    _section("HOW TO USE THIS TOOL")
    print(f"  {'A) Interactive Assistant  ← YOU ARE HERE':}")
    print(f"     Start:   python assistant.py")
    print(f"     Then type commands below (no API key needed for scanning)")
    print()
    print(f"  B) Train an RL Agent")
    print(f"     python -m agents.train --vuln sqli --algo dqn --timesteps 50000")
    print(f"     python -m agents.train --vuln xss  --algo ppo --timesteps 50000")
    print(f"     python -m agents.train --vuln both --algo dqn --timesteps 50000")
    print()
    print(f"  C) Evaluate a Trained Model")
    print(f"     python -m agents.evaluate --model models/<run>/sqli_dqn_final \\")
    print(f"       --vuln sqli --episodes 100 --include-random")
    print()
    print(f"  Targets (docker compose up -d to start):")
    print(f"    DVWA        http://localhost:8080  — SQLi, XSS, CMDi")
    print(f"    Juice Shop  http://localhost:3000  — SQLi, XSS, SSTI")
    print(f"    WebGoat     http://localhost:8081  — SQLi, XSS")

    _section("SCANNING  (no API key needed)")
    _row("/scan <url> [vuln] [episodes=N]",    "Scan a URL for vulnerabilities")
    _row("/auth <url> <user> <pass> [vuln]",   "Log in then scan as authenticated user")
    _row("/crawl <url> [cookie=<v>]",          "Preview all injection points on a page")
    print()
    _example("/scan http://localhost:8080 sqli")
    _example("/auth http://localhost:8080 admin password xss")
    _example("/crawl http://localhost:3000")

    _section("RESULTS")
    _row("/report",                            "Print last scan results as a table")
    _row("/explain [param]",                   "Detailed vulnerability breakdown: how/why/fix")
    _row("/payloads [param|success|fail]",     "Show all payloads tried + results")
    _row("/export [file]",                     "Save last scan to JSON")
    print()
    _example("/report")
    _example("/explain               ← full explanation of all vulnerabilities")
    _example("/explain username      ← explain only 'username' parameter")
    _example("/payloads              ← all payloads, grouped by parameter")
    _example("/payloads success      ← only successful payloads")
    _example("/payloads comment      ← only payloads for 'comment' param")
    _example("/export my_results.json")

    _section("INFORMATION")
    _row("/targets",                           "List DVWA, Juice Shop, WebGoat URLs")
    _row("/vulns [xss|sqli|cmdi|ssti]",        "Explain a vulnerability type")
    _row("/models",                            "List trained RL models on disk")
    _row("/status",                            "Session info: key, scans run, verbose")
    print()
    _example("/targets")
    _example("/vulns sqli")

    _section("SESSION")
    _row("/history [n]",                       "Show last n conversation turns")
    _row("/save [file]",                       "Save chat transcript to file")
    _row("/clear",                             "Wipe conversation history")
    _row("/setkey <key>",                      "Set API key for this session")
    _row("/verbose",                           "Toggle debug logging")
    _row("/cancel",                            "Stop the current scan (returns partial results)")
    _row("/quit  (or: exit / bye)",            "Exit")

    _section("ADMIN  (requires password in .env)")
    _row("/admin <pw> list",                   "List daily error log files")
    _row("/admin <pw> today [lines=N]",        "View today's errors")
    _row("/admin <pw> date=YYYY-MM-DD",        "View errors from a past date")

    _section("AI PROMPTS  (require API key — /setkey sk-ant-...)")
    _example("scan http://localhost:8080 for SQL injection")
    _example("login to http://localhost:8080 as admin/password and scan")
    _example("what payloads worked?  /  what could an attacker do with this?")

    _section("QUICK-START — scan all vulnerabilities on a target")
    _example("/scan http://localhost:8080 all            ← test SQLi + XSS + CMDi + SSTI")
    _example("/auth http://localhost:8080 admin pass all ← login first, then scan all")
    _example("/crawl http://localhost:8080               ← preview injection points first")

    print()
    print("  Type /help <topic> for full detail on any section:")
    print("  Topics: scan  auth  crawl  report  vulns  targets  session  admin  ai  examples")
    print()
    print("  /help examples   — complete copy-paste examples for every command")
    print("=" * W + "\n")


def _cmd_models():
    result = tool_list_models()
    _console.print("\n  [bold bright_yellow]▸ Available Models[/bold bright_yellow]")
    _console.print("  [dim]" + "─" * 60 + "[/dim]")
    for vuln, info in result["default_models"].items():
        if info["exists"]:
            status = "[green]✓ OK     [/green]"
        else:
            status = "[red]✗ MISSING[/red]"
        size = f"{info['size_mb']} MB" if info["size_mb"] else "—"
        _console.print(
            f"  [bold cyan]{vuln.upper():<6}[/bold cyan]  {status}  "
            f"[dim]{info['default_path']}[/dim]  [white]({size})[/white]"
        )
    if result["other_model_dirs"]:
        _console.print("\n  [bold]Other model directories:[/bold]")
        for d in result["other_model_dirs"]:
            _console.print(f"    [dim]models/[/dim][cyan]{d}/[/cyan]")
    print()


def _cmd_history(history: list, arg: str = ""):
    """
    /history [n]    — show last n plain-text turns (default 5)
    /history full   — show every message including tool_use/tool_result blocks
    """
    full_mode = arg.strip().lower() == "full"
    n = 5
    if not full_mode and arg.strip().isdigit():
        n = int(arg.strip())

    if full_mode:
        if not history:
            print("  [No conversation history yet]\n")
            return
        print(f"\n  Full history — {len(history)} message(s):")
        print("  " + "-" * 56)
        for idx, msg in enumerate(history):
            role = msg["role"].upper()
            content = msg["content"]
            if isinstance(content, str):
                snippet = content[:300].replace("\n", " ")
                if len(content) > 300:
                    snippet += "…"
                print(f"  [{idx}] {role}: {snippet}")
            elif isinstance(content, list):
                for block in content:
                    btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", "?")
                    if btype == "text":
                        text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                        snippet = text[:200].replace("\n", " ")
                        print(f"  [{idx}] {role} [text]: {snippet}")
                    elif btype == "tool_use":
                        name = block.get("name", "?") if isinstance(block, dict) else getattr(block, "name", "?")
                        inp  = block.get("input", {}) if isinstance(block, dict) else getattr(block, "input", {})
                        print(f"  [{idx}] {role} [tool_use: {name}] input={json.dumps(inp, default=str)[:120]}")
                    elif btype == "tool_result":
                        tid = block.get("tool_use_id", "?") if isinstance(block, dict) else "?"
                        raw = block.get("content", "") if isinstance(block, dict) else ""
                        snippet = str(raw)[:120].replace("\n", " ")
                        print(f"  [{idx}] {role} [tool_result id={tid}]: {snippet}")
                    else:
                        print(f"  [{idx}] {role} [block type={btype!r}]")
        print()
        return

    # Normal mode: plain-text turns only
    readable = [
        m for m in history
        if isinstance(m["content"], str) and m["role"] in ("user", "assistant")
    ]
    recent = readable[-(n * 2):]
    if not recent:
        print("  [No conversation history yet]\n")
        return
    print(f"\n  Last {len(recent)} messages:")
    print("  " + "-" * 54)
    for msg in recent:
        label = "You      " if msg["role"] == "user" else "Assistant"
        snippet = msg["content"][:220].replace("\n", " ")
        if len(msg["content"]) > 220:
            snippet += "…"
        print(f"  {label}: {snippet}")
    print()


def _cmd_save(history: list, filename: str = ""):
    if not filename:
        filename = f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    lines = [
        f"AI Pentest Assistant — Transcript ({datetime.now().isoformat()})\n",
        "=" * 60 + "\n\n",
    ]
    for msg in history:
        content = msg["content"]
        label = "You" if msg["role"] == "user" else "Assistant"
        if isinstance(content, str):
            lines.append(f"{label}: {content}\n\n")
        elif isinstance(content, list):
            for block in content:
                text = None
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block["text"]
                elif hasattr(block, "type") and block.type == "text":
                    text = block.text
                if text:
                    lines.append(f"{label}: {text}\n\n")
    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  [Transcript saved to {filename}]\n")
    except Exception as e:
        _elog.error("Transcript save failed | file=%s | %s: %s", filename, type(e).__name__, e)
        print(f"  [Could not save transcript — check the file path and permissions.]\n")


def _cmd_status(history: list, api_key: str, scans_run: int):
    user_turns = sum(
        1 for m in history
        if m["role"] == "user" and isinstance(m["content"], str)
    )
    last_url = _session.get("last_scan_url") or "none"
    if api_key:
        api_status = f"[green]✓ SET[/green] [dim]({ASSISTANT_MODEL})[/dim]"
    else:
        api_status = "[red]✗ NOT SET[/red] [dim]— use[/dim] [cyan]/setkey sk-ant-...[/cyan]"
    verbose_status = "[green]ON[/green]" if _session.get('verbose') else "[dim]OFF[/dim]"

    _console.print("\n  [bold bright_yellow]▸ Session Status[/bold bright_yellow]")
    _console.print("  [dim]" + "─" * 60 + "[/dim]")
    _console.print(f"  [bold]API key   [/bold] : {api_status}")
    _console.print(f"  [bold]Messages  [/bold] : [cyan]{user_turns}[/cyan] user turn(s)")
    _console.print(f"  [bold]Scans run [/bold] : [cyan]{scans_run}[/cyan]")
    _console.print(f"  [bold]Last scan [/bold] : [white]{last_url}[/white]")
    _console.print(f"  [bold]Verbose   [/bold] : {verbose_status}")
    _console.print(f"  [bold]History   [/bold] : [dim]{HISTORY_FILE}[/dim]")
    print()


# ---------------------------------------------------------------------------
# Known targets and vulnerability info (used by /targets and /vulns)
# ---------------------------------------------------------------------------

_KNOWN_TARGETS = [
    {
        "name": "DVWA",
        "url": "http://localhost:8080",
        "container": "dvwa_target",
        "default_creds": "admin / password",
        "notes": "Set security to 'low' before scanning. Login: /login.php",
    },
    {
        "name": "Juice Shop",
        "url": "http://localhost:3000",
        "container": "juiceshop_target",
        "default_creds": "admin@juice-sh.op / admin123",
        "notes": "REST API + SPA. Uses JWT tokens.",
    },
    {
        "name": "WebGoat",
        "url": "http://localhost:8081",
        "container": "webgoat_target",
        "default_creds": "webgoat / webgoat",
        "notes": "Java/Spring. Uses /WebGoat/ prefix. CSRF exempt for XHR.",
    },
]

_VULN_DESCRIPTIONS = {
    "xss": {
        "full": "Cross-Site Scripting (XSS)",
        "cwe": "CWE-79",
        "owasp": "A03:2021 — Injection",
        "summary": (
            "Attacker injects malicious scripts into pages viewed by other users. "
            "Reflected XSS bounces the payload off the server in a single request; "
            "stored XSS persists in the database; DOM-based XSS runs entirely in the browser."
        ),
        "model": "models/universal_xss_dqn/xss_dqn_final",
        "action_space": "12 actions (script, img, svg, event-handler families)",
    },
    "sqli": {
        "full": "SQL Injection (SQLi)",
        "cwe": "CWE-89",
        "owasp": "A03:2021 — Injection",
        "summary": (
            "Attacker inserts SQL syntax into user-supplied fields to manipulate the "
            "backend database query — bypassing authentication, dumping data, or modifying records."
        ),
        "model": "models/universal_sqli_dqn/sqli_dqn_final",
        "action_space": "10 actions (OR-true, comment, UNION, tautology families)",
    },
    "cmdi": {
        "full": "OS Command Injection (CMDi)",
        "cwe": "CWE-78",
        "owasp": "A03:2021 — Injection",
        "summary": (
            "Attacker appends shell operators (;, |, &&) to a user-supplied value that "
            "is passed directly to a system() or exec() call, running arbitrary OS commands."
        ),
        "model": "models/cmdi_dqn_curriculum_high_.../cmdi_dqn_high_5000_steps",
        "action_space": "10 actions (semicolon, pipe, logical-op, newline families)",
    },
    "ssti": {
        "full": "Server-Side Template Injection (SSTI)",
        "cwe": "CWE-94",
        "owasp": "A03:2021 — Injection",
        "summary": (
            "Attacker injects template syntax ({{ 7*7 }}, #{7*7}) into fields rendered "
            "by a templating engine (Pug, Twig, Jinja2), enabling code execution or "
            "sensitive data disclosure."
        ),
        "model": "models/ssti_dqn_low_20260228_175703/ssti_dqn_final",
        "action_space": "10 actions (pug, twig, jinja2, generic families)",
    },
}


# ---------------------------------------------------------------------------
# New slash-command implementations
# ---------------------------------------------------------------------------

def _cmd_report():
    """Display last scan result in a formatted table."""
    scan = _session.get("last_scan")
    if not scan:
        _console.print("\n  [yellow]⚠  No scan results yet.[/yellow] [dim]Run[/dim] [cyan]/scan <url> [vuln][/cyan] [dim]or ask Claude to scan.[/dim]\n")
        return

    url = _session.get("last_scan_url") or scan.get("url", "?")
    ts  = scan.get("scan_timestamp", "unknown time")
    _console.print(f"\n  [bold bright_yellow]▸ Last Scan Report[/bold bright_yellow] [dim]—[/dim] [white]{url}[/white]")
    _console.print(f"  [dim]Scanned at : {ts}[/dim]")
    _console.print("  [dim]" + "═" * 64 + "[/dim]")

    results_by_type = scan.get("results_by_type", {})
    if not results_by_type:
        # Flat results list (from direct /scan command)
        results = scan.get("results", [])
        if results:
            results_by_type = {scan.get("vuln_type", "unknown"): results}

    if not results_by_type:
        _console.print("  [dim]No detailed results in last scan data.[/dim]\n")
        return

    total_vuln = 0
    for vtype, results in results_by_type.items():
        vuln_count = sum(1 for r in results if r.get("vulnerable") or r.get("success_rate", 0) > 0)
        total_vuln += vuln_count
        color = _VULN_COLORS.get(vtype.lower(), "white")
        vc_color = "bright_red" if vuln_count > 0 else "green"
        _console.print(
            f"\n  [bold {color}]▸ {vtype.upper()}[/bold {color}]  "
            f"[{vc_color}]{vuln_count}/{len(results)}[/{vc_color}] [white]injection points vulnerable[/white]"
        )
        _console.print(
            f"  [bold]{'Parameter':<22} {'Method':<6} {'Success':>8}  {'Avg Reward':>10}[/bold]  [bold]Payloads[/bold]"
        )
        _console.print("  [dim]" + "─" * 64 + "[/dim]")
        for r in results:
            param    = (r.get("parameter") or "?")[:21]
            method   = (r.get("method") or "?")[:5]
            rate     = r.get("success_rate", 0)
            reward   = r.get("mean_reward", 0.0)
            payloads = r.get("successful_payloads", [])
            if rate > 0:
                flag = "  [bold red]✗ VULNERABLE[/bold red]"
                rate_color = "red"
            else:
                flag = "  [green]✓ clean[/green]"
                rate_color = "green"
            _console.print(
                f"  [cyan]{param:<22}[/cyan] [yellow]{method:<6}[/yellow] "
                f"[{rate_color}]{rate:>7.0%}[/{rate_color}]  [white]{reward:>10.1f}[/white]{flag}"
            )
            for pl in payloads[:3]:
                _console.print(f"      [dim]payload:[/dim] [magenta]{pl}[/magenta]")
    summary_color = "bright_red" if total_vuln > 0 else "green"
    _console.print(f"\n  [bold]Total vulnerable injection points:[/bold] [{summary_color}]{total_vuln}[/{summary_color}]")
    print()


def _cmd_payloads(filter_arg: str = ""):
    """Display all payloads tried during the last scan, grouped by parameter.

    Optional filter_arg:
      - A parameter name to show only that parameter's payloads
      - "success" or "hit" to show only successful payloads
      - "fail" to show only failed payloads
    """
    scan = _session.get("last_scan")
    if not scan:
        print("\n  [No scan results yet — run /scan <url> first]\n")
        return

    results_by_type = scan.get("results_by_type", {})
    if not results_by_type:
        results = scan.get("results", [])
        if results:
            results_by_type = {scan.get("vuln_type", "unknown"): results}

    if not results_by_type:
        print("\n  [No detailed results in last scan data]\n")
        return

    # Parse filter
    filter_lower = filter_arg.strip().lower()
    only_success = filter_lower in ("success", "hit", "successful", "hits")
    only_fail = filter_lower in ("fail", "failed", "miss")
    param_filter = filter_lower if (filter_lower and not only_success and not only_fail) else ""

    url = _session.get("last_scan_url") or scan.get("url", "?")
    print(f"\n  Payload Log — {url}")
    print("  " + "=" * 78)

    total_attempts = 0
    total_success = 0

    for vtype, results in results_by_type.items():
        for r in results:
            param = r.get("parameter", "?")
            if param_filter and param_filter not in param.lower():
                continue

            plog = r.get("payload_log", [])
            if not plog:
                continue

            # Apply success/fail filter
            if only_success:
                plog = [e for e in plog if e.get("success")]
            elif only_fail:
                plog = [e for e in plog if not e.get("success")]

            if not plog:
                continue

            method = r.get("method", "?")
            rate = r.get("success_rate", 0)
            status_tag = "VULNERABLE" if rate > 0 else "clean"
            print(f"\n  [{vtype.upper()}] Parameter: {param}  ({method})  — {status_tag} ({rate:.0%})")
            print(f"  {'#':<4} {'Ep':>3} {'Step':>4} {'Reward':>7} {'Result':<12} {'Reflected':<10} Payload")
            print("  " + "-" * 78)

            for i, entry in enumerate(plog, 1):
                ep = entry.get("episode", "?")
                step = entry.get("step", "?")
                reward = entry.get("reward", 0.0)
                success = entry.get("success", False)
                reflected = entry.get("reflected", False)
                payload = entry.get("payload", "?")

                result_tag = "✓ SUCCESS" if success else "✗ fail"
                ref_tag = "yes" if reflected else "no"
                # Truncate long payloads for display
                disp_payload = payload if len(payload) <= 60 else payload[:57] + "..."
                print(f"  {i:<4} {ep:>3} {step:>4} {reward:>+7.1f} {result_tag:<12} {ref_tag:<10} {disp_payload}")

                total_attempts += 1
                if success:
                    total_success += 1

                # Show response snippet for successful payloads
                snippet = entry.get("response_snippet", "")
                if success and snippet:
                    # Show first 120 chars of response
                    disp_snippet = snippet[:120].replace("\n", " ")
                    print(f"  {'':4} {'':3} {'':4} {'':7} {'response:':<12}           {disp_snippet}")

    if total_attempts == 0:
        if param_filter:
            print(f"\n  [No payload data found for parameter matching '{filter_arg}']\n")
        else:
            print("\n  [No payload attempt data recorded — re-run scan to capture payload log]\n")
        return

    print(f"\n  Summary: {total_success}/{total_attempts} payloads succeeded "
          f"({total_success / total_attempts:.0%})")
    if not only_success and not only_fail and not param_filter:
        print("  Tip: /payloads success  — show only successful payloads")
        print("        /payloads <param>  — filter by parameter name")
    print()


# ---------------------------------------------------------------------------
# /explain — Detailed vulnerability explanation
# ---------------------------------------------------------------------------

# Static explanations: why each payload family works (per vuln type)
_FAMILY_EXPLANATIONS = {
    # -- SQLi --
    "single_quote": (
        "The single quote (') broke out of the SQL string literal. "
        "The database interpreted everything after it as SQL code "
        "rather than data, allowing the attacker to modify the query logic."
    ),
    "or_true": (
        "OR 1=1 (or OR 'a'='a') made the WHERE clause always evaluate to TRUE. "
        "This bypassed any filtering condition, causing the database to return "
        "ALL rows instead of just the intended subset — including hidden or "
        "restricted data."
    ),
    "union_select": (
        "UNION SELECT appended a second query to the original one. "
        "This allowed the attacker to extract data from other tables "
        "(e.g. usernames, passwords) by combining it with the legitimate "
        "query results."
    ),
    "comment_bypass": (
        "The comment sequence (-- or #) truncated the rest of the original "
        "SQL query. This removed security checks like password verification "
        "or access control conditions that came after the injection point."
    ),
    "time_based": (
        "A SLEEP() or WAITFOR DELAY command was injected into the SQL query. "
        "The database paused for the specified duration, confirming that "
        "attacker-controlled SQL is being executed — even though no data "
        "was returned directly in the response (blind injection)."
    ),
    "error_based": (
        "The payload triggered a SQL syntax error or type conversion error. "
        "The error message revealed database internals (table names, column "
        "types, query structure), confirming the injection and leaking "
        "sensitive schema information."
    ),
    "stacked": (
        "A semicolon (;) terminated the original query and started a new one. "
        "This allowed the attacker to execute arbitrary SQL statements "
        "(INSERT, UPDATE, DELETE, DROP) beyond just reading data."
    ),
    "encoded": (
        "The payload used URL encoding, hex encoding, or double encoding "
        "to bypass input filters or WAF rules. The database decoded the "
        "payload and executed it normally, even though it looked harmless "
        "to the application's input validation."
    ),
    # -- XSS --
    "basic_script": (
        "A <script> tag was injected and rendered as executable HTML. "
        "The server did not encode or sanitize the output, so the browser "
        "executed the JavaScript code — allowing cookie theft, session "
        "hijacking, or page manipulation."
    ),
    "exploit_proof": (
        "A high-impact proof-of-exploit payload was injected (e.g. "
        "document.cookie exfiltration, CSRF bypass, or password capture). "
        "This demonstrates real-world attack impact beyond simple alert() popups."
    ),
    "img_onerror": (
        "An <img> tag with an invalid src triggered the onerror event handler. "
        "The browser automatically fires onerror when an image fails to load, "
        "executing the attacker's JavaScript without any user interaction."
    ),
    "svg_onload": (
        "An <svg> tag with an onload handler was injected. SVG elements "
        "fire onload immediately when rendered, executing JavaScript "
        "automatically — bypassing filters that only block <script> tags."
    ),
    "event_handler": (
        "An HTML event handler (onfocus, onmouseover, onclick) was attached "
        "to an injected element. When the user interacts with the page "
        "(or the element auto-focuses), the JavaScript executes."
    ),
    "attribute_escape": (
        "The payload broke out of an HTML attribute context using a quote "
        "character, then injected a new event handler attribute. This works "
        "when user input is placed inside an attribute value without proper "
        "escaping."
    ),
    "case_bypass": (
        "Mixed case (<ScRiPt>) or alternate tag spellings bypassed a "
        "case-sensitive blacklist filter. The browser is case-insensitive "
        "when parsing HTML tags, so the payload executed normally."
    ),
    "encoding_bypass": (
        "HTML entity encoding, Unicode escapes, or URL encoding was used "
        "to bypass input sanitization. The browser decoded the payload "
        "and executed it, even though it passed through the filter undetected."
    ),
    "nested_tags": (
        "Nested or malformed tags confused the server's HTML parser or "
        "sanitizer. The incomplete first tag was stripped, but the inner "
        "tag survived and executed in the browser."
    ),
    "dom_based": (
        "The payload was processed by client-side JavaScript (e.g. "
        "document.location, innerHTML, eval) rather than reflected by "
        "the server. The vulnerability exists entirely in the frontend "
        "code — the server never sees the malicious content."
    ),
    "polyglot": (
        "A polyglot payload works across multiple injection contexts "
        "(HTML, attribute, JavaScript, URL). It contains multiple escape "
        "sequences so that at least one activates regardless of where "
        "the input is reflected."
    ),
    # -- CMDi --
    "command_separator": (
        "A semicolon (;) terminated the original command and started a "
        "new one. The server's shell executed both commands sequentially, "
        "allowing arbitrary command execution on the underlying OS."
    ),
    "pipe_operator": (
        "The pipe operator (|) redirected the output of the original "
        "command into the attacker's command. The shell executed the "
        "injected command with the original command's output as input."
    ),
    "logical_and": (
        "The && operator chained a new command after the original one. "
        "The shell executed the attacker's command only if the first "
        "command succeeded — a reliable way to inject commands."
    ),
    "logical_or": (
        "The || operator ran the attacker's command when the first "
        "command failed. This works even when the original command "
        "is intentionally broken by the injection."
    ),
    "backtick_exec": (
        "Backtick characters (`command`) caused the shell to execute "
        "the enclosed command and substitute its output. This is often "
        "missed by filters that only look for semicolons and pipes."
    ),
    "dollar_paren": (
        "$(command) syntax triggered command substitution in bash/sh. "
        "The shell executed the inner command and replaced the expression "
        "with its output — similar to backticks but nestable."
    ),
    "newline_inject": (
        "A newline character (\\n or %0a) started a new line in the "
        "shell, effectively starting a new command. This bypasses "
        "filters that look for command separators on a single line."
    ),
    # -- SSTI --
    "arithmetic_probe": (
        "A template expression like {{7*7}} was evaluated by the server's "
        "template engine, producing 49 in the response. This confirms "
        "the engine processes user input as code, not data."
    ),
    "string_ops": (
        "A string operation (e.g. 'a'*3 → 'aaa') was evaluated by the "
        "template engine. This confirms code execution and helps identify "
        "which template engine is in use (Jinja2, Twig, Pug, etc.)."
    ),
    "global_object": (
        "The payload accessed a global object (e.g. self, config, "
        "global, __builtins__) exposed by the template engine. This "
        "escalates from expression evaluation to accessing server "
        "internals and potentially reading files or environment variables."
    ),
    "env_access": (
        "The payload read environment variables or process information "
        "from the server (e.g. process.env, os.environ). This leaks "
        "sensitive configuration like API keys, database credentials, "
        "and secret tokens."
    ),
    "waf_bypass_encoding": (
        "Encoding tricks (hex, unicode, HTML entities) bypassed the "
        "WAF or input filter while still being interpreted by the "
        "template engine. The filter saw harmless text, but the "
        "engine decoded and executed it."
    ),
    "waf_bypass_concat": (
        "String concatenation or attribute chaining bypassed keyword "
        "filters. Instead of directly calling a blocked function, the "
        "payload constructed it from parts the filter didn't recognize."
    ),
    "rce_direct": (
        "The payload achieved Remote Code Execution by directly calling "
        "OS commands through the template engine (e.g. os.popen, "
        "child_process.exec). The attacker can run arbitrary commands "
        "on the server."
    ),
    "rce_indirect": (
        "The payload achieved Remote Code Execution through an indirect "
        "chain — accessing internal objects, traversing the class hierarchy, "
        "and reaching a code execution method. This works even when direct "
        "command functions are blocked."
    ),
}

# Vuln-type-level fix recommendations (static fallback if LLM remediation unavailable)
_STATIC_REMEDIATION = {
    "sqli": (
        "Use parameterized queries (prepared statements) for ALL database access.\n"
        "  Never concatenate user input into SQL strings.\n"
        "  Example (Python):\n"
        "    cursor.execute('SELECT * FROM users WHERE name = ?', [user_input])\n"
        "  Also apply least-privilege database accounts and input validation as defense-in-depth."
    ),
    "xss": (
        "Apply context-aware output encoding for ALL user-controlled data:\n"
        "  - HTML context: encode < > & \" ' to HTML entities\n"
        "  - JavaScript context: use JSON.stringify() or \\xNN encoding\n"
        "  - URL context: use percent-encoding\n"
        "  Adopt a Content Security Policy (CSP) header that blocks inline scripts.\n"
        "  Use frameworks with auto-escaping (React, Angular, Jinja2 autoescape)."
    ),
    "cmdi": (
        "Never pass user input to shell commands.\n"
        "  Use language-native APIs instead of OS commands:\n"
        "    - File operations: use open(), pathlib, shutil (not os.system('cat ...'))\n"
        "    - Network: use socket/requests (not os.system('ping ...'))\n"
        "  If shell commands are unavoidable, use subprocess with shell=False\n"
        "  and pass arguments as a list, never a string."
    ),
    "ssti": (
        "Never pass user input directly into template strings.\n"
        "  Use templates as files with data passed as context variables:\n"
        "    - Jinja2: render_template('page.html', name=user_input)\n"
        "    - NOT: Template(user_input).render()\n"
        "  Enable sandboxing if available (Jinja2 SandboxedEnvironment).\n"
        "  Restrict template engine capabilities to deny code execution."
    ),
}


def _cmd_explain(filter_arg: str = ""):
    """Show detailed explanation of each vulnerability found: what, how, why, and fix."""
    scan = _session.get("last_scan")
    if not scan:
        print("\n  [No scan results yet — run /scan <url> first]\n")
        return

    results_by_type = scan.get("results_by_type", {})
    if not results_by_type:
        results = scan.get("results", [])
        if results:
            results_by_type = {scan.get("vuln_type", "unknown"): results}

    if not results_by_type:
        print("\n  [No detailed results in last scan data]\n")
        return

    param_filter = filter_arg.strip().lower() if filter_arg else ""
    url = _session.get("last_scan_url") or scan.get("url", "?")
    remediation_by_type = scan.get("remediation_by_type", {})

    # Collect only vulnerable results
    vuln_entries = []
    for vtype, results in results_by_type.items():
        for r in results:
            if r.get("success_rate", 0) <= 0 and not r.get("vulnerable"):
                continue
            param = r.get("parameter", "?")
            if param_filter and param_filter not in param.lower():
                continue
            vuln_entries.append((vtype, r))

    if not vuln_entries:
        if param_filter:
            print(f"\n  [No vulnerabilities found matching '{filter_arg}']\n")
        else:
            print("\n  [No vulnerabilities found in last scan — nothing to explain]\n")
        return

    print(f"\n  {'=' * 72}")
    print(f"  VULNERABILITY EXPLANATION REPORT")
    print(f"  Target: {url}")
    print(f"  {'=' * 72}")

    for idx, (vtype, r) in enumerate(vuln_entries, 1):
        param = r.get("parameter", "?")
        method = r.get("method", "?")
        inj_url = r.get("injection_url") or r.get("url", "?")
        found_on = r.get("found_on_page", "")
        subtype = r.get("vuln_subtype", "")
        success_rate = r.get("success_rate", 0)
        episodes = r.get("num_episodes", 0)
        successes = r.get("successes", 0)
        early = r.get("early_stopped", False)
        nav_hint = r.get("how_to_reach", "") or r.get("nav_hint", "")
        evidence = r.get("impact_evidence", [])
        payloads = r.get("successful_payloads", [])
        plog = r.get("payload_log", [])
        snippet = r.get("response_snippet", "")

        type_label = vtype.upper()
        if subtype:
            type_label = f"{vtype.upper()} ({subtype})"

        print(f"\n  {'─' * 72}")
        print(f"  [{idx}] VULNERABILITY: {type_label}")
        print(f"  {'─' * 72}")
        print(f"  Parameter : {param}")
        print(f"  Method    : {method}")
        print(f"  URL       : {inj_url}")
        if found_on and found_on != inj_url:
            print(f"  Found on  : {found_on}")
        if nav_hint:
            print(f"  Nav hint  : {nav_hint}")

        # --- HOW IT WAS FOUND ---
        print(f"\n  HOW IT WAS FOUND:")
        if found_on:
            print(f"    Discovered by deep crawl on {found_on}")
        eps_str = f"{successes}/{episodes} episodes succeeded"
        if early:
            eps_str += " (early-stopped — confirmed quickly)"
        print(f"    Success rate: {success_rate:.0%} — {eps_str}")

        # --- SUCCESSFUL PAYLOADS + WHY IT WORKED ---
        if payloads:
            print(f"\n  SUCCESSFUL PAYLOAD(S):")
            for pi, pl in enumerate(payloads[:5], 1):
                print(f"    {pi}. {pl}")

        # Find the first successful entry in payload_log for full detail
        successful_entries = [e for e in plog if e.get("success")]
        if successful_entries:
            first_hit = successful_entries[0]
            print(f"\n  FULL REQUEST:")
            hit_url = first_hit.get("url", inj_url)
            hit_payload = first_hit.get("payload", "")
            if method == "GET":
                print(f"    GET {hit_url}?{param}={hit_payload}")
            else:
                print(f"    POST {hit_url}")
                print(f"    Body: {param}={hit_payload}")

            hit_snippet = first_hit.get("response_snippet", "") or snippet
            if hit_snippet:
                print(f"\n  SERVER RESPONSE (snippet):")
                # Show first 400 chars, split into lines for readability
                disp = hit_snippet[:400].replace("\n", "\n    ")
                print(f"    {disp}")
                if len(hit_snippet) > 400:
                    print(f"    ...")

        # --- WHY IT WORKED ---
        # Try to match the successful payload to a family explanation
        explanation_shown = False
        if payloads:
            # Identify which family the payload came from
            matched_family = _identify_payload_family(payloads[0], vtype)
            if matched_family and matched_family in _FAMILY_EXPLANATIONS:
                print(f"\n  WHY IT WORKED:")
                # Word-wrap the explanation at ~68 chars
                expl = _FAMILY_EXPLANATIONS[matched_family]
                for line in _wrap_text(expl, 68):
                    print(f"    {line}")
                explanation_shown = True

        # --- EVIDENCE ---
        if evidence:
            print(f"\n  EVIDENCE:")
            for ev in evidence:
                print(f"    > {ev}")

        # --- HOW TO FIX ---
        print(f"\n  HOW TO FIX:")
        llm_remediation = remediation_by_type.get(vtype, "")
        if llm_remediation:
            # Show first ~500 chars of LLM-generated remediation
            for line in llm_remediation[:500].split("\n"):
                print(f"    {line}")
            if len(llm_remediation) > 500:
                print(f"    ...")
        elif vtype in _STATIC_REMEDIATION:
            for line in _STATIC_REMEDIATION[vtype].split("\n"):
                print(f"    {line}")
        else:
            print(f"    Sanitize and validate all user input before use.")

    print(f"\n  {'=' * 72}")
    print(f"  Total: {len(vuln_entries)} vulnerability/ies explained")
    print(f"  {'=' * 72}\n")


def _identify_payload_family(payload: str, vuln_type: str) -> str | None:
    """Best-effort match a payload string to its family name."""
    pl = payload.lower()

    if vuln_type == "xss":
        if "document.cookie" in pl or "exploit.test" in pl:
            return "exploit_proof"
        if "<script" in pl:
            return "basic_script"
        if "onerror" in pl and "<img" in pl:
            return "img_onerror"
        if "<svg" in pl and "onload" in pl:
            return "svg_onload"
        if any(h in pl for h in ("onfocus", "onmouseover", "onclick", "onmouseenter")):
            return "event_handler"
        if "onerror" in pl and ("<video" in pl or "<source" in pl or "<body" in pl):
            return "event_handler"
        if "javascript:" in pl or "formaction" in pl:
            return "attribute_escape"
        if pl != payload.lower() or "<ScRiPt" in payload or "<IMG" in payload:
            return "case_bypass"
        if "&#" in pl or "%3c" in pl or "\\u" in pl:
            return "encoding_bypass"
        if pl.count("<") >= 2:
            return "nested_tags"
        if "document." in pl and "innerhtml" in pl:
            return "dom_based"
        if ("<svg" in pl or "javascript:" in pl) and "<script" in pl:
            return "polyglot"

    elif vuln_type == "sqli":
        if "sleep" in pl or "waitfor" in pl or "pg_sleep" in pl or "benchmark" in pl:
            return "time_based"
        if "union" in pl and "select" in pl:
            return "union_select"
        if " or " in pl and ("1=1" in pl or "'a'='a" in pl or "true" in pl):
            return "or_true"
        if "convert(" in pl or "extractvalue" in pl or "updatexml" in pl:
            return "error_based"
        if ";" in pl and ("select" in pl or "insert" in pl or "drop" in pl or "update" in pl):
            return "stacked"
        if "%27" in pl or "0x" in pl or "%2527" in pl:
            return "encoded"
        if "--" in pl or "#" in pl:
            return "comment_bypass"
        if "'" in pl:
            return "single_quote"

    elif vuln_type == "cmdi":
        if "`" in pl:
            return "backtick_exec"
        if "$(" in pl:
            return "dollar_paren"
        if "%0a" in pl or "\\n" in pl:
            return "newline_inject"
        if "&&" in pl:
            return "logical_and"
        if "||" in pl:
            return "logical_or"
        if "|" in pl:
            return "pipe_operator"
        if ";" in pl:
            return "command_separator"
        if "%3b" in pl or "%7c" in pl:
            return "encoded"

    elif vuln_type == "ssti":
        if "popen" in pl or "system" in pl or "exec" in pl or "child_process" in pl:
            return "rce_direct"
        if "__mro__" in pl or "__subclasses__" in pl or "__globals__" in pl:
            return "rce_indirect"
        if "process.env" in pl or "os.environ" in pl:
            return "env_access"
        if "self." in pl or "config" in pl or "__builtins__" in pl or "global" in pl:
            return "global_object"
        if "\\x" in pl or "&#" in pl or "%7b" in pl:
            return "waf_bypass_encoding"
        if "~" in pl or "|join" in pl or "|attr" in pl:
            return "waf_bypass_concat"
        if "*" in pl and any(c.isdigit() for c in pl):
            return "arithmetic_probe"
        if "'a'" in pl or "'b'" in pl or "concat" in pl:
            return "string_ops"

    return None


def _wrap_text(text: str, width: int) -> list[str]:
    """Simple word-wrap for display."""
    words = text.split()
    lines = []
    current = ""
    for w in words:
        if current and len(current) + 1 + len(w) > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines


def _cmd_export(filename: str = ""):
    """Save last scan result as JSON."""
    scan = _session.get("last_scan")
    if not scan:
        print("\n  [No scan results yet — run a scan first]\n")
        return
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reports/assistant_scan_{ts}.json"
    try:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(scan, f, indent=2, default=str)
        print(f"\n  [Scan exported to {filename}]\n")
    except Exception as e:
        _elog.error("Scan export failed | file=%s | %s: %s", filename, type(e).__name__, e)
        print(f"\n  [Could not export scan — check the file path and permissions.]\n")


def _cmd_targets():
    """Show known test application targets with live reachability check."""
    _console.print("\n  [bold bright_yellow]▸ Known Test Application Targets[/bold bright_yellow]")
    _console.print("  [dim]" + "═" * 60 + "[/dim]")
    for t in _KNOWN_TARGETS:
        try:
            r = requests.get(t["url"], timeout=2)
            status = "[green]● UP  [/green]" if r.status_code < 500 else "[red]● DOWN[/red]"
        except Exception:
            status = "[red]● DOWN[/red]"
        _console.print(f"\n  [bold bright_cyan]{t['name']:12}[/bold bright_cyan] [white]{t['url']}[/white]   {status}")
        _console.print(f"    [bold]Creds  [/bold] [dim]:[/dim] [yellow]{t['default_creds']}[/yellow]")
        _console.print(f"    [bold]Notes  [/bold] [dim]:[/dim] [white]{t['notes']}[/white]")
        _console.print(f"    [bold]Docker [/bold] [dim]:[/dim] [cyan]docker start {t['container']}[/cyan]")
        _console.print(f"    [bold]Scan   [/bold] [dim]:[/dim] [green]/scan {t['url']} all[/green]")
    print()


_VULN_COLORS = {"xss": "bright_magenta", "sqli": "bright_red", "cmdi": "bright_yellow", "ssti": "bright_green"}


def _cmd_vulns(arg: str = ""):
    """Explain vulnerability types. Pass a type name to filter."""
    arg = arg.lower().strip()
    items = (
        {arg: _VULN_DESCRIPTIONS[arg]}
        if arg in _VULN_DESCRIPTIONS
        else _VULN_DESCRIPTIONS
    )
    _console.print("\n  [bold bright_yellow]▸ Supported Vulnerability Types[/bold bright_yellow]")
    _console.print("  [dim]" + "═" * 60 + "[/dim]")
    for vtype, info in items.items():
        color = _VULN_COLORS.get(vtype, "white")
        _console.print(f"\n  [bold {color}]▸ {vtype.upper()}[/bold {color}]  [white]{info['full']}[/white]")
        _console.print(f"    [dim]{info['cwe']}  |  {info['owasp']}[/dim]")
        _console.print(f"    [white]{info['summary']}[/white]")
        _console.print(f"    [bold]Model  [/bold] [dim]:[/dim] [cyan]{info['model']}[/cyan]")
        _console.print(f"    [bold]Actions[/bold] [dim]:[/dim] [yellow]{info['action_space']}[/yellow]")
    print()


def _cmd_verbose():
    """Toggle verbose (DEBUG) logging."""
    _session["verbose"] = not _session["verbose"]
    level = logging.DEBUG if _session["verbose"] else logging.WARNING
    logging.getLogger().setLevel(level)
    state = "ON  (DEBUG logging enabled)" if _session["verbose"] else "OFF (warnings only)"
    print(f"\n  [Verbose mode: {state}]\n")


def _cmd_cancel():
    """Signal any running scan to stop and return partial results."""
    if _cancel_event.is_set():
        print("\n  [Cancel already requested — waiting for the scan to stop...]\n")
    else:
        _cancel_event.set()
        print("\n  [Cancel requested — the current scan will stop after this episode.]\n")
        print("  [Run /report to see partial results once it finishes.]\n")


def _cmd_scan_direct(arg: str, api_key: str):
    """
    /scan <url> [xss|sqli|cmdi|ssti|all] [episodes=N] [shallow]
    Scan a URL directly — no API key required.
    Deep crawl is ON by default (follows internal links to find all pages).
    Add 'shallow' to only scan the given URL without following links.
    """
    parts = arg.split()
    if not parts:
        print("\n  Usage: /scan <url> [xss|sqli|cmdi|ssti|all] [episodes=N] [shallow]")
        print("  Example: /scan http://localhost:8080 xss episodes=20\n")
        return

    url       = parts[0]
    vuln_type = "all"
    episodes  = 10
    deep      = True   # Deep crawl by default — finds signup/login/admin pages

    for part in parts[1:]:
        if part.lower() in ("xss", "sqli", "cmdi", "ssti", "all"):
            vuln_type = part.lower()
        elif part.lower().startswith("episodes="):
            try:
                episodes = int(part.split("=", 1)[1])
            except ValueError:
                pass
        elif part.lower() == "shallow":
            deep = False
        elif part.lower() == "deep":
            deep = True  # explicit deep still accepted

    print(f"\n  [Direct scan] {url}  vuln={vuln_type}  episodes={episodes}  deep={deep}\n")
    result = tool_scan_target(url=url, vuln_type=vuln_type, episodes=episodes, api_key=api_key, deep=deep)
    result["scan_timestamp"] = datetime.now().isoformat()
    _session["last_scan"]     = result
    _session["last_scan_url"] = url
    _session["scans_run"]    += 1

    if "error" in result:
        _elog.error("Direct scan error | url=%s vuln=%s | %s", url, vuln_type, result["error"])
        _safe_err = result["error"] if "Model" in result["error"] or "No injection" in result["error"] else "Check that the target is running and the URL is correct."
        print(f"  [Scan could not be completed — {_safe_err}]\n")
    else:
        print(f"\n  Summary: {result.get('summary', 'No results.')}")
        vuln_total = result.get("total_vulnerable", 0)
        tested     = result.get("injection_points_tested", 0)
        print(f"  {vuln_total}/{tested} injection point(s) vulnerable.")
        if vuln_total > 0:
            print("  Run /report for the full formatted table.\n")
        else:
            print()


def _cmd_crawl_direct(arg: str, api_key: str):
    """
    /crawl <url> [cookie=<value>]
    Discover injection points without attacking — no API key required.
    """
    parts = arg.split()
    if not parts:
        print("\n  Usage: /crawl <url> [cookie=<value>]")
        print("  Example: /crawl http://localhost:8080 cookie=PHPSESSID=abc\n")
        return

    url    = parts[0]
    cookie = ""
    for part in parts[1:]:
        if part.startswith("cookie="):
            cookie = part.split("=", 1)[1]

    print(f"\n  [Direct crawl] {url} ...\n")
    result = tool_crawl_target(url=url, cookie=cookie, api_key=api_key)

    if "error" in result:
        _elog.error("Crawl error | url=%s | %s", url, result["error"])
        print(f"  [Could not crawl target — make sure it is running and the URL is correct.]\n")
        return

    pts = result.get("injection_points", [])
    print(f"  Found {len(pts)} injection point(s) at {url}")
    if pts:
        print(f"  {'#':<3} {'Parameter':<24} {'Method':<6} Type")
        print("  " + "-" * 54)
        for i, p in enumerate(pts, 1):
            print(f"  {i:<3} {p['parameter']:<24} {p['method']:<6} {p['input_type']}")
            if p.get("description"):
                print(f"      {p['description']}")
    print()


def _cmd_auth_direct(arg: str, api_key: str):
    """
    /auth <url> <username> <password> [xss|sqli|cmdi|ssti|all] [episodes=N]
    Log in and run an authenticated scan — no API key required.
    """
    parts = arg.split()
    if len(parts) < 3:
        print("\n  Usage: /auth <url> <username> <password> [vuln] [episodes=N]")
        print("  Example: /auth http://localhost:8080 admin password xss\n")
        return

    url, username, password = parts[0], parts[1], parts[2]
    vuln_type = "all"
    episodes  = 10

    for part in parts[3:]:
        if part.lower() in ("xss", "sqli", "cmdi", "ssti", "all"):
            vuln_type = part.lower()
        elif part.lower().startswith("episodes="):
            try:
                episodes = int(part.split("=", 1)[1])
            except ValueError:
                pass

    print(f"\n  [Authenticated scan] {url}  user={username!r}  vuln={vuln_type}  episodes={episodes}\n")
    result = tool_authenticated_scan(
        url=url, username=username, password=password,
        vuln_type=vuln_type, episodes=episodes, api_key=api_key, deep=True,
    )
    result["scan_timestamp"] = datetime.now().isoformat()
    _session["last_scan"]     = result
    _session["last_scan_url"] = url
    _session["scans_run"]    += 1

    if "error" in result and not result.get("authenticated"):
        _elog.error("Auth scan error | url=%s user=%s | %s", url, username, result["error"])
        print(f"  [Login failed — check the target is running and the credentials are correct.]\n")
    else:
        auth_ok = result.get("authenticated", False)
        print(f"  Auth: {'OK' if auth_ok else 'FAILED'}  — {result.get('auth_message', '')}")
        print(f"  Summary: {result.get('summary', 'No results.')}")
        vuln_total = result.get("total_vulnerable", 0)
        tested     = result.get("injection_points_tested", 0)
        print(f"  {vuln_total}/{tested} injection point(s) vulnerable.")
        if vuln_total > 0:
            print("  Run /report for the full formatted table.\n")
        else:
            print()


def _list_error_logs() -> list[Path]:
    """Return all daily error log files in ERROR_LOG_DIR, newest first."""
    if not ERROR_LOG_DIR.exists():
        return []
    files = sorted(ERROR_LOG_DIR.glob("errors_*.log*"), reverse=True)
    return files


def _cmd_admin(arg: str):
    """
    /admin <password> [list] [date=YYYY-MM-DD] [lines=N] [today]

    Subcommands (all require the correct password first):
      list                 Show all available daily error log files with entry counts.
      date=YYYY-MM-DD      View the log for a specific past date.
      today                Filter to today's entries only (default view).
      lines=N              Show the last N lines (default 50).

    Requires ASSISTANT_ADMIN_PASSWORD in the environment.
    """
    admin_pw = os.environ.get("ASSISTANT_ADMIN_PASSWORD", "")
    if not admin_pw:
        print(
            "\n  [Admin access is not configured.]\n"
            "  Set ASSISTANT_ADMIN_PASSWORD=<your-password> in your .env file.\n"
        )
        return

    parts = arg.split()
    if not parts:
        print(
            "\n  Usage: /admin <password> [list | date=YYYY-MM-DD | today] [lines=N]\n"
            "  Example: /admin secret list\n"
            "  Example: /admin secret date=2026-03-14\n"
            "  Example: /admin secret today lines=100\n"
        )
        return

    given_pw   = parts[0]
    do_list    = False
    specific_date: str | None = None
    lines_n    = 50
    today_only = False

    for p in parts[1:]:
        pl = p.lower()
        if pl == "list":
            do_list = True
        elif pl == "today":
            today_only = True
        elif pl.startswith("date="):
            specific_date = pl.split("=", 1)[1]
        elif pl.startswith("lines="):
            try:
                lines_n = int(pl.split("=", 1)[1])
            except ValueError:
                pass

    if given_pw != admin_pw:
        _elog.warning("Failed admin login attempt.")
        print("\n  [Access denied.]\n")
        return

    # ---- /admin <pw> list -----------------------------------------------
    if do_list:
        log_files = _list_error_logs()
        if not log_files:
            print("\n  [No error log files found — no errors have been recorded yet.]\n")
            return
        print(f"\n  Error Log Directory  —  {ERROR_LOG_DIR.resolve()}")
        print(f"  {'FILE':<40}  {'ENTRIES':>7}  {'SIZE':>8}")
        print("  " + "-" * 60)
        for lf in log_files:
            try:
                with open(lf, encoding="utf-8", errors="replace") as f:
                    count = sum(1 for _ in f)
                size_kb = lf.stat().st_size / 1024
                marker = "  <-- today" if lf.stem == f"errors_{datetime.now().strftime('%Y-%m-%d')}" else ""
                print(f"  {lf.name:<40}  {count:>7}  {size_kb:>6.1f} KB{marker}")
            except Exception:
                print(f"  {lf.name:<40}  {'?':>7}  {'?':>8}")
        print()
        return

    # ---- /admin <pw> date=YYYY-MM-DD -------------------------------------
    if specific_date:
        # Look for the exact daily file
        target_file = ERROR_LOG_DIR / f"errors_{specific_date}.log"
        if not target_file.exists():
            # Check for rotated variants (TimedRotatingFileHandler appends the date suffix)
            candidates = list(ERROR_LOG_DIR.glob(f"errors_*.log.{specific_date}"))
            if candidates:
                target_file = candidates[0]
            else:
                print(f"\n  [No error log found for {specific_date}.]\n"
                      f"  Use '/admin <pw> list' to see available dates.\n")
                return
    else:
        target_file = ERROR_LOG_FILE

    # ---- Read and display -----------------------------------------------
    if not target_file.exists():
        print("\n  [No error log found — no errors have been recorded yet.]\n")
        return

    try:
        with open(target_file, encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except Exception as e:
        _elog.error("Admin log read failed | file=%s | %s: %s", target_file, type(e).__name__, e)
        print("\n  [Could not read the error log.]\n")
        return

    if today_only and not specific_date:
        today_str = datetime.now().strftime("%Y-%m-%d")
        all_lines = [l for l in all_lines if l.startswith(today_str)]

    shown = all_lines[-lines_n:]
    label = specific_date or datetime.now().strftime("%Y-%m-%d")
    print(f"\n  Error Log  —  {target_file.resolve()}")
    print(f"  Date: {label}  |  Showing last {len(shown)} of {len(all_lines)} "
          f"entr{'y' if len(all_lines) == 1 else 'ies'}")
    if today_only:
        print(f"  Filter: today ({datetime.now().strftime('%Y-%m-%d')})")
    print("  " + "=" * 72)
    if shown:
        for line in shown:
            print(f"  {line.rstrip()}")
    else:
        print("  (no entries match)")
    print()


def _handle_slash(cmd_line: str, history: list, api_key: str) -> str:
    """
    Dispatch a slash command.
    Returns the (possibly updated) api_key.
    Mutable state (scans_run, last_scan, verbose) lives in module-level _session.
    """
    raw = cmd_line.lstrip("/").strip()
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower() if parts else ""
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("help", "h", "?"):
        _cmd_help(arg)
    elif cmd == "models":
        _cmd_models()
    elif cmd == "history":
        _cmd_history(history, arg)
    elif cmd == "save":
        _cmd_save(history, arg)
    elif cmd == "status":
        _cmd_status(history, api_key, _session["scans_run"])
    elif cmd in ("clear", "reset"):
        history.clear()
        _save_history([])
        print("  [Conversation cleared]\n")
    elif cmd == "setkey":
        if arg:
            api_key = arg.strip()
            print("  [API key updated — AI chat is now enabled]\n")
        else:
            print("  Usage: /setkey sk-ant-...\n")
    elif cmd == "report":
        _cmd_report()
    elif cmd == "payloads":
        _cmd_payloads(arg)
    elif cmd == "explain":
        _cmd_explain(arg)
    elif cmd == "export":
        _cmd_export(arg)
    elif cmd == "targets":
        _cmd_targets()
    elif cmd == "vulns":
        _cmd_vulns(arg)
    elif cmd == "verbose":
        _cmd_verbose()
    elif cmd == "cancel":
        _cmd_cancel()
    elif cmd == "scan":
        _cancel_event.clear()
        _cmd_scan_direct(arg, api_key)
        _cancel_event.clear()
    elif cmd == "crawl":
        _cancel_event.clear()
        _cmd_crawl_direct(arg, api_key)
        _cancel_event.clear()
    elif cmd == "auth":
        _cancel_event.clear()
        _cmd_auth_direct(arg, api_key)
        _cancel_event.clear()
    elif cmd == "admin":
        _cmd_admin(arg)
    elif cmd in ("quit", "exit", "q"):
        raise SystemExit(0)
    else:
        print(f"  [Unknown command: /{cmd} — type /help for a list]\n")

    return api_key


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

def chat(api_key: str):
    client = anthropic.Anthropic(api_key=api_key) if api_key else None
    history: list = []

    # ── Pretty banner ─────────────────────────────────────────────────────
    _banner_body = Text()
    _banner_body.append("AI PENTEST ASSISTANT", style="bold bright_cyan")
    _banner_body.append("   ·   ", style="dim")
    _banner_body.append("RL-Powered Web Vulnerability Scanner", style="italic white")
    _banner_body.append("\n\n")
    _banner_body.append("Trained models  ", style="bold")
    _banner_body.append("XSS ", style="bright_magenta")
    _banner_body.append("· ", style="dim")
    _banner_body.append("SQLi ", style="bright_red")
    _banner_body.append("· ", style="dim")
    _banner_body.append("CMDi ", style="bright_yellow")
    _banner_body.append("· ", style="dim")
    _banner_body.append("SSTI", style="bright_green")
    _banner_body.append("\n")
    _banner_body.append("Commands        ", style="bold")
    _banner_body.append("/help", style="cyan")
    _banner_body.append("  ·  ", style="dim")
    _banner_body.append("quit · exit · bye", style="cyan")
    _banner_body.append("  ·  ", style="dim")
    _banner_body.append("Ctrl+C to cancel", style="cyan")
    _console.print()
    _console.print(Panel(
        Align.left(_banner_body),
        border_style="bright_cyan",
        padding=(1, 2),
        title="[bold white]🛡  Security Research Tool[/bold white]",
        title_align="left",
    ))
    if not api_key:
        _console.print(
            "\n  [yellow]⚠  No API key set.[/yellow] "
            "[dim]Slash commands still work:[/dim] "
            "[cyan]/scan /crawl /auth /report /targets /vulns[/cyan]"
        )
        _console.print(
            "  [dim]Run[/dim] [cyan]/setkey sk-ant-...[/cyan] "
            "[dim]to enable AI-powered chat.[/dim]\n"
        )
    else:
        _console.print()

    # Offer to restore previous session
    saved = _load_history()
    if saved:
        try:
            _console.print(
                f"  [dim]💾 {len(saved)} saved messages found.[/dim]",
            )
            restore = input("  Restore previous session? (y/N): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            restore = "n"
        if restore == "y":
            history = saved
            _console.print(f"  [green]✓ Session restored — {len(saved)} messages[/green]\n")
        else:
            print()

    while True:
        try:
            _console.print("[bold bright_cyan]You[/bold bright_cyan] ", end="")
            user_input = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            _console.print("\n  [dim yellow]👋 Session ended — goodbye.[/dim yellow]")
            try:
                _save_history(history)
            except (Exception, KeyboardInterrupt):
                pass
            break

        if not user_input:
            continue

        # Quit — many aliases, case-insensitive
        if user_input.lower().lstrip("/") in _QUIT_WORDS:
            _console.print("  [dim yellow]👋 Session ended — goodbye.[/dim yellow]")
            try:
                _save_history(history)
            except (Exception, KeyboardInterrupt):
                pass
            break

        # Slash commands — work without an API key
        if user_input.startswith("/") or user_input == "?":
            try:
                api_key = _handle_slash(user_input, history, api_key)
                # Rebuild client if key was set/updated via /setkey
                if api_key:
                    client = anthropic.Anthropic(api_key=api_key)
            except SystemExit:
                _console.print("  [dim yellow]👋 Goodbye.[/dim yellow]")
                try:
                    _save_history(history)
                except (Exception, KeyboardInterrupt):
                    pass
                break
            continue

        # Natural language input — requires API key
        if client is None:
            _console.print(
                "\n  [yellow]⚠  No API key set — AI chat is disabled.[/yellow]"
            )
            _console.print(
                "  [dim]Use[/dim] [cyan]/setkey sk-ant-...[/cyan] "
                "[dim]to enable it, or[/dim] [cyan]/help[/cyan] "
                "[dim]for local commands.[/dim]\n"
            )
            continue

        history.append({"role": "user", "content": user_input})

        # Agentic loop: keep calling Claude until it stops using tools
        _turn = 0
        while True:
            _turn += 1
            _spinner = None
            try:
                # Estimate seconds from approximate input token count
                _approx_tokens = sum(
                    len(str(m.get("content", ""))) for m in history
                ) // 4
                _est_s = max(3, _approx_tokens // 600)
                _spinner_text = (
                    f"[bold cyan]Thinking...[/bold cyan] "
                    f"[dim](~{_est_s}s estimated)[/dim]"
                )
                _t0 = time.monotonic()
                _spinner = RichStatus(_spinner_text, console=_console, spinner="dots")
                _spinner.start()
                with client.messages.stream(
                    model=ASSISTANT_MODEL,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=history,
                ) as stream:
                    printed_header = False
                    for text_chunk in stream.text_stream:
                        if not printed_header:
                            _spinner.stop()
                            _elapsed = time.monotonic() - _t0
                            _console.print(
                                f"[dim]  (~{_elapsed:.1f}s)[/dim]",
                                highlight=False,
                            )
                            _console.print("\n[bold bright_green]Assistant[/bold bright_green] [dim]›[/dim] ", end="")
                            printed_header = True
                        print(text_chunk, end="", flush=True)
                    if printed_header:
                        print("\n")
                    else:
                        # Pure tool-use turn — stop spinner silently
                        _spinner.stop()
                        _elapsed = time.monotonic() - _t0
                        _console.print(
                            f"[dim]  (~{_elapsed:.1f}s)[/dim]",
                            highlight=False,
                        )
                    response = stream.get_final_message()
            except anthropic.APIConnectionError as e:
                if _spinner: _spinner.stop()
                _elog.error("APIConnectionError | %s: %s", type(e).__name__, e)
                _console.print("\n  [red]✗ Could not reach the AI service — check your network connection.[/red]\n")
                history.pop()
                break
            except anthropic.RateLimitError as e:
                if _spinner: _spinner.stop()
                _elog.error("RateLimitError | %s", e)
                _console.print("\n  [yellow]⏳ The AI service is temporarily busy — please wait a moment and try again.[/yellow]\n")
                history.pop()
                break
            except anthropic.AuthenticationError as e:
                if _spinner: _spinner.stop()
                _elog.error("AuthenticationError | %s", e)
                _console.print("\n  [red]✗ Invalid API key.[/red] [dim]Use[/dim] [cyan]/setkey sk-ant-...[/cyan] [dim]to update it.[/dim]\n")
                client = None
                api_key = ""
                history.pop()
                break
            except KeyboardInterrupt:
                if _spinner: _spinner.stop()
                _console.print("\n  [dim yellow]⊗ Cancelled — AI response interrupted.[/dim yellow]\n")
                history.pop()
                break
            except anthropic.APIStatusError as e:
                if _spinner: _spinner.stop()
                _msg = str(e).lower()
                if "credit" in _msg or "billing" in _msg or "balance" in _msg:
                    _console.print("\n  [red]✗ Anthropic API credit balance is empty.[/red]")
                    _console.print("  [dim]Top up at[/dim] [cyan]https://console.anthropic.com/settings/billing[/cyan]\n")
                else:
                    _console.print("\n  [red]✗ The AI service returned an unexpected response — please try again.[/red]\n")
                history.pop()
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        _console.print(f"  [bold bright_blue]🔧 Tool[/bold bright_blue] [cyan]{block.name}[/cyan] [dim](Ctrl+C to cancel)[/dim]")
                        _cancel_event.clear()
                        _tool_result: list = [None]

                        def _run_tool():
                            try:
                                _tool_result[0] = execute_tool(block.name, block.input, api_key)
                            except Exception as _e:
                                _elog.error("Tool execution error | tool=%s | %s: %s", block.name, type(_e).__name__, _e)
                                _tool_result[0] = {"error": "The action could not be completed — check that the target is running and reachable.", "results": []}

                        _t = threading.Thread(target=_run_tool, daemon=True)
                        _t.start()
                        try:
                            while _t.is_alive():
                                _t.join(timeout=0.25)
                        except KeyboardInterrupt:
                            _cancel_event.set()
                            print("\n  [Cancelling — waiting for current step to finish...]\n", flush=True)
                            _t.join(timeout=15)
                            _cancel_event.clear()
                            _tool_result[0] = {"aborted": True, "error": "Cancelled by user.", "results": []}

                        result = _tool_result[0]
                        if block.name in _SCAN_TOOLS:
                            _session["scans_run"] += 1
                            # Persist scan result for /report and /export
                            result.setdefault("scan_timestamp", datetime.now().isoformat())
                            _session["last_scan"]     = result
                            _session["last_scan_url"] = block.input.get("url", "")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                history.append({"role": "assistant", "content": response.content})
                history.append({"role": "user", "content": tool_results})
            else:
                history.append({"role": "assistant", "content": response.content})
                _save_history(history)
                break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat interface for the AI Pentest Assistant"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)"
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(
            "\n  Note: ANTHROPIC_API_KEY is not set.\n"
            "  All direct scan commands work without a key:\n"
            "    /scan <url> [xss|sqli|cmdi|ssti|all]\n"
            "    /crawl <url>\n"
            "    /auth <url> <user> <pass>\n"
            "    /targets  /vulns  /report  /export\n"
            "  Use /setkey sk-ant-... inside the chat to enable AI-powered conversation.\n"
        )

    chat(api_key)


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\n  [Session ended — goodbye.]")
    except Exception as _exc:
        print(f"\n  [Unexpected error — {type(_exc).__name__}: {_exc}]")
        print("  Please report this issue if it persists.")
        sys.exit(1)
