"""
AI Pentest Assistant — Generic Web Vulnerability Scanner

Crawls any URL, discovers injection points, and runs the trained RL agent
against each one. Works on any web application, not just DVWA/Juice Shop.

Minimum usage:
    python -m agents.scan --url http://target.com/search
    python -m agents.scan --url http://target.com/search --vuln xss
    python -m agents.scan --url http://target.com/login  --vuln sqli

By default --vuln all tests every vulnerability type (xss, sqli, cmdi, ssti).
The universal model is selected automatically for each type; types whose model
is not yet trained are skipped with a warning.
An ANTHROPIC_API_KEY enables LLM-assisted crawling; without it, the static
HTML parser is used automatically (no --no-llm-crawl flag needed).

Optional flags:
    --vuln all|xss|sqli|cmdi|ssti  Vulnerability types to scan (default: all)
    --episodes 20              Episodes per injection point (default: 20)
    --cookie "session=abc123"  Session cookie for authenticated scans
    --username admin           Username to log in with (auto-detects login form)
    --password secret          Password for --username
    --login-url http://.../login  Login page URL (defaults to --url if omitted)
    --model path/to/model      Override the default universal model
    --output results.json      Custom report path
    --api-key sk-ant-...       Override ANTHROPIC_API_KEY env var
    --no-llm-crawl             Force static-only crawl (no LLM)
"""

import argparse
import contextlib
import io
import json
import logging
import os
import re
import sys
import time
from collections import deque
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from environments.dynamic_env import (
    make_dynamic_env,
    XSS_ACTION_TO_FAMILY,
    SQLI_ACTION_TO_FAMILY,
    CMDI_ACTION_TO_FAMILY,
    SSTI_ACTION_TO_FAMILY,
)
from utils.auth_helper import authenticate, authenticate_basic, detect_auth_type, parse_cookies
from utils.generic_http_client import GenericHttpClient, InjectionPoint
from utils.model_loader import load_model
from utils.llm_payload_generator import LLMPayloadGenerator
from utils.response_analyzer import ResponseAnalyzer
from utils.web_crawler import LLMCrawler
from utils.heuristic_checks import run_all_heuristic_checks
from utils.logger import setup_logging

# ---------------------------------------------------------------------------
# Adaptive scanning constants
# ---------------------------------------------------------------------------

# Q-spread (max Q - second-max Q) below this → agent is uncertain → LLM override
Q_SPREAD_THRESHOLD = 0.5

# If a family fails this many consecutive steps without reward gain → switch family
FAMILY_FAILURE_THRESHOLD = 3

logger = logging.getLogger(__name__)

# Default model paths — selected automatically based on --vuln.
# Universal models (multi-target) are preferred; fall back to single-target if
# the universal model hasn't been trained yet.
DEFAULT_MODELS = {
    "xss":  "models/universal_xss_dqn/xss_dqn_final",
    "sqli": "models/universal_sqli_dqn/sqli_dqn_final",
    "cmdi": "models/cmdi_dqn_curriculum_high_20260226_193546/cmdi_dqn_high_5000_steps",
    "ssti": "models/ssti_dqn_low_20260228_175703/ssti_dqn_final",
}

# ---------------------------------------------------------------------------
# Color helpers (graceful fallback if colorama is not installed)
# ---------------------------------------------------------------------------

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    _RED    = Fore.RED    + Style.BRIGHT
    _GREEN  = Fore.GREEN  + Style.BRIGHT
    _YELLOW = Fore.YELLOW + Style.BRIGHT
    _CYAN   = Fore.CYAN
    _RESET  = Style.RESET_ALL
except ImportError:
    _RED = _GREEN = _YELLOW = _CYAN = _RESET = ""


def _red(text):    return f"{_RED}{text}{_RESET}"
def _green(text):  return f"{_GREEN}{text}{_RESET}"
def _yellow(text): return f"{_YELLOW}{text}{_RESET}"
def _cyan(text):   return f"{_CYAN}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Q-value helpers for adaptive scanning
# ---------------------------------------------------------------------------

def _get_q_values(model, obs: np.ndarray):
    """Extract raw Q-values from the DQN policy network. Returns None on failure."""
    try:
        import torch
        obs_tensor, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            return model.policy.q_net(obs_tensor).cpu().numpy().squeeze()
    except Exception:
        return None


def _q_confidence(q_values) -> float:
    """Max Q minus second-max Q. Low value = agent uncertain between actions."""
    if q_values is None or len(q_values) < 2:
        return float("inf")
    s = np.sort(q_values)[::-1]
    return float(s[0] - s[1])


# ---------------------------------------------------------------------------
# Adaptive scan session — wraps one injection-point episode with LLM steering
# ---------------------------------------------------------------------------

class AdaptiveScanSession:
    """
    Per-injection-point adaptive controller.

    Each step it checks two trigger conditions:
      1. Q-spread < Q_SPREAD_THRESHOLD  → agent is uncertain
      2. Current RL-preferred family has no_gain_streak >= FAMILY_FAILURE_THRESHOLD

    When triggered (and LLM is available), it calls suggest_family() to get a
    better family and forces that action instead of the greedy RL choice.
    Falls back silently to RL when the LLM is unavailable or returns nothing.
    """

    def __init__(
        self,
        model,
        vuln_type: str,
        action_to_family: dict,
        llm_gen,
        q_threshold: float = Q_SPREAD_THRESHOLD,
        family_fail_thresh: int = FAMILY_FAILURE_THRESHOLD,
    ):
        self.model = model
        self.vuln_type = vuln_type
        self.action_to_family = action_to_family
        self.family_to_action = {
            f: a for a, f in action_to_family.items() if f is not None
        }
        self.available_families = list(self.family_to_action.keys())
        self.llm_gen = llm_gen
        self.q_threshold = q_threshold
        self.family_fail_thresh = family_fail_thresh

        # Reset per episode
        self.family_stats: dict = {}    # {family: {attempts, no_gain_streak, last_snippet}}
        self.response_history: deque = deque(maxlen=10)
        self.adaptation_log: list = []    # LLM override events this episode

    def reset(self):
        self.family_stats = {}
        self.response_history = deque(maxlen=10)
        self.adaptation_log = []

    def select_action(self, obs) -> tuple[int, str]:
        """
        Choose the next action. Returns (action_id, reason_str).
        reason_str is one of: 'rl_greedy', 'llm_override', 'rl_fallback'.
        """
        q_values = _get_q_values(self.model, obs)

        if q_values is None:
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action), "rl_fallback"

        best_action = int(np.argmax(q_values))
        best_family = self.action_to_family.get(best_action)
        q_spread = _q_confidence(q_values)

        # Check trigger conditions
        family_stuck = (
            best_family is not None
            and self.family_stats.get(best_family, {}).get("no_gain_streak", 0)
            >= self.family_fail_thresh
        )
        uncertain = q_spread < self.q_threshold

        if (family_stuck or uncertain) and self.llm_gen and self.llm_gen.available:
            tried = [
                {
                    "family": f,
                    "attempts": s["attempts"],
                    "no_gain_streak": s["no_gain_streak"],
                    "last_snippet": s.get("last_snippet", ""),
                }
                for f, s in self.family_stats.items()
            ]
            suggested = self.llm_gen.suggest_family(
                vuln_type=self.vuln_type,
                tried_families_with_outcomes=tried,
                response_history=list(self.response_history)[-3:],
                available_families=self.available_families,
            )
            if suggested and suggested in self.family_to_action:
                forced = self.family_to_action[suggested]
                reason = "uncertain" if uncertain else "stuck"
                self.adaptation_log.append({
                    "from_family": best_family,
                    "to_family": suggested,
                    "trigger": reason,
                    "q_spread": round(q_spread, 3),
                })
                return forced, f"llm_override({best_family}→{suggested})"

        return best_action, "rl_greedy"

    def record_step(self, action_id: int, reward: float, info: dict):
        """Update per-family stats and response history after each step."""
        family = self.action_to_family.get(action_id)
        if family is None:
            return

        if family not in self.family_stats:
            self.family_stats[family] = {
                "attempts": 0, "no_gain_streak": 0, "last_snippet": ""
            }

        self.family_stats[family]["attempts"] += 1
        if reward > 0:
            self.family_stats[family]["no_gain_streak"] = 0
        else:
            self.family_stats[family]["no_gain_streak"] += 1

        pi = info.get("payload_info", {})
        snippet = str(pi.get("response_snippet", pi.get("payload", "")))[:200]
        self.family_stats[family]["last_snippet"] = snippet

        self.response_history.append({
            "payload": pi.get("payload", ""),
            "reward": reward,
            "reflected": pi.get("reflected", False),
            "snippet": snippet,
        })


# ---------------------------------------------------------------------------
# Single injection point evaluation
# ---------------------------------------------------------------------------

def run_episodes(
    model,
    env,
    num_episodes: int,
    injection_point: InjectionPoint,
    adaptive_session: "AdaptiveScanSession | None" = None,
    early_stop_rate: float = 0.8,
    early_stop_min_eps: int = 3,
    early_giveup_min_eps: int = 3,
    early_giveup_max_reward: float = 10.0,
    time_limit: float = 90.0,
    progress_callback=None,
) -> dict:
    """
    Run the agent against one injection point for N episodes.

    Early-stop: if success_rate >= early_stop_rate after at least
    early_stop_min_eps episodes, stop early — the point is confirmed
    vulnerable.  This saves both time and LLM API calls.

    progress_callback: optional callable(ep_num, num_episodes, ep_reward, successes)
    called after each episode to report progress.
    """
    successes = 0
    rewards = []
    steps_list = []
    successful_payloads = []
    total_llm_overrides = 0
    all_adaptation_logs = []
    episodes_run = 0
    payload_log: list[dict] = []  # Every payload attempt across all episodes

    best_response_snippet = ""
    vuln_subtype = ""  # Populated by Dynamic*Env when vulnerability is confirmed
    impact_evidence = []  # Proof-of-impact evidence lines

    _point_start = time.monotonic()

    for ep in range(num_episodes):
        # Per-point time limit — abort if we've spent too long on this point
        if time.monotonic() - _point_start > time_limit:
            logger.debug(
                "Time limit (%.0fs) reached after %d episodes — moving on",
                time_limit, episodes_run,
            )
            break

        try:
            obs, _ = env.reset()
        except KeyboardInterrupt:
            break

        ep_reward = 0.0
        ep_steps = 0
        done = False
        best_payload = ""

        if adaptive_session is not None:
            adaptive_session.reset()

        while not done:
            if adaptive_session is not None:
                action, reason = adaptive_session.select_action(obs)
                if "llm_override" in reason:
                    total_llm_overrides += 1
            else:
                action, _ = model.predict(obs, deterministic=True)

            try:
                obs, reward, terminated, truncated, info = env.step(int(action))
            except KeyboardInterrupt:
                done = True
                break

            ep_reward += reward
            ep_steps += 1
            done = terminated or truncated

            if adaptive_session is not None:
                adaptive_session.record_step(int(action), reward, info)

            pi = info.get("payload_info", {})
            p = pi.get("payload", "")

            # Log every payload attempt for the /payloads command
            if p:
                payload_log.append({
                    "episode": ep + 1,
                    "step": ep_steps,
                    "action": int(action),
                    "payload": p,
                    "parameter": pi.get("parameter", ""),
                    "reward": float(reward),
                    "success": bool(terminated and done),
                    "reflected": pi.get("reflected", False),
                    "response_snippet": str(pi.get("response_snippet", ""))[:300],
                    "url": pi.get("url_path", "") or pi.get("full_request_url", ""),
                })

            if terminated and p:
                best_payload = p
                if not best_response_snippet:
                    best_response_snippet = str(pi.get("response_snippet", ""))[:400]
                # Capture vuln subtype from successful payload info
                for key in ("xss_subtype", "sqli_subtype", "cmdi_subtype", "ssti_subtype"):
                    if pi.get(key):
                        vuln_subtype = pi[key]
                # Capture proof-of-impact evidence (keep first successful set)
                if pi.get("evidence") and not impact_evidence:
                    impact_evidence = pi["evidence"]
            elif p and reward > 20 and not best_payload:
                best_payload = p

        if terminated:
            successes += 1
            if best_payload:
                successful_payloads.append(best_payload)

        if adaptive_session is not None and adaptive_session.adaptation_log:
            all_adaptation_logs.extend(
                [{"episode": ep, **e} for e in adaptive_session.adaptation_log]
            )

        rewards.append(ep_reward)
        steps_list.append(ep_steps)
        episodes_run = ep + 1

        if progress_callback is not None:
            progress_callback(episodes_run, num_episodes, ep_reward, successes)

        # Early-stop: confirmed vulnerable — no need to keep testing
        if episodes_run >= early_stop_min_eps:
            current_rate = successes / episodes_run
            if current_rate >= early_stop_rate:
                logger.debug(
                    "Early stop: %d/%d succeeded (%.0f%%) after %d episodes",
                    successes, episodes_run, current_rate * 100, episodes_run,
                )
                break

        # Early give-up: likely safe — no signal after several episodes
        if (
            episodes_run >= early_giveup_min_eps
            and successes == 0
            and max(rewards) < early_giveup_max_reward
        ):
            logger.debug(
                "Early give-up: 0/%d succeeded, best reward=%.1f after %d episodes",
                episodes_run, max(rewards), episodes_run,
            )
            break

    success_rate = successes / episodes_run if episodes_run > 0 else 0.0
    unique_payloads = list(dict.fromkeys(successful_payloads))

    # Try to read subtype from the env itself (Dynamic*Env stores it)
    if not vuln_subtype:
        for attr in ("xss_subtype", "sqli_subtype", "cmdi_subtype", "ssti_subtype"):
            if hasattr(env, attr) and getattr(env, attr):
                vuln_subtype = getattr(env, attr)
                break

    result = {
        "injection_point": str(injection_point),
        "url": injection_point.url,
        "parameter": injection_point.parameter,
        "method": injection_point.method,
        "nav_hint": injection_point.nav_hint,
        "num_episodes": episodes_run,
        "episodes_requested": num_episodes,
        "early_stopped": episodes_run < num_episodes,
        "successes": successes,
        "success_rate": success_rate,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "successful_payloads": unique_payloads[:5],
        "response_snippet": best_response_snippet,
        "vuln_subtype": vuln_subtype,
        "impact_evidence": impact_evidence,
        "payload_log": payload_log,
    }
    if adaptive_session is not None:
        result["llm_overrides"] = total_llm_overrides
        result["adaptation_log"] = all_adaptation_logs
    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_banner(
    url: str,
    vuln_types: "list[str] | str",
    model_path: str,
    api_key: str,
    llm_payloads: bool,
    adaptive: bool = False,
    username: str = "",
    heuristic: bool = True,
):
    crawl_mode = "LLM-assisted" if api_key else "static HTML parser"
    payload_mode = "LLM-generated" if (api_key and llm_payloads) else "static list"
    adaptive_mode = (
        _yellow("ON (LLM family steering)") if (adaptive and api_key)
        else ("ON (statistical fallback)" if adaptive else "OFF")
    )
    heuristic_mode = _green("ON") if heuristic else "OFF"
    auth_mode = _green(f"authenticated as '{username}'") if username else "unauthenticated"
    if isinstance(vuln_types, list):
        scanning_label = ", ".join(v.upper() for v in vuln_types)
        model_label = "auto (per vuln type)"
    else:
        scanning_label = vuln_types.upper()
        model_label = model_path
    print("\n" + "=" * 68)
    print(_cyan("  AI PENTEST ASSISTANT -- GENERIC SCANNER"))
    print("=" * 68)
    print(f"  Target    : {url}")
    print(f"  Scanning  : {scanning_label}")
    print(f"  Auth      : {auth_mode}")
    print(f"  Model     : {model_label}")
    print(f"  Crawl     : {crawl_mode}")
    print(f"  Payloads  : {payload_mode}")
    print(f"  Adaptive  : {adaptive_mode}")
    print(f"  Heuristic : {heuristic_mode}")
    print("=" * 68 + "\n")


def print_progress(current: int, total: int, method: str, url: str, param: str, episodes: int = 0, score: float = 0):
    """Print a clear per-injection-point header."""
    tier = _red("HIGH") if score >= 60 else (_yellow("MED") if score >= 30 else "LOW")
    eps_str = f"  [{episodes} eps]" if episodes else ""
    print(f"\n  [{current}/{total}] {tier} Testing {_cyan(method)} "
          f"param={_cyan(repr(param))}{eps_str}  ({url})")


def print_inline_result(result: dict, vuln_type: str):
    """Print immediate result after each injection point finishes."""
    if result.get("waf_blocked"):
        return  # Already printed by WAF check
    early = ""
    if result.get("early_stopped"):
        early = f" (early-stop: {result['num_episodes']}/{result['episodes_requested']} eps)"
    if result["success_rate"] > 0:
        subtype = result.get("vuln_subtype", "")
        subtype_str = f"  [{subtype}]" if subtype else ""
        header = _red(f"  [VULNERABLE]  param={result['parameter']!r} "
                      f"via {result['method']} {result['url']}{subtype_str}")
        print(header + early)
        hint = result.get("nav_hint", "")
        if hint:
            print(f"  How to find it: {hint}")
        for p in result["successful_payloads"]:
            print(f"    payload: {_yellow(p)}")
        # Show proof-of-impact evidence
        for ev in result.get("impact_evidence", []):
            print(f"    {_cyan('evidence')}: {ev}")
    else:
        print(_green(f"  [SAFE]  param={result['parameter']!r} "
                     f"-- no {vuln_type.upper()} confirmed") + early)


def print_results_table(all_results: list[dict], vuln_type: str):
    print("\n" + "=" * 68)
    print(f"  SCAN COMPLETE -- {vuln_type.upper()} RESULTS")
    print("=" * 68)

    # Show subtype column for all vuln types
    _subtype_label = {
        "xss": "XSS Type",
        "sqli": "SQLi Type",
        "cmdi": "CMDi Type",
        "ssti": "SSTI Engine",
    }
    subtype_header = _subtype_label.get(vuln_type, "Type")

    print(
        f"  {'Parameter':<15} {'Method':<6} {subtype_header:<18} {'Success':>8} "
        f"{'Avg Reward':>11} {'Avg Steps':>10}"
    )
    print("  " + "-" * 72)

    for r in all_results:
        rate_str = f"{r['success_rate']:.0%}"
        rate_col = _red(f"{rate_str:>8}") if r["success_rate"] > 0 else f"{rate_str:>8}"
        param = r["parameter"][:14]
        subtype = r.get("vuln_subtype", "")[:17] if r["success_rate"] > 0 else ""
        print(
            f"  {param:<15} {r['method']:<6} {subtype:<18}"
            f"{rate_col} "
            f"{r['mean_reward']:>11.1f} "
            f"{r['mean_steps']:>10.1f}"
        )

    print("  " + "=" * 72)

    vulnerabilities = [r for r in all_results if r["success_rate"] > 0]
    if vulnerabilities:
        print(f"\n  {_red(f'VULNERABLE PARAMETERS FOUND: {len(vulnerabilities)}')}\n")
        for r in vulnerabilities:
            subtype = r.get("vuln_subtype", "")
            if subtype:
                print(f"  {r['method']} {r['url']}  [{_red(subtype)}]")
            else:
                print(f"  {r['method']} {r['url']}")
            hint = r.get("nav_hint", "")
            if hint:
                print(f"  Navigate  : {hint}")
            print(f"  Parameter : {r['parameter']!r}")
            if subtype:
                print(f"  Type      : {subtype}")
            for p in r["successful_payloads"]:
                print(f"  Payload   : {_yellow(p)}")
            # Show proof-of-impact evidence
            evidence = r.get("impact_evidence", [])
            if evidence:
                print(f"  {'-' * 40}")
                print(f"  {_cyan('Impact Evidence:')}")
                for ev in evidence:
                    print(f"    > {ev}")
            print()
    else:
        print(f"\n  {_green('No vulnerabilities confirmed on the tested endpoints.')}\n")


# ---------------------------------------------------------------------------
# Injection point priority scoring
# ---------------------------------------------------------------------------
#
# Universal scoring — works on any web app regardless of URL naming.
# Based on two things that are ALWAYS true:
#
#   1. ATTACKER CONTROL — can an external attacker realistically set this
#      value? A query string param: always. A CSRF token: never.
#      This is determined by input_type + param name patterns.
#
#   2. SINK LIKELIHOOD — does the parameter name suggest the value reaches
#      a dangerous backend operation (DB query, OS command, file read,
#      template render, HTML output)?  This is language/framework agnostic
#      because developers everywhere use similar names for similar things.
#
# URL paths are NOT scored — /x7f2a and /admin.php are treated equally.

# ── Attacker control score (0-50) ──────────────────────────────────────────
#
# How easily can an external attacker set this value?
#
# 50: Trivially controllable — attacker crafts the URL or fills a text box
# 30: Controllable via form submission but constrained (dropdowns, etc.)
# 10: Technically controllable but rarely reaches app logic (headers)
#  0: Not attacker-controlled (CSRF tokens, server-set values)

_ATTACKER_CONTROL = {
    # Fully controllable — attacker builds the URL or types in the box
    "url_param":  50,
    "text":       45,
    "search":     45,
    "textarea":   45,
    "form_field": 40,   # generic input — assume text-like until proven otherwise
    # Partially controllable — attacker can submit but value is constrained
    "hidden":     25,   # often IDs or state — attacker can tamper via devtools/proxy
    "number":     20,   # numeric input — can overflow or inject via proxy
    "password":   30,   # may reach backend unhashed (SQLi auth bypass)
    "email":      15,   # reaches backend but usually validated
    "select":     10,   # fixed options — but server may not re-validate
    "radio":      10,
    # Low controllability in real attacks
    "header":      5,   # X-Forwarded-For etc. — only controllable via proxy/infra
}

# ── Sink likelihood score (0-50) ───────────────────────────────────────────
#
# Does the parameter name suggest the value is used in a dangerous operation?
# These patterns are universal — developers name things the same way in PHP,
# Python, Java, Node, Go, .NET, Ruby, etc.
#
# The patterns are grouped by WHAT THEY DO, not by vulnerability type.
# A param named "id" is dangerous for SQLi AND IDOR AND access control.

# Parameters whose values almost certainly reach a dangerous sink
_SINK_HIGH = re.compile(r"(?i)^(" + "|".join([
    # Database queries — the value ends up in a WHERE/ORDER/SELECT
    r"id", r"user_id", r"uid", r"pid", r"cid", r"item_id", r"product_id",
    r"account", r"profile", r"category", r"cat",
    r"query", r"search", r"q", r"s", r"keyword", r"term", r"find", r"filter",
    r"sort", r"order", r"column", r"col", r"field", r"group_by", r"order_by",
    r"where", r"having", r"table", r"db",
    # File system operations — the value becomes a file path
    r"file", r"filename", r"filepath", r"path", r"dir", r"directory",
    r"folder", r"doc", r"document", r"attachment",
    r"include", r"require", r"load", r"read", r"open", r"fetch",
    r"page", r"pg", r"p", r"view", r"tpl", r"template", r"layout", r"theme",
    r"src", r"source", r"resource",
    # OS commands — the value is passed to a shell or process
    r"cmd", r"command", r"exec", r"run", r"shell",
    r"ping", r"target", r"host", r"ip", r"address", r"server",
    r"domain", r"dns", r"port",
    # URL/redirect — the value becomes a URL the server fetches or redirects to
    r"url", r"uri", r"redirect", r"return", r"next", r"goto",
    r"link", r"href", r"callback", r"continue", r"dest", r"destination",
    # HTML output — the value is rendered in a page (XSS sink)
    r"name", r"username", r"user", r"title", r"subject",
    r"comment", r"message", r"msg", r"text", r"body", r"content",
    r"description", r"desc", r"bio", r"about",
    r"feedback", r"review", r"note", r"label",
    r"input", r"data", r"value", r"param", r"payload",
    # Template rendering — the value is embedded in a template engine
    r"render", r"format", r"expression", r"expr", r"eval",
    r"greeting", r"email_body", r"email_subject",
    # Dispatch/routing — the value controls server-side behavior
    r"action", r"type", r"mode", r"func", r"method", r"module",
    r"class", r"handler", r"controller", r"op", r"operation",
    r"lang", r"locale", r"language",
]) + r")$")

# Parameters whose values almost never reach a dangerous sink
_SINK_NONE = re.compile(r"(?i)^(" + "|".join([
    # Anti-CSRF tokens — server-generated, server-validated, never in a query
    r"csrf", r"_?csrf_?token", r"nonce", r"_token",
    r"csrfmiddlewaretoken", r"__RequestVerificationToken",
    r"authenticity_token", r"_xsrf",
    # Submit buttons — the value is "Submit" or "Save", never processed
    r"submit", r"button", r"btn", r"save", r"cancel", r"reset",
    # Confirmation fields — duplicate of another field, validated client-side
    r".*_?confirm", r".*_?confirmation", r"confirm_.*", r"retype_.*", r"repeat_.*",
    # CAPTCHAs — validated by external service, not the app
    r"captcha", r"recaptcha", r"g-recaptcha.*", r"h-captcha.*",
    # Booleans/checkboxes — "on"/"off"/"1"/"0", never interpolated
    r"remember", r"remember_me", r"stay_logged", r"keep_me",
    r"agree", r"tos", r"consent", r"newsletter",
    # Timestamps — server-set or display-only
    r"timestamp", r"ts", r"_ts", r"time", r"date",
    r"created_?at", r"updated_?at", r"modified",
    # Pagination — integer values, rarely vulnerable in practice
    r"page_?size", r"per_?page", r"rows",
    # Content negotiation — not used in business logic
    r"encoding", r"charset", r"accept",
]) + r")$")

# Everything else gets a moderate sink score — unknown params could go either way


# ── HTML context analysis ─────────────────────────────────────────────────
#
# When the param name is unknown (doesn't match _SINK_HIGH or _SINK_NONE),
# we analyse the surrounding HTML to infer what the field actually does.
# This works even with obfuscated param names like "f1", "inp_3", "a".

# Keywords extracted from labels/placeholders/aria that suggest a high-risk sink
_CTX_HIGH_KEYWORDS = re.compile(r"(?i)(" + "|".join([
    # User-facing text that will be rendered (XSS sinks)
    r"search", r"comment", r"message", r"name", r"title",
    r"description", r"feedback", r"review", r"note", r"post",
    r"reply", r"subject", r"bio", r"about", r"content", r"text",
    r"write", r"compose", r"editor", r"input",
    # File/path operations
    r"file", r"upload", r"path", r"directory", r"folder",
    r"import", r"include", r"load", r"attach",
    # URL/redirect
    r"url", r"link", r"redirect", r"website", r"homepage",
    r"goto", r"return", r"callback",
    # Database lookups
    r"query", r"filter", r"find", r"lookup", r"select",
    r"category", r"product", r"item", r"account", r"profile",
    # Command/execution
    r"command", r"execute", r"run", r"ping", r"host",
    r"server", r"address", r"domain", r"target",
    # Template/eval
    r"template", r"expression", r"format", r"render",
]) + r")")

# Keywords that suggest a low-risk / infrastructure field
_CTX_LOW_KEYWORDS = re.compile(r"(?i)(" + "|".join([
    r"captcha", r"token", r"csrf", r"nonce",
    r"password", r"confirm.*password", r"re.?type",
    r"remember", r"agree", r"terms", r"consent",
    r"submit", r"cancel", r"reset", r"back",
    r"page.*size", r"per.*page", r"rows.*per",
    r"timezone", r"locale", r"language", r"currency",
]) + r")")

# Form action paths that suggest the endpoint processes user input dangerously
_ACTION_HIGH = re.compile(r"(?i)(" + "|".join([
    r"search", r"comment", r"post", r"message", r"feedback",
    r"upload", r"import", r"exec", r"eval", r"ping",
    r"query", r"filter", r"redirect", r"forward",
    r"profile", r"edit", r"update", r"save", r"create",
    r"register", r"signup", r"contact",
]) + r")")


def _analyze_html_context(context_html: str, param_name: str) -> int | None:
    """
    Analyse the HTML snippet surrounding an injection point to infer its purpose.

    Extracts text from:
      - <label> elements (especially those with for= matching the param)
      - placeholder attributes
      - aria-label / aria-describedby attributes
      - Visible text near the input (parent container text)
      - Form action URL path

    Returns a sink score (0-50) if context provides a clear signal, or None if
    the context is empty / inconclusive (caller should fall back to default 20).
    """
    if not context_html:
        return None

    from bs4 import BeautifulSoup as _BS
    snippet = _BS(context_html, "lxml")

    signals: list[str] = []

    # 1. Labels — most reliable signal: <label for="f1">Search query</label>
    for label in snippet.find_all("label"):
        label_for = (label.get("for") or "").lower()
        label_text = label.get_text(strip=True).lower()
        # Match label to our param by for= attribute or proximity
        if label_for == param_name.lower() or label_text:
            signals.append(label_text)

    # 2. The input element itself — placeholder, aria-label, title attribute
    for inp in snippet.find_all(["input", "textarea", "select"]):
        inp_name = (inp.get("name") or "").lower()
        if inp_name == param_name.lower() or not inp_name:
            if inp.get("placeholder"):
                signals.append(inp["placeholder"].lower())
            if inp.get("aria-label"):
                signals.append(inp["aria-label"].lower())
            if inp.get("title"):
                signals.append(inp["title"].lower())

    # 3. Form action path
    form_tag = snippet.find("form")
    if form_tag:
        action = (form_tag.get("action") or "").lower()
        signals.append(action)

    # 4. Page-level context (title, headings injected for URL params)
    for tag in snippet.find_all(["title", "h1", "h2", "h3"]):
        signals.append(tag.get_text(strip=True).lower())

    # 5. Visible text in the container (trimmed)
    container_text = snippet.get_text(" ", strip=True).lower()[:200]
    signals.append(container_text)

    combined = " ".join(signals)
    if not combined.strip():
        return None

    # Count high and low keyword matches
    high_hits = len(_CTX_HIGH_KEYWORDS.findall(combined))
    low_hits = len(_CTX_LOW_KEYWORDS.findall(combined))

    if high_hits > 0 and high_hits > low_hits:
        # Scale: 1 hit = 35, 2+ hits = 45, 3+ = 50
        return min(30 + high_hits * 10, 50)
    elif low_hits > 0 and low_hits >= high_hits:
        return max(5, 15 - low_hits * 5)
    else:
        return None   # no clear signal — caller uses default


def _score_injection_point(pt: InjectionPoint, vuln_type: str = "") -> float:
    """
    Score an injection point from 0-100 based on universal signals.

    Three dimensions:
      1. Attacker control (0-50): can an external attacker realistically set this value?
      2. Sink likelihood  (0-50): does the param name suggest it reaches a dangerous op?
      3. HTML context (fallback):  if param name is unknown, analyse surrounding HTML
         (labels, placeholders, form action, visible text) to infer what the field does.
    """
    param = pt.parameter or ""

    # Dimension 1: Attacker control (0-50)
    control = _ATTACKER_CONTROL.get(pt.input_type, 20)

    # Dimension 2: Sink likelihood (0-50) — param name first
    if _SINK_HIGH.match(param):
        sink = 50
    elif _SINK_NONE.match(param):
        sink = 0
    else:
        # Dimension 3: HTML context analysis (fallback for unknown param names)
        ctx_score = _analyze_html_context(getattr(pt, "context_html", ""), param)
        sink = ctx_score if ctx_score is not None else 20

    return min(control + sink, 100.0)


def _prioritize_points(
    points: list[InjectionPoint], base_episodes: int, vuln_type: str = ""
) -> list[tuple[InjectionPoint, int]]:
    """
    Score, sort, and assign episode counts to injection points.

    HIGH  (score >= 60): full episodes, tested first  — value is attacker-controlled AND likely reaches a sink
    MED   (30-59):       75% episodes                 — one signal is strong, the other uncertain
    LOW   (< 30):        25% episodes, tested last    — unlikely to be exploitable

    Returns list of (point, episodes) tuples sorted high-priority first.
    """
    scored = []
    for pt in points:
        s = _score_injection_point(pt, vuln_type)
        if s >= 60:
            eps = base_episodes
        elif s >= 30:
            eps = max(3, int(base_episodes * 0.75))
        else:
            eps = max(3, int(base_episodes * 0.25))
        scored.append((s, pt, eps))

    scored.sort(key=lambda x: -x[0])

    # Log the priority ranking
    for rank, (s, pt, eps) in enumerate(scored, 1):
        tier = "HIGH" if s >= 60 else ("MED" if s >= 30 else "LOW")
        logger.debug(
            "Priority #%d [%s %.0f] %s %s param=%s (%d eps)",
            rank, tier, s, pt.method, pt.url, pt.parameter, eps,
        )

    return [(pt, eps) for _, pt, eps in scored]


def print_heuristic_results(findings: list[dict]):
    """Print heuristic check findings to console."""
    if not findings:
        print(f"\n  {_green('Heuristic checks: no issues found.')}\n")
        return

    print("\n" + "=" * 68)
    print(_cyan("  HEURISTIC SECURITY CHECKS"))
    print("=" * 68)
    print(f"  {_red(f'{len(findings)} issue(s) found')}\n")

    for i, f in enumerate(findings, 1):
        severity = f.get("severity", "Medium")
        sev_color = _red if severity == "High" else (_yellow if severity == "Medium" else _green)
        category = f.get("vuln_category", "Unknown")
        desc = f.get("description", "")
        url = f.get("url", "")
        param = f.get("parameter", "")
        method = f.get("method", "")

        print(f"  {i}. [{sev_color(severity)}] {category}")
        print(f"     {desc}")
        if url:
            print(f"     URL       : {method} {url}")
        if param:
            print(f"     Parameter : {param}")
        for p in f.get("successful_payloads", []):
            print(f"     Payload   : {_yellow(p)}")
        for ev in f.get("impact_evidence", []):
            print(f"     > {ev}")
        remediation = f.get("remediation", "")
        if remediation:
            print(f"     Fix       : {remediation}")
        print()


# ---------------------------------------------------------------------------
# Remediation recommendations
# ---------------------------------------------------------------------------

# Module-level client cache — keyed by api_key string so we never create more
# than one Anthropic client per key within a process lifetime.
_remediation_client_cache: dict = {}

_REMEDIATION_STATIC = {
    "sqli": [
        "Use parameterised queries / prepared statements — never concatenate user input into SQL strings.",
        "Apply an ORM (e.g. SQLAlchemy, Hibernate) that prevents raw query construction by default.",
        "Validate and whitelist input types (e.g. enforce integer IDs before they reach the query).",
        "Restrict database user privileges: the app account should have no DROP/INSERT rights it doesn't need.",
        "Enable a Web Application Firewall rule for SQL meta-characters as a secondary defence layer.",
    ],
    "xss": [
        "HTML-encode all user-supplied output with a context-aware library (e.g. OWASP Java Encoder, DOMPurify for JS).",
        "Adopt a Content Security Policy (CSP) header that disallows inline scripts and restricts script sources.",
        "Set HttpOnly and Secure flags on session cookies so a stolen cookie cannot be read by injected scripts.",
        "Use framework auto-escaping (e.g. Jinja2 autoescape=True, React JSX) instead of raw innerHTML.",
        "Validate input server-side against an allow-list of expected characters before storing or reflecting it.",
    ],
    "cmdi": [
        "Never pass user input to shell functions (os.system, subprocess with shell=True, exec, etc.).",
        "Use language-native APIs instead of shelling out (e.g. Python's zipfile module instead of calling `zip`).",
        "If a shell call is unavoidable, pass arguments as a list (subprocess.run(['cmd', arg])) — never a joined string.",
        "Validate input strictly against an allow-list of safe characters before any system call.",
        "Run the web server process under a dedicated low-privilege OS user with no shell access.",
    ],
    "ssti": [
        "Never render user-controlled strings as templates — keep template logic server-side.",
        "Use the template engine's sandbox mode or a restricted environment (e.g. Jinja2 SandboxedEnvironment).",
        "Treat template names/paths as configuration, not user input; validate against a strict allow-list.",
        "Upgrade the template engine to a version that has sandbox escapes patched.",
        "Apply Content Security Policy and output encoding as a defence-in-depth measure.",
    ],
}


def _fetch_page_context(url: str, cookies: dict, extra_headers: dict) -> str:
    """
    Fetch the page at *url* and return a trimmed HTML snippet useful for
    understanding what kind of input/form is present.
    Returns an empty string on any error.
    """
    try:
        fetch_url = url
        resp = requests.get(
            fetch_url,
            cookies=cookies,
            headers={**extra_headers, "User-Agent": "Mozilla/5.0"},
            timeout=8,
            allow_redirects=True,
        )
        html = resp.text
        # Keep only the first 2 000 chars — enough to see forms/inputs without
        # blowing out the token budget
        return html[:2000]
    except Exception:
        return ""


def generate_remediation_advice(
    vuln_type: str,
    vulnerable_results: list[dict],
    api_key: str = "",
    cookies: dict | None = None,
    extra_headers: dict | None = None,
) -> str:
    """
    Return remediation advice as a formatted string.

    When an API key is present, fetches each vulnerable page's HTML and the
    captured server response snippet so Claude can give advice that is specific
    to the actual page implementation — not just the vulnerability class.
    Falls back to static advice when the API is unavailable.
    """
    static_lines = _REMEDIATION_STATIC.get(vuln_type, [])
    cookies = cookies or {}
    extra_headers = extra_headers or {}

    if not api_key:
        return "\n".join(f"  • {line}" for line in static_lines)

    vuln_labels = {
        "sqli": "SQL Injection (CWE-89)",
        "xss":  "Cross-Site Scripting (CWE-79)",
        "cmdi": "OS Command Injection (CWE-78)",
        "ssti": "Server-Side Template Injection (CWE-94)",
    }
    label = vuln_labels.get(vuln_type, vuln_type.upper())

    # Build a per-endpoint evidence block
    endpoint_blocks = []
    for r in vulnerable_results:
        url       = r.get("url", "")
        param     = r.get("parameter", "")
        method    = r.get("method", "GET")
        payloads  = r.get("successful_payloads", [])[:3]
        snippet   = r.get("response_snippet", "")

        # Fetch the live page source so Claude can see the form/field context
        page_html = _fetch_page_context(url, cookies, extra_headers)

        block = f"Endpoint : {method} {url}\nParameter: {param!r}\n"
        if payloads:
            block += "Payloads that succeeded:\n" + "\n".join(f"  {p}" for p in payloads) + "\n"
        if snippet:
            block += f"Server response excerpt:\n  {snippet}\n"
        if page_html:
            block += f"Page source (first 2000 chars):\n```html\n{page_html}\n```\n"
        endpoint_blocks.append(block)

    endpoints_text = "\n---\n".join(endpoint_blocks)

    prompt = f"""\
A penetration test confirmed {label} on the following endpoint(s).
Full evidence — including the live page source — is provided below.

{endpoints_text}

Based on the actual page source and the payloads that worked, give 4-5 specific,
actionable remediation steps tailored to THIS page's implementation.

Requirements:
- Reference the exact parameter name ({", ".join(r.get("parameter","") for r in vulnerable_results)})
- Name the exact function, library, or framework feature to use (e.g. "replace cursor.execute(f'...') with cursor.execute('...', [value])")
- If you can infer the backend language or framework from the page source, say so and give language-specific advice
- If the page source shows a form, describe where in the template the fix should go
- Format as a numbered list. One sentence per item. No preamble."""

    try:
        import anthropic as _anthropic
        if api_key not in _remediation_client_cache:
            _remediation_client_cache[api_key] = _anthropic.Anthropic(api_key=api_key)
        client = _remediation_client_cache[api_key]
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        from utils.api_error_handler import handle_api_error
        handle_api_error(e, logger, context="remediation advice")
        return "\n".join(f"  • {line}" for line in static_lines)



def save_report(all_results: list[dict], url: str, vuln_type: str, output: str, username: str = "", remediation: str = "", heuristic_findings: list[dict] | None = None, static_findings: list[dict] | None = None):
    total_overrides = sum(r.get("llm_overrides", 0) for r in all_results)
    report = {
        "scan_timestamp": datetime.now().isoformat(),
        "target_url": url,
        "vuln_type": vuln_type,
        "authenticated_as": username if username else None,
        "total_points_tested": len(all_results),
        "vulnerable_points": sum(1 for r in all_results if r["success_rate"] > 0),
        "llm_overrides_total": total_overrides,
        "remediation": remediation or None,
        "results": all_results,
        "heuristic_checks": heuristic_findings or [],
        "static_analysis": static_findings or [],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {output}")
    return report


def save_report_multi(
    results_by_vuln: dict,
    remediations_by_vuln: dict,
    url: str,
    output: str,
    username: str = "",
    heuristic_findings: list[dict] | None = None,
    static_findings: list[dict] | None = None,
):
    """Save a combined report for an --vuln all scan."""
    all_flat = [r for results in results_by_vuln.values() for r in results]
    total_overrides = sum(r.get("llm_overrides", 0) for r in all_flat)
    vuln_summaries = {}
    for vt, results in results_by_vuln.items():
        vuln_summaries[vt] = {
            "points_tested": len(results),
            "vulnerable_points": sum(1 for r in results if r.get("success_rate", 0) > 0),
            "remediation": remediations_by_vuln.get(vt) or None,
            "results": results,
        }
    report = {
        "scan_timestamp": datetime.now().isoformat(),
        "target_url": url,
        "vuln_types": list(results_by_vuln.keys()),
        "authenticated_as": username if username else None,
        "total_points_tested": len(all_flat),
        "total_vulnerable_points": sum(
            s["vulnerable_points"] for s in vuln_summaries.values()
        ),
        "llm_overrides_total": total_overrides,
        "by_vuln_type": vuln_summaries,
        "heuristic_checks": heuristic_findings or [],
        "static_analysis": static_findings or [],
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Report saved to: {output}")
    return report


# ---------------------------------------------------------------------------
# Stored XSS chain test (register → login → sweep)
# ---------------------------------------------------------------------------

_SIGNUP_HINTS = re.compile(
    r"(?i)(signup|sign.?up|register|registration|create.?account|new.?user|enroll|join)"
)
_LOGIN_HINTS = re.compile(
    r"(?i)(login|log.?in|signin|sign.?in|auth|session)"
)
_USERNAME_HINTS_RE = re.compile(
    r"(?i)(user|uid|login|email|name|handle|nick|account)"
)

_XSS_CHAIN_PAYLOAD = '<script>alert("xssChainTest")</script>'
_XSS_CHAIN_PASSWORD = "ChainTestPwd123!"


def _find_signup_and_login_forms(
    injection_points: list[InjectionPoint],
) -> tuple[list[list[InjectionPoint]], list[list[InjectionPoint]]]:
    """
    From the discovered injection points, identify signup and login forms.
    Returns (signup_points, login_points) — grouped by URL.
    """
    signup_points: dict[str, list[InjectionPoint]] = {}
    login_points: dict[str, list[InjectionPoint]] = {}

    for pt in injection_points:
        if pt.method != "POST" or pt.input_type != "form_field":
            continue
        url_lower = pt.url.lower()
        if _SIGNUP_HINTS.search(url_lower):
            signup_points.setdefault(pt.url, []).append(pt)
        elif _LOGIN_HINTS.search(url_lower):
            login_points.setdefault(pt.url, []).append(pt)

    return (
        list(signup_points.values()),
        list(login_points.values()),
    )


def run_stored_xss_chain_test(
    injection_points: list[InjectionPoint],
    crawled_pages: list[str],
    cookies: dict | None = None,
    extra_headers: dict | None = None,
    base_url: str = "",
) -> list[dict]:
    """
    Multi-step stored XSS test:
      1. Find registration (signup) and login forms from injection points
      2. Register a new user with XSS payload as username
      3. Log in as that user
      4. Sweep all crawled pages for the XSS payload

    This catches stored XSS where the username/email is persisted in the DB
    and displayed on authenticated pages (index, profile, admin panel, etc.)
    without sanitisation.

    Returns a list of finding dicts compatible with scan results.
    """
    signup_groups, login_groups = _find_signup_and_login_forms(injection_points)

    if not signup_groups:
        logger.debug("[XSS chain] No signup forms found — skipping chain test")
        return []
    if not login_groups:
        logger.debug("[XSS chain] No login forms found — skipping chain test")
        return []

    print(f"\n  [XSS Chain] Found {len(signup_groups)} signup form(s), "
          f"{len(login_groups)} login form(s)")

    findings: list[dict] = []
    payload = _XSS_CHAIN_PAYLOAD
    payload_lower = payload.lower()
    test_pwd = _XSS_CHAIN_PASSWORD

    for signup_form_points in signup_groups:
        signup_url = signup_form_points[0].url
        # Identify which field is the username (identity) field
        username_field = None
        password_field = None
        other_fields: dict[str, str] = {}

        for pt in signup_form_points:
            param_lower = pt.parameter.lower()
            if "pwd" in param_lower or "pass" in param_lower:
                password_field = pt.parameter
            elif username_field is None and _USERNAME_HINTS_RE.search(pt.parameter):
                username_field = pt.parameter
            # Collect defaults for all sibling fields
            if pt.default_form_values:
                other_fields.update(pt.default_form_values)

        if not username_field:
            print(f"  [XSS Chain] Skipping {signup_url} — no username field found")
            continue

        print(f"  [XSS Chain] Testing: register at {signup_url} "
              f"(username={username_field!r}, password={password_field!r})")

        # --- Step 1: Register with XSS payload as username ---
        reg_session = requests.Session()
        reg_session.headers["User-Agent"] = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
        )
        if cookies:
            reg_session.cookies.update(cookies)

        # Fetch signup page to get CSRF tokens / hidden fields
        form_page_url = signup_form_points[0].form_page_url or base_url
        try:
            page_resp = reg_session.get(form_page_url, timeout=10)
            from bs4 import BeautifulSoup as _BS
            soup = _BS(page_resp.text, "lxml")
            # Extract hidden fields from the signup form
            hidden_fields: dict[str, str] = {}
            for form in soup.find_all("form"):
                action = (form.get("action") or "").strip()
                form_url = requests.compat.urljoin(form_page_url, action) if action else form_page_url
                if form_url.rstrip("/") == signup_url.rstrip("/") or _SIGNUP_HINTS.search(form_url):
                    for inp in form.find_all("input", type="hidden"):
                        n = inp.get("name", "").strip()
                        if n:
                            hidden_fields[n] = inp.get("value", "")
                    # Also grab submit button name/value
                    for inp in form.find_all("input", type="submit"):
                        n = inp.get("name", "").strip()
                        if n:
                            hidden_fields[n] = inp.get("value", "Submit")
                    break
        except Exception:
            hidden_fields = {}

        # Build registration POST data
        reg_data = dict(other_fields)
        reg_data.update(hidden_fields)
        reg_data[username_field] = payload
        if password_field:
            reg_data[password_field] = test_pwd

        try:
            reg_resp = reg_session.post(signup_url, data=reg_data, timeout=15,
                                        allow_redirects=True)
            logger.debug("[XSS chain] Register response: %d, length=%d",
                        reg_resp.status_code, len(reg_resp.text))
        except Exception as e:
            print(f"  [XSS Chain] Registration failed: {e}")
            continue

        # --- Step 2: Try to login as the XSS user ---
        login_session = None
        for login_form_points in login_groups:
            login_url = login_form_points[0].url
            login_username_field = None
            login_password_field = None
            login_other: dict[str, str] = {}

            for pt in login_form_points:
                param_lower = pt.parameter.lower()
                if "pwd" in param_lower or "pass" in param_lower:
                    login_password_field = pt.parameter
                elif login_username_field is None and _USERNAME_HINTS_RE.search(pt.parameter):
                    login_username_field = pt.parameter
                if pt.default_form_values:
                    login_other.update(pt.default_form_values)

            if not login_username_field:
                continue

            # Fresh session for login
            login_session = requests.Session()
            login_session.headers["User-Agent"] = reg_session.headers["User-Agent"]

            # Fetch login page for CSRF tokens
            login_page_url = login_form_points[0].form_page_url or base_url
            try:
                lp_resp = login_session.get(login_page_url, timeout=10)
                soup = _BS(lp_resp.text, "lxml")
                login_hidden: dict[str, str] = {}
                for form in soup.find_all("form"):
                    action = (form.get("action") or "").strip()
                    fu = requests.compat.urljoin(login_page_url, action) if action else login_page_url
                    if fu.rstrip("/") == login_url.rstrip("/") or _LOGIN_HINTS.search(fu):
                        for inp in form.find_all("input", type="hidden"):
                            n = inp.get("name", "").strip()
                            if n:
                                login_hidden[n] = inp.get("value", "")
                        for inp in form.find_all("input", type="submit"):
                            n = inp.get("name", "").strip()
                            if n:
                                login_hidden[n] = inp.get("value", "Submit")
                        break
            except Exception:
                login_hidden = {}

            login_data = dict(login_other)
            login_data.update(login_hidden)
            login_data[login_username_field] = payload
            if login_password_field:
                login_data[login_password_field] = test_pwd

            try:
                login_resp = login_session.post(login_url, data=login_data,
                                                timeout=15, allow_redirects=True)
                logger.debug("[XSS chain] Login response: %d", login_resp.status_code)
            except Exception as e:
                print(f"  [XSS Chain] Login failed: {e}")
                login_session = None
                continue
            break  # use first login form

        # Also create a session using the scanner's existing auth cookies.
        # This catches stored XSS visible to OTHER users (e.g. admin sees
        # malicious usernames on admin.php, log pages, user lists).
        auth_session = None
        if cookies:
            auth_session = requests.Session()
            auth_session.headers["User-Agent"] = reg_session.headers["User-Agent"]
            auth_session.cookies.update(cookies)

        # Collect sessions to sweep with (XSS-user session + admin session)
        sweep_sessions = []
        if login_session:
            sweep_sessions.append(("xss_user", login_session))
        if auth_session:
            sweep_sessions.append(("auth_user", auth_session))
        if not sweep_sessions:
            # Fallback: use registration session
            sweep_sessions.append(("reg_session", reg_session))

        from urllib.parse import urljoin as _urljoin, urlsplit as _urlsplit
        for session_label, sweep_session in sweep_sessions:
            # --- Step 3: Sweep all crawled pages + auth pages for stored payload ---
            pages_to_check = list(crawled_pages)
            # Extract links from the post-login page (only if login succeeded)
            if login_session is not None:
                try:
                    soup = _BS(login_resp.text, "lxml")
                    base_parsed = _urlsplit(login_url)
                    for a in soup.find_all("a", href=True):
                        href = a["href"].strip()
                        if href and not href.startswith(("#", "javascript:", "mailto:")):
                            full = _urljoin(login_url, href)
                            p = _urlsplit(full)
                            if p.netloc == base_parsed.netloc:
                                clean = f"{p.scheme}://{p.netloc}{p.path}"
                                if clean not in pages_to_check:
                                    pages_to_check.append(clean)
                except Exception:
                    pass
            # Add common post-auth pages where username is often displayed
            _scope = base_url.rstrip("/") + "/"
            for suffix in ("index.php", "index.html", "home.php", "home",
                           "dashboard", "profile", "admin.php", "admin",
                           "auth1.php", "auth2.php", "account", "welcome",
                           "main", "user", "settings"):
                candidate = _urljoin(_scope, suffix)
                if candidate not in pages_to_check:
                    pages_to_check.append(candidate)

            for page_url in pages_to_check:
                if any(h in page_url.lower() for h in ("logout", "logoff", "signout")):
                    continue
                # Skip if already found on this page
                already = any(f.get("display_page") == page_url for f in findings)
                if already:
                    continue
                try:
                    resp = sweep_session.get(page_url, timeout=8, allow_redirects=True)
                    body_lower = resp.text.lower()
                    # Strict detection: payload must appear unescaped AND outside
                    # form-input value attributes (echoed form state is not XSS).
                    # Require the full <script> tag with our marker to be present.
                    marker = "xsschaintest"
                    has_unescaped_script = (
                        "<script" in body_lower
                        and marker in body_lower
                        and "&lt;script" not in body_lower[:body_lower.find(marker) + 200]
                    )
                    # Guard against reflected-in-input false positives: if the
                    # only occurrence is inside an input value="..." attribute,
                    # it's HTML-attribute-escaped and not executable.
                    import re as _re
                    in_input_only = bool(_re.search(
                        r'<input[^>]*value\s*=\s*["\'][^"\']*' + _re.escape(marker),
                        body_lower,
                    )) and body_lower.count(marker) == 1
                    if has_unescaped_script and not in_input_only:
                        print(_red(f"  [XSS Chain] STORED XSS FOUND on {page_url} ({session_label})"))
                        print(f"    Chain: register({username_field}={payload!r}) "
                              f"-> {session_label} -> {page_url}")
                        findings.append({
                            "injection_point": f"InjectionPoint(POST {signup_url} "
                                               f"param={username_field!r} type=form_field)",
                            "found_on_page": signup_url,
                            "injection_url": signup_url,
                            "url": signup_url,
                            "parameter": username_field,
                            "method": "POST",
                            "input_type": "form_field",
                            "vulnerable": True,
                            "vuln_subtype": "Stored XSS",
                            "xss_chain": True,
                            "chain_description": (
                                f"Register with {username_field}={payload!r}, "
                                f"then visit {page_url} (as {session_label})"
                            ),
                            "display_page": page_url,
                            "success_rate": 1.0,
                            "mean_reward": 80.0,
                            "mean_steps": 3.0,
                            "successful_payloads": [payload],
                            "impact_evidence": [
                                f"Stored XSS: payload persisted in DB via {username_field} "
                                f"field, rendered unescaped on {page_url}",
                                f"Visible to {session_label} session",
                            ],
                            "response_snippet": resp.text[:300],
                            "algo": "chain_test",
                        })
                except Exception:
                    continue

    if findings:
        print(f"\n  [XSS Chain] Found {len(findings)} stored XSS via chain tests")
    else:
        print(f"  [XSS Chain] No stored XSS found via chain tests")

    return findings


# ---------------------------------------------------------------------------
# Per-vuln-type scan helper
# ---------------------------------------------------------------------------

ALL_VULNS = ["xss", "sqli", "cmdi", "ssti"]


def scan_one_vuln(
    vuln_type: str,
    injection_points: list,
    args,
    cookies: dict,
    extra_headers: dict,
    api_key: str,
    use_llm_payloads: bool,
    crawled_pages: list[str] | None = None,
    auth_credentials: tuple[str, str, str] | None = None,
) -> tuple[list[dict], str]:
    """
    Run the trained agent for *vuln_type* against all *injection_points*.
    Returns (results_list, remediation_text).
    Skips gracefully if the model for this vuln type is not found.
    """
    model_path = args.model or DEFAULT_MODELS.get(vuln_type, "")
    if not model_path:
        print(f"\n  [SKIP] {vuln_type.upper()}: no default model path configured.")
        return [], ""
    if not Path(model_path + ".zip").exists() and not Path(model_path).exists():
        print(f"\n  [SKIP] {vuln_type.upper()}: model not found at '{model_path}' — train it first.")
        return [], ""

    print(f"\n{'=' * 68}")
    print(_cyan(f"  SCANNING: {vuln_type.upper()}  ({len(injection_points)} injection point(s), {args.episodes} episodes each)"))
    print(f"{'=' * 68}")

    _action_to_family_map = {
        "xss": XSS_ACTION_TO_FAMILY,
        "sqli": SQLI_ACTION_TO_FAMILY,
        "cmdi": CMDI_ACTION_TO_FAMILY,
        "ssti": SSTI_ACTION_TO_FAMILY,
    }
    action_to_family = _action_to_family_map.get(vuln_type, {})
    adaptive = not args.no_adaptive
    llm_gen = LLMPayloadGenerator(api_key=api_key) if adaptive else None

    prioritized = _prioritize_points(injection_points, args.episodes, vuln_type=vuln_type)
    total = len(prioritized)
    all_results = []
    # Response fingerprints: (url, status, len_bucket) -> first result for that handler
    _fingerprints: dict[tuple, bool] = {}  # fingerprint -> was_vulnerable

    for i, (point, point_episodes) in enumerate(prioritized, 1):
        point_score = _score_injection_point(point, vuln_type)
        print_progress(i, total, point.method, point.url, point.parameter,
                       episodes=point_episodes, score=point_score)

        # WAF pre-check + reflection probe (single request does both)
        probe_body, probe_status = "", 0
        _canary = "zQ7xR3pL9"
        probe_payload = f"<script>{_canary}</script>"
        try:
            waf_client = GenericHttpClient(
                point.url, cookies=cookies if cookies else None,
                headers=extra_headers if extra_headers else None,
            )
            probe_body, probe_status, _, _ = waf_client.send_payload(
                point, probe_payload
            )
            waf_blocked, waf_name = ResponseAnalyzer.detect_waf(
                probe_body, probe_status, waf_client.last_response_headers,
            )
            if waf_blocked:
                print(_yellow(f"  [WAF]  param={point.parameter!r} blocked by {waf_name} — skipping"))
                all_results.append({
                    "injection_point": str(point),
                    "url": point.url,
                    "parameter": point.parameter,
                    "method": point.method,
                    "waf_blocked": True,
                    "waf_name": waf_name,
                    "success_rate": 0.0,
                    "mean_reward": 0.0,
                    "mean_steps": 0.0,
                    "successful_payloads": [],
                    "payload_log": [],
                })
                continue

            # Reflection pre-check: if the canary doesn't appear in the
            # response, this param can't be exploited for XSS.  Skip to
            # save episodes and API calls.  Does not apply to stored XSS
            # (which has a separate verify_url) or non-XSS vuln types.
            if vuln_type == "xss" and not point.verify_url:
                if _canary not in (probe_body or ""):
                    print(f"  [Skip] param={point.parameter!r} — canary not reflected, skipping")
                    all_results.append({
                        "injection_point": str(point),
                        "url": point.url,
                        "parameter": point.parameter,
                        "method": point.method,
                        "skipped": True,
                        "skip_reason": "canary not reflected",
                        "success_rate": 0.0,
                        "mean_reward": 0.0,
                        "mean_steps": 0.0,
                        "successful_payloads": [],
                        "payload_log": [],
                    })
                    continue
        except Exception:
            pass  # WAF/reflection check failure is non-fatal, proceed with scan

        # Response fingerprint dedup: if another param on the same URL
        # produced an identical response shape and was safe, reduce episodes
        if probe_body:
            len_bucket = len(probe_body) // 500  # bucket by ~500-char increments
            fp = (point.url.rstrip("/"), probe_status, len_bucket)
            if fp in _fingerprints and not _fingerprints[fp]:
                # Same handler, previously safe — reduce episodes
                point_episodes = max(3, point_episodes // 4)
            # Will be updated after testing with actual result

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env = make_dynamic_env(
                    injection_point=point,
                    vuln_type=vuln_type,
                    max_steps=args.max_steps,
                    api_key=api_key,
                    wrap_monitor=True,
                    use_llm_payloads=use_llm_payloads,
                    cookies=cookies if cookies else None,
                    crawled_pages=crawled_pages,
                    auth_credentials=auth_credentials,
                )
                model, algo = load_model(model_path, env)

            adaptive_session = None
            if adaptive and point_score >= 60:
                adaptive_session = AdaptiveScanSession(
                    model=model,
                    vuln_type=vuln_type,
                    action_to_family=action_to_family,
                    llm_gen=llm_gen,
                    q_threshold=args.q_threshold,
                    family_fail_thresh=args.family_fail_thresh,
                )

            result = run_episodes(model, env, point_episodes, point, adaptive_session)
            result["algo"] = algo
            all_results.append(result)
            env.close()
            print_inline_result(result, vuln_type)

            # Update fingerprint cache for dedup
            if probe_body:
                len_bucket = len(probe_body) // 500
                fp = (point.url.rstrip("/"), probe_status, len_bucket)
                _fingerprints[fp] = result.get("success_rate", 0) > 0

        except Exception as e:
            logger.debug("Agent test error on %s param=%s: %s", point.url, point.parameter, e)
            print(_red(f"  [ERROR] Could not test parameter {point.parameter!r} — skipping."))
            all_results.append({
                "injection_point": str(point),
                "url": point.url,
                "parameter": point.parameter,
                "method": point.method,
                "error": "Could not test this injection point — check target connectivity.",
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "mean_steps": 0.0,
                "successful_payloads": [],
            })

    print_results_table(all_results, vuln_type)

    vulnerable = [r for r in all_results if r.get("success_rate", 0) > 0]
    remediation_text = ""
    if vulnerable:
        print("\n" + "=" * 68)
        print(f"  HOW TO FIX ({vuln_type.upper()})")
        print("=" * 68)
        if api_key:
            print(_cyan("  Fetching page context and generating targeted advice...\n"))
        remediation_text = generate_remediation_advice(
            vuln_type, vulnerable, api_key, cookies=cookies, extra_headers=extra_headers,
        )
        print(remediation_text)
        print()

    return all_results, remediation_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _require_authorization(target_url: str) -> bool:
    """Prompt before scanning non-local targets. Returns True if authorized."""
    from urllib.parse import urlsplit
    host = (urlsplit(target_url).hostname or "").lower()
    local_hosts = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
    is_local = host in local_hosts or host.endswith(".local")
    if is_local:
        return True
    if os.environ.get("PENTEST_AUTHORIZED") == "1":
        return True
    print(f"\n  WARNING: Target {host!r} is not a local test instance.")
    print("  Scanning systems you don't own or lack written permission for may")
    print("  violate the Computer Misuse Act / CFAA.")
    try:
        reply = input("  Type 'I HAVE PERMISSION' to continue: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return reply == "I HAVE PERMISSION"


def _preview_scan_scope(injection_points: list, target_url: str) -> bool:
    """Show every host/path that will be probed. Return True to proceed."""
    from collections import defaultdict
    from urllib.parse import urlsplit

    target_host = (urlsplit(target_url).hostname or "?").lower()
    by_host: dict[str, set[str]] = defaultdict(set)
    for pt in injection_points:
        parts = urlsplit(pt.url)
        by_host[(parts.hostname or "?").lower()].add(parts.path or "/")

    print(f"\n  -- Scan scope preview " + "-" * 42)
    print(f"  Target: {target_url}")
    print(f"  {len(injection_points)} injection point(s) across {len(by_host)} host(s):\n")
    for host in sorted(by_host):
        marker = "*" if host == target_host else "o"
        paths = sorted(by_host[host])
        print(f"    {marker} {host}  ({len(paths)} path(s))")
        for p in paths[:5]:
            print(f"        {p}")
        if len(paths) > 5:
            print(f"        ... +{len(paths) - 5} more")
    print("  " + "-" * 62)
    if os.environ.get("PENTEST_SKIP_PREVIEW") == "1":
        return True
    try:
        reply = input("  Proceed with scan? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return reply in ("", "y", "yes")


def main():
    parser = argparse.ArgumentParser(
        description="Scan any web application for vulnerabilities using trained RL agents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--url", required=True,
        help="Target URL to scan (e.g. http://target.com/search)"
    )
    parser.add_argument(
        "--vuln", default="all", choices=["all", "xss", "sqli", "cmdi", "ssti"],
        help="Vulnerability type to scan, or 'all' to test every type (default: all)"
    )

    # Optional — sensible defaults for everything below
    parser.add_argument(
        "--model", default=None,
        help="Path to trained model (only valid when --vuln is a single type, not 'all')"
    )
    parser.add_argument(
        "--episodes", type=int, default=20,
        help="Episodes per injection point (default: 20)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=30,
        help="Max steps per episode (default: 50)"
    )
    parser.add_argument(
        "--cookie", type=str, default=None,
        help="Session cookie for authenticated scans (e.g. 'PHPSESSID=abc; security=low')"
    )
    parser.add_argument(
        "--username", type=str, default=None,
        help="Username to log in with (auto-detects the login form)"
    )
    parser.add_argument(
        "--password", type=str, default=None,
        help="Password for --username"
    )
    parser.add_argument(
        "--login-url", type=str, default=None,
        help="Login page URL (defaults to --url if not specified)"
    )
    parser.add_argument(
        "--header", type=str, action="append", default=[],
        help="Extra HTTP header (e.g. 'Authorization: Bearer token'). Repeatable."
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key for LLM-assisted crawling (reads ANTHROPIC_API_KEY env if omitted)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="JSON report output path (auto-named in reports/ if omitted)"
    )
    parser.add_argument(
        "--deep", action="store_true",
        help="Follow internal links after the initial page — discovers more authenticated pages"
    )
    parser.add_argument(
        "--max-pages", type=int, default=50,
        help="Maximum pages to visit during --deep crawl (default: 50)"
    )
    parser.add_argument(
        "--basic-auth", type=str, default=None, metavar="USER:PASS",
        help="HTTP Basic Auth credentials (e.g. 'admin:secret'). Auto-detected if omitted."
    )
    parser.add_argument(
        "--bearer-token", type=str, default=None, metavar="TOKEN",
        help="Bearer token for API authentication (sets Authorization: Bearer <TOKEN> header)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, metavar="SECONDS",
        help="Seconds to wait between HTTP requests (default: 0). Use 0.5-2.0 to avoid rate limiting."
    )
    parser.add_argument(
        "--no-headless", action="store_true",
        help="Disable Playwright headless browser fallback even if installed"
    )
    parser.add_argument(
        "--no-llm-crawl", action="store_true",
        help="Force static HTML parser only (LLM crawl is skipped automatically if no API key)"
    )
    parser.add_argument(
        "--no-llm-payloads", action="store_true",
        help="Disable LLM-generated payloads; use static payload lists only"
    )
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="Disable adaptive scanning (on by default; falls back to RL-only when no API key)"
    )
    parser.add_argument(
        "--no-heuristic", action="store_true",
        help="Skip heuristic security checks (CSRF, access control, header injection, etc.)"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast reconnaissance mode: fewer episodes (5), lower max-steps (20), aggressive early give-up"
    )
    parser.add_argument(
        "--source-path", type=str, default=None,
        help="Path to source code for static analysis (requires ANTHROPIC_API_KEY)"
    )
    parser.add_argument(
        "--source-lang", type=str, default=None,
        help="Filter source files by language (e.g. php, python, javascript)"
    )
    parser.add_argument(
        "--q-threshold", type=float, default=Q_SPREAD_THRESHOLD,
        help=f"Q-spread threshold for LLM family override (default: {Q_SPREAD_THRESHOLD})"
    )
    parser.add_argument(
        "--family-fail-thresh", type=int, default=FAMILY_FAILURE_THRESHOLD,
        help=f"Consecutive failures before switching family (default: {FAMILY_FAILURE_THRESHOLD})"
    )

    args = parser.parse_args()

    if not _require_authorization(args.url):
        print("  Scan aborted — authorization not confirmed.")
        sys.exit(1)

    # --fast mode: override defaults for speed
    if args.fast:
        if args.episodes == 20:  # only override if user didn't set explicitly
            args.episodes = 5
        if args.max_steps == 30:
            args.max_steps = 20

    setup_logging(level="WARNING")
    logging.getLogger("utils.llm_success_detector").setLevel(logging.ERROR)
    # Let crawler log INFO so we can see pages visited and points discovered
    logging.getLogger("utils.web_crawler").setLevel(logging.INFO)

    # Determine which vuln types to run
    vuln_types = ALL_VULNS if args.vuln == "all" else [args.vuln]
    multi = len(vuln_types) > 1

    # --model is only meaningful for a single vuln type
    if multi and args.model:
        print("Warning: --model is ignored when --vuln all is used (each type uses its default model).")
        args.model = None

    # Resolve API key
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    # For single-vuln mode, validate the model up-front so we fail fast
    if not multi:
        model_path = args.model or DEFAULT_MODELS.get(args.vuln, "")
        if not model_path:
            print(f"Error: no default model for --vuln={args.vuln!r}. Pass --model explicitly.")
            sys.exit(1)
        if not Path(model_path + ".zip").exists() and not Path(model_path).exists():
            print(f"Error: model not found at '{model_path}'")
            print(f"  Make sure you have trained the universal model:")
            if args.vuln == "cmdi":
                print(f"  python -m agents.train --vuln cmdi --targets dvwa webgoat")
            elif args.vuln == "ssti":
                print(f"  python -m agents.train --vuln ssti --targets juiceshop dvwa")
            else:
                print(f"  python -m agents.train --vuln {args.vuln} --targets dvwa juiceshop")
            sys.exit(1)
    else:
        model_path = ""

    # Parse cookies
    cookies = parse_cookies(args.cookie or "")

    # Parse extra headers
    extra_headers = {}
    for h in args.header:
        if ":" in h:
            k, v = h.split(":", 1)
            extra_headers[k.strip()] = v.strip()

    # Bearer token → Authorization header
    if args.bearer_token:
        extra_headers["Authorization"] = f"Bearer {args.bearer_token}"

    # ------------------------------------------------------------------
    # Optional: authenticate with credentials before scanning
    # ------------------------------------------------------------------
    _auth_session = None  # Will hold the live requests.Session after form login
    if args.basic_auth:
        # Explicit --basic-auth user:pass
        parts = args.basic_auth.split(":", 1)
        if len(parts) != 2:
            print("Error: --basic-auth must be in 'user:pass' format.")
            sys.exit(1)
        ba_user, ba_pass = parts
        print(f"  [Auth] HTTP Basic Auth as '{ba_user}' ...", flush=True)
        auth_cookies, success, msg = authenticate_basic(args.url, ba_user, ba_pass)
        print(f"  [Auth] {msg}", flush=True)
        if not success:
            print(_red("  [Auth] Basic Auth failed — aborting scan."))
            sys.exit(1)
        cookies = {**auth_cookies, **cookies}
        extra_headers["Authorization"] = (
            "Basic " + __import__("base64").b64encode(
                f"{ba_user}:{ba_pass}".encode()
            ).decode()
        )
    elif args.username and args.password:
        login_url = args.login_url or args.url
        # Auto-detect auth type
        auth_type = detect_auth_type(login_url)
        if auth_type == "basic":
            print(f"  [Auth] Detected HTTP Basic Auth at {login_url} ...", flush=True)
            auth_cookies, success, msg = authenticate_basic(login_url, args.username, args.password)
            _auth_session = None
        else:
            print(f"  [Auth] Logging in at {login_url} as '{args.username}' ...", flush=True)
            auth_cookies, success, msg, _auth_session = authenticate(login_url, args.username, args.password)
        print(f"  [Auth] {msg}", flush=True)
        if not success:
            print(_red("  [Auth] Login failed — aborting scan."))
            sys.exit(1)
        cookies = {**auth_cookies, **cookies}

    use_llm_payloads = not args.no_llm_payloads
    print_banner(
        args.url,
        vuln_types if multi else args.vuln,
        model_path,
        api_key,
        use_llm_payloads,
        not args.no_adaptive,
        username=args.username or "",
        heuristic=not args.no_heuristic,
    )

    # ------------------------------------------------------------------
    # Step 1: Crawl the target (once, shared across all vuln types)
    # ------------------------------------------------------------------
    use_llm = bool(api_key) and not args.no_llm_crawl
    if not api_key:
        print("  Note: ANTHROPIC_API_KEY not set -- using static HTML parser.")
        print("        Set the key to enable LLM-assisted injection point discovery.\n")

    # Headless availability notice
    from utils.headless_crawler import HeadlessCrawler as _HC
    if _HC.available and not args.no_headless:
        print("  Note: Playwright detected — headless browser fallback enabled.\n")
    elif not _HC.available and not args.no_headless:
        print("  Note: Playwright not installed — headless fallback disabled.")
        print("        Install with: pip install playwright && playwright install chromium\n")

    if args.no_headless:
        # Monkey-patch availability off so the crawler skips headless
        import utils.headless_crawler as _hcmod
        _hcmod.HeadlessCrawler.available = False

    # Reuse the authenticated session (if any) so the crawler inherits the
    # exact server-side session state — just transferring cookies can fail
    # when the server ties sessions to internal state (e.g. PHP session_start).
    crawler = LLMCrawler(
        api_key=api_key if use_llm else None,
        request_delay=args.delay,
        http_session=_auth_session,
    )

    if args.deep:
        print(f"  Deep-crawling {args.url} (max {args.max_pages} pages) ...")
        injection_points = crawler.deep_crawl(
            args.url,
            cookies=cookies,
            extra_headers=extra_headers,
            max_pages=args.max_pages,
        )
    else:
        print(f"  Crawling {args.url} ...")
        injection_points = crawler.crawl(args.url, cookies=cookies, extra_headers=extra_headers)

    # ------------------------------------------------------------------
    # Dual-mode crawl: also crawl *unauthenticated* to discover login
    # forms and other inputs only visible to anonymous users.
    # ------------------------------------------------------------------
    if _auth_session is not None or cookies:
        print(f"  Crawling {args.url} (unauthenticated) to find login forms ...")
        unauth_crawler = LLMCrawler(
            api_key=api_key if use_llm else None,
            request_delay=args.delay,
        )
        if args.deep:
            unauth_points = unauth_crawler.deep_crawl(
                args.url, cookies={}, extra_headers={},
                max_pages=args.max_pages,
            )
        else:
            unauth_points = unauth_crawler.crawl(args.url, cookies={}, extra_headers={})

        # Merge: add any injection points not already discovered
        existing_keys = {
            (p.method, p.url, p.parameter) for p in injection_points
        }
        added = 0
        for pt in unauth_points:
            key = (pt.method, pt.url, pt.parameter)
            if key not in existing_keys:
                injection_points.append(pt)
                existing_keys.add(key)
                added += 1
        if added:
            print(f"  +{added} additional injection point(s) from unauthenticated crawl")

    if not injection_points:
        print("\n  No injection points discovered.")
        print("  Tips:")
        print("    - Point --url at the specific page with the form (e.g. /post?postId=1)")
        print("    - If the page requires login, pass --username/--password or --cookie")
        print("    - Use --deep to follow internal links and discover more pages")
        print("    - Set ANTHROPIC_API_KEY for LLM-assisted discovery of hidden inputs")
        sys.exit(1)

    print(f"  Found {len(injection_points)} injection point(s):\n")
    for idx, pt in enumerate(injection_points, 1):
        print(f"    {idx:>3}. {pt.method:<5} {pt.url}  param={pt.parameter!r}  ({pt.input_type})")
    print()

    if not _preview_scan_scope(injection_points, args.url):
        print("  Scan cancelled.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Step 2: Run the agent for each vulnerability type
    # ------------------------------------------------------------------
    results_by_vuln: dict[str, list[dict]] = {}
    remediations_by_vuln: dict[str, str] = {}

    # Collect crawled page URLs for stored XSS sweep
    _crawled_pages = getattr(crawler, "crawled_pages", [])

    # Build auth credentials tuple for session re-authentication
    _auth_creds = None
    if args.username and args.password:
        _auth_creds = (args.login_url or args.url, args.username, args.password)

    for vuln_type in vuln_types:
        results, remediation = scan_one_vuln(
            vuln_type, injection_points, args, cookies, extra_headers, api_key, use_llm_payloads,
            crawled_pages=_crawled_pages,
            auth_credentials=_auth_creds,
        )
        if results:
            results_by_vuln[vuln_type] = results
            remediations_by_vuln[vuln_type] = remediation

    # ------------------------------------------------------------------
    # Step 2a: Stored XSS chain test (register → login → sweep)
    # ------------------------------------------------------------------
    if "xss" in vuln_types and _crawled_pages:
        print("\n" + "=" * 68)
        print(_cyan("  STORED XSS CHAIN TEST (register -> login -> sweep)"))
        print("=" * 68)
        chain_findings = run_stored_xss_chain_test(
            injection_points=injection_points,
            crawled_pages=_crawled_pages,
            cookies=cookies,
            extra_headers=extra_headers,
            base_url=args.url,
        )
        if chain_findings:
            # Merge into XSS results
            xss_results = results_by_vuln.get("xss", [])
            xss_results.extend(chain_findings)
            results_by_vuln["xss"] = xss_results
            # Print chain findings summary
            print(f"\n  {'=' * 64}")
            for f in chain_findings:
                print(_red(f"  STORED XSS: {f['parameter']!r} on {f['url']}"))
                print(f"    Display page: {f.get('display_page', 'N/A')}")
                print(f"    Chain: {f.get('chain_description', '')}")
                for ev in f.get("impact_evidence", []):
                    print(f"    > {ev}")
                print()

    # Upgrade nav hints to LLM-generated for vulnerable endpoints only (saves API calls)
    if api_key and not args.no_llm_crawl:
        vuln_points = []
        for results in results_by_vuln.values():
            for r in results:
                if r.get("success_rate", 0) > 0:
                    # Find the matching injection point
                    for pt in injection_points:
                        if pt.url == r.get("url") and pt.parameter == r.get("parameter"):
                            vuln_points.append(pt)
                            break
        if vuln_points:
            crawler.upgrade_nav_hints(vuln_points, args.url)

    # ------------------------------------------------------------------
    # Step 2b: Heuristic security checks (CSRF, access control, etc.)
    # ------------------------------------------------------------------
    heuristic_findings = []
    if not args.no_heuristic:
        print("\n" + "=" * 68)
        print(_cyan("  RUNNING HEURISTIC SECURITY CHECKS"))
        print("=" * 68 + "\n")

        # Build a session for heuristic checks
        heuristic_session = _auth_session or requests.Session()
        if cookies:
            for k, v in cookies.items():
                heuristic_session.cookies.set(k, v)

        # Fetch HTML for crawled pages (needed by check_passwords_in_get)
        crawled_html: dict[str, str] = {}
        for page_url in _crawled_pages[:30]:  # cap at 30 to avoid slowdown
            try:
                resp = heuristic_session.get(page_url, timeout=8, allow_redirects=True)
                crawled_html[page_url] = resp.text
            except Exception:
                pass

        heuristic_findings = run_all_heuristic_checks(
            session=heuristic_session,
            injection_points=injection_points,
            crawled_urls=_crawled_pages,
            crawled_html=crawled_html,
            base_url=args.url,
            cookies=cookies,
            login_url=args.login_url or (args.url if args.username else ""),
            username=args.username or "",
            password=args.password or "",
        )

        print_heuristic_results(heuristic_findings)

    # ------------------------------------------------------------------
    # Step 2c: Static source code analysis (optional)
    # ------------------------------------------------------------------
    static_findings = []
    if args.source_path and api_key:
        print("\n" + "=" * 68)
        print(_cyan("  RUNNING STATIC CODE ANALYSIS"))
        print("=" * 68 + "\n")

        try:
            from agents.code_scan import collect_files, quick_prescreen, analyse_file, SEVERITY_ORDER, LANGUAGE_MAP
            import anthropic as _anthropic

            code_client = _anthropic.Anthropic(api_key=api_key)
            source_files = collect_files(args.source_path, lang_filter=args.source_lang)
            print(f"  Found {len(source_files)} source file(s) to analyse.\n")

            for i, fpath in enumerate(source_files, 1):
                try:
                    content = fpath.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                lang = LANGUAGE_MAP.get(fpath.suffix.lower(), "Unknown")
                if not quick_prescreen(content, lang):
                    continue
                print(f"  [{i}/{len(source_files)}] {_cyan(str(fpath.name))} ({lang}) ...", end=" ", flush=True)
                findings = analyse_file(fpath, content, lang, code_client, "claude-haiku-4-5-20251001")
                # Filter to medium+ severity
                findings = [f for f in findings if SEVERITY_ORDER.get(f.get("severity", "info").lower(), 4) <= 2]
                if findings:
                    print(_red(f"{len(findings)} issue(s)"))
                else:
                    print(_green("clean"))
                static_findings.extend(findings)

            if static_findings:
                print(f"\n  Static analysis: {_red(f'{len(static_findings)} issue(s) found')}")
            else:
                print(f"\n  {_green('Static analysis: no issues found.')}")
        except ImportError as e:
            print(_yellow(f"  Static analysis skipped: {e}"))
        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="static code analysis")
    elif args.source_path and not api_key:
        print(_yellow("\n  Static analysis requires ANTHROPIC_API_KEY -- skipping."))

    if not results_by_vuln and not heuristic_findings and not static_findings:
        print(_red("\n  No vuln types could be tested and no issues found."))
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: Combined summary + report
    # ------------------------------------------------------------------
    if multi:
        all_flat = [r for rs in results_by_vuln.values() for r in rs]
        total_vuln = sum(1 for r in all_flat if r.get("success_rate", 0) > 0)
        print("\n" + "=" * 68)
        print(_cyan("  FULL SCAN SUMMARY"))
        print("=" * 68)
        for vt, results in results_by_vuln.items():
            n_vuln = sum(1 for r in results if r.get("success_rate", 0) > 0)
            label = _red(f"{n_vuln} VULNERABLE") if n_vuln else _green("clean")
            print(f"  {vt.upper():<6}  {len(results)} point(s) tested -- {label}")
        if heuristic_findings:
            h_high = sum(1 for f in heuristic_findings if f.get("severity") == "High")
            h_med = sum(1 for f in heuristic_findings if f.get("severity") == "Medium")
            h_low = len(heuristic_findings) - h_high - h_med
            parts = []
            if h_high: parts.append(_red(f"{h_high} High"))
            if h_med: parts.append(_yellow(f"{h_med} Medium"))
            if h_low: parts.append(f"{h_low} Low")
            print(f"  HEUR.  {len(heuristic_findings)} issue(s) found -- {', '.join(parts)}")
        print(f"\n  Total RL-confirmed vulnerabilities: {_red(str(total_vuln)) if total_vuln else _green('0')}")
        if heuristic_findings:
            print(f"  Total heuristic issues           : {_red(str(len(heuristic_findings)))}")
        if static_findings:
            print(f"  Total static analysis issues     : {_red(str(len(static_findings)))}")
        print("=" * 68 + "\n")

        output_path = args.output or (
            f"reports/scan_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_report_multi(
            results_by_vuln, remediations_by_vuln, args.url, output_path,
            username=args.username or "",
            heuristic_findings=heuristic_findings,
            static_findings=static_findings,
        )
        any_vuln = total_vuln > 0 or len(heuristic_findings) > 0
    else:
        vt = vuln_types[0]
        results = results_by_vuln.get(vt, [])
        remediation = remediations_by_vuln.get(vt, "")
        output_path = args.output or (
            f"reports/scan_{vt}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        save_report(
            results, args.url, vt, output_path,
            username=args.username or "", remediation=remediation,
            heuristic_findings=heuristic_findings,
            static_findings=static_findings,
        )
        any_vuln = any(r.get("success_rate", 0) > 0 for r in results) or len(heuristic_findings) > 0 or len(static_findings) > 0

    sys.exit(1 if any_vuln else 0)


if __name__ == "__main__":
    main()
