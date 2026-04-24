"""
LLM Success Detector
Uses Claude to determine whether an HTTP response indicates a successful
SQL injection or XSS attack. Replaces the hardcoded per-target success
logic (e.g. DVWA's script-tag check, Juice Shop's scoreboard poll).

Design:
  - Called only when heuristics indicate a *possible* success (score >= threshold).
  - Caches results to avoid repeated API calls for identical payloads.
  - Returns a structured verdict with confidence and evidence.

Usage:
    detector = LLMSuccessDetector(api_key="sk-...")
    verdict  = detector.detect(
        vuln_type="xss",
        payload="<img src=x onerror=alert(1)>",
        response_body="...HTML...",
        status_code=200,
    )
    if verdict["success"]:
        print(verdict["evidence"])
"""

import os
import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Heuristic thresholds — LLM is only consulted above these
_XSS_HEURISTIC_THRESHOLD = 30    # AnalysisResult.severity_score
_SQLI_HEURISTIC_THRESHOLD = 25


class LLMSuccessDetector:
    """
    Determines whether an attack succeeded by asking Claude.
    Falls back to heuristic-only mode if the API is unavailable.
    """

    XSS_SYSTEM = """\
You are a web security expert reviewing an HTTP response to determine
whether a reflected or DOM-based XSS payload executed successfully.

Answer ONLY with a JSON object containing:
  "success"    : true or false
  "confidence" : 0.0 – 1.0
  "evidence"   : one sentence explaining your verdict

Look for: unescaped script/event-handler tags in HTML body, alert() calls,
onerror/onload attributes reflected without HTML entity encoding,
iframe with javascript: src, or any sign the payload reached the DOM unescaped.
Do NOT consider the payload merely being present in the page source as success —
it must appear in a context where a browser would execute it.
"""

    SQLI_SYSTEM = """\
You are a web security expert reviewing an HTTP response to determine
whether a SQL injection payload succeeded.

Answer ONLY with a JSON object containing:
  "success"    : true or false
  "confidence" : 0.0 – 1.0
  "evidence"   : one sentence explaining your verdict

Look for: SQL error messages, database records in the response body,
authentication bypass indicators (welcome message, admin panel access,
JWT/auth token in JSON), UNION SELECT results, or time-based delays.
A generic 200 response with no data does NOT indicate success.
"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-haiku-4-5-20251001",
    ):
        try:
            import anthropic as _anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            self._client = _anthropic.Anthropic(api_key=key)
            self._model = model
            self._available = True
        except Exception as e:
            logger.warning("LLM detector unavailable — %s", e)
            self._available = False

        # Simple in-memory cache: (vuln_type, payload_hash, body_hash) -> verdict
        self._cache: dict[tuple, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        vuln_type: str,
        payload: str,
        response_body: str,
        status_code: int = 200,
    ) -> dict:
        """
        Determine whether the attack succeeded.

        Args:
            vuln_type:     "xss" or "sqli"
            payload:       The payload that was sent.
            response_body: The raw HTTP response body.
            status_code:   HTTP status code of the response.

        Returns:
            dict with keys: success (bool), confidence (float), evidence (str),
                            method (str — "llm" or "heuristic")
        """
        cache_key = self._cache_key(vuln_type, payload, response_body)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if self._available:
            verdict = self._llm_detect(vuln_type, payload, response_body, status_code)
        else:
            verdict = self._heuristic_fallback(vuln_type, payload, response_body, status_code)

        self._cache[cache_key] = verdict
        return verdict

    def should_consult_llm(
        self, vuln_type: str, severity_score: int
    ) -> bool:
        """
        Gate: only call detect() when heuristics suggest possible success.
        Prevents an LLM call on every single step.
        """
        if vuln_type == "xss":
            return severity_score >= _XSS_HEURISTIC_THRESHOLD
        return severity_score >= _SQLI_HEURISTIC_THRESHOLD

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _llm_detect(
        self,
        vuln_type: str,
        payload: str,
        response_body: str,
        status_code: int,
    ) -> dict:
        system = self.XSS_SYSTEM if vuln_type == "xss" else self.SQLI_SYSTEM

        # Truncate body to keep token costs low
        truncated = response_body[:4000]
        user_msg = (
            f"HTTP Status: {status_code}\n"
            f"Payload sent: {payload}\n\n"
            f"Response body:\n```\n{truncated}\n```"
        )

        try:
            import json
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()

            # Strip markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw.strip())
            return {
                "success": bool(data.get("success", False)),
                "confidence": float(data.get("confidence", 0.5)),
                "evidence": data.get("evidence", ""),
                "method": "llm",
            }

        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="LLM detection", once_flag_obj=self)
            return self._heuristic_fallback(vuln_type, payload, response_body, status_code)

    # ------------------------------------------------------------------
    # Heuristic fallback (no API key needed)
    # ------------------------------------------------------------------

    def _heuristic_fallback(
        self,
        vuln_type: str,
        payload: str,
        response_body: str,
        status_code: int,
    ) -> dict:
        """Simple regex/keyword heuristics used when LLM is unavailable."""
        import re
        body_lower = response_body.lower()

        if vuln_type == "xss":
            payload_lower = payload.lower()
            # Check payload reflected unescaped
            reflected = payload_lower in body_lower
            # Check for event handlers in response
            event_handlers = ["onerror=", "onload=", "onfocus=", "onmouseover="]
            has_handler = any(h in body_lower for h in event_handlers) and reflected
            # Check for unescaped script tag
            has_script = "<script" in body_lower and reflected

            success = has_handler or has_script
            evidence = (
                "Payload reflected with unescaped event handler or script tag."
                if success else
                "No clear XSS execution indicator found in response."
            )
            return {
                "success": success,
                "confidence": 0.7 if success else 0.3,
                "evidence": evidence,
                "method": "heuristic",
            }

        else:  # sqli
            sql_errors = [
                "syntax error", "sql error", "mysql_", "mysqli_",
                "sqlite", "odbc", "ora-", "unclosed quotation",
            ]
            data_indicators = [
                "first_name", "surname", "password", "admin",
                '"data":[', '"token":', "welcome",
            ]
            has_error = any(e in body_lower for e in sql_errors)
            has_data = any(d in body_lower for d in data_indicators)
            success = has_error or has_data

            evidence = (
                "SQL error or data leak detected in response."
                if success else
                "No SQL injection indicator found in response."
            )
            return {
                "success": success,
                "confidence": 0.65 if success else 0.3,
                "evidence": evidence,
                "method": "heuristic",
            }

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(vuln_type: str, payload: str, body: str) -> tuple:
        body_hash = hashlib.md5(body[:2000].encode(), usedforsecurity=False).hexdigest()
        return (vuln_type, payload[:200], body_hash)
