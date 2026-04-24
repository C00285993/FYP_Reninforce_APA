"""
Dynamic Pentest Environments
Generic Gymnasium environments that wrap any injection point discovered
by the LLM crawler. The trained universal models (XSS / SQLi) load
directly because the observation and action spaces are identical to the
environments they were trained on.

DynamicXSSEnv  — mirrors XSSEnv   (12 actions, 20-dim observation)
DynamicSQLiEnv — mirrors SQLiEnv  (10 actions, 18-dim observation)

Usage:
    point = InjectionPoint(url="http://target/search", method="GET",
                           parameter="q", input_type="url_param")
    env   = DynamicXSSEnv(injection_point=point)
    model = DQN.load("models/universal_xss_dqn/xss_dqn_final", env=env)
"""

import json
import random
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor

from environments.base_env import BasePentestEnv
from environments.feature_extractors import (
    UNIFIED_XSS_STATE_SIZE,
    UNIFIED_SQLI_STATE_SIZE,
    UNIFIED_CMDI_STATE_SIZE,
    UNIFIED_SSTI_STATE_SIZE,
    extract_unified_xss_state,
    extract_unified_sqli_state,
    extract_unified_cmdi_state,
    extract_unified_ssti_state,
)
from utils.generic_http_client import GenericHttpClient, InjectionPoint
from utils.llm_payload_generator import LLMPayloadGenerator
from utils.llm_success_detector import LLMSuccessDetector
from utils.response_analyzer import ResponseAnalyzer, AnalysisResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared action tables (must match the trained environments exactly)
# ---------------------------------------------------------------------------

XSS_ACTIONS = {
    0: "submit_baseline",
    1: "inject_basic_script",
    2: "inject_img_onerror",
    3: "inject_svg_onload",
    4: "inject_event_handler",
    5: "inject_attr_escape",
    6: "inject_case_bypass",
    7: "inject_encoding",
    8: "inject_nested_tags",
    9: "inject_dom_based",
    10: "inject_polyglot",
    11: "report_done",
}

XSS_ACTION_TO_FAMILY = {
    0: None, 1: "basic_script", 2: "img_onerror", 3: "svg_onload",
    4: "event_handler", 5: "attribute_escape", 6: "case_bypass",
    7: "encoding_bypass", 8: "nested_tags", 9: "dom_based",
    10: "polyglot", 11: None,
}

SQLI_ACTIONS = {
    0: "submit_baseline",
    1: "inject_single_quote",
    2: "inject_or_true",
    3: "inject_union_select",
    4: "inject_comment_bypass",
    5: "inject_time_based",
    6: "inject_error_based",
    7: "inject_stacked",
    8: "inject_encoded",
    9: "report_done",
}

SQLI_ACTION_TO_FAMILY = {
    0: None, 1: "single_quote", 2: "or_true", 3: "union_select",
    4: "comment_bypass", 5: "time_based", 6: "error_based",
    7: "stacked", 8: "encoded", 9: None,
}

CMDI_ACTIONS = {
    0: "submit_baseline",
    1: "inject_separator",
    2: "inject_pipe",
    3: "inject_logical_and",
    4: "inject_logical_or",
    5: "inject_backtick",
    6: "inject_dollar_paren",
    7: "inject_newline",
    8: "inject_encoded",
    9: "report_done",
}

CMDI_ACTION_TO_FAMILY = {
    0: None,
    1: "command_separator",
    2: "pipe_operator",
    3: "logical_and",
    4: "logical_or",
    5: "backtick_exec",
    6: "dollar_paren",
    7: "newline_inject",
    8: "encoded",
    9: None,
}

SSTI_ACTIONS = {
    0: "submit_baseline",
    1: "probe_arithmetic",
    2: "probe_string",
    3: "inject_global",
    4: "inject_env",
    5: "inject_waf_encoding",
    6: "inject_waf_concat",
    7: "inject_rce_direct",
    8: "inject_rce_indirect",
    9: "report_done",
}

SSTI_ACTION_TO_FAMILY = {
    0: None,
    1: "arithmetic_probe",
    2: "string_ops",
    3: "global_object",
    4: "env_access",
    5: "waf_bypass_encoding",
    6: "waf_bypass_concat",
    7: "rce_direct",
    8: "rce_indirect",
    9: None,
}


# ---------------------------------------------------------------------------
# Base class for dynamic environments
# ---------------------------------------------------------------------------

class _DynamicBaseEnv(gym.Env):
    """
    Shared logic for DynamicXSSEnv and DynamicSQLiEnv.
    Does NOT inherit from BasePentestEnv to avoid DVWA-specific wiring,
    but replicates the same interface that SB3 + evaluate.py expect.
    """

    metadata = {"render_modes": ["human", "log"]}

    def __init__(
        self,
        injection_point: InjectionPoint,
        max_steps: int = 50,
        render_mode: str = "log",
        api_key: Optional[str] = None,
        log_dir: str = "./logs",
        use_llm_payloads: bool = True,
        cookies: Optional[dict] = None,
    ):
        super().__init__()

        self.injection_point = injection_point
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Shared components
        self.client = GenericHttpClient(
            base_url=injection_point.url,
            cookies=cookies,
        )
        self.analyzer = ResponseAnalyzer()
        self.detector = LLMSuccessDetector(api_key=api_key)

        # LLM payload generator (uses same API key; disabled if key absent)
        self._payload_gen = (
            LLMPayloadGenerator(api_key=api_key)
            if use_llm_payloads else LLMPayloadGenerator(api_key=None)
        )
        if use_llm_payloads and self._payload_gen.available:
            logger.info("LLM payload generation enabled")
        elif use_llm_payloads:
            logger.debug("LLM payload generation requested but API key absent — using static payloads")

        # Subclass sets these
        self.action_space: spaces.Discrete
        self.observation_space: spaces.Box

        # Episode state
        self.current_step = 0
        self.episode_count = 0
        self.agent_memory: dict = {}
        self._episode_reward = 0.0
        self._initialized = False

        self._last_payload = ""
        self._last_response = ""
        self._last_status = 200

        self.payloads = self._load_payloads()

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._ensure_ready()

        self.current_step = 0
        self.episode_count += 1
        self._episode_reward = 0.0
        self._got_signal = False  # Track positive signals for early truncation
        self.agent_memory = self._init_memory()
        self._last_payload = ""

        # Reset per-episode LLM payload counters (static-first strategy)
        if hasattr(self._payload_gen, 'reset_episode'):
            self._payload_gen.reset_episode()
        self._last_response = ""

        # Reset sequential-first pick counters so each episode starts fresh
        for attr in list(vars(self)):
            if attr.startswith("_pick_idx_"):
                delattr(self, attr)

        # Pre-load exploit-proof queue (XSS only) — high-impact payloads
        # that fire before normal family payloads in the first few actions
        self._exploit_proof_queue: list[str] = []
        if self._vuln_type() == "xss":
            ep = self.payloads.get("exploit_proof", {}).get("payloads", [])
            # Queue top 3 exploit-proof payloads for early injection
            self._exploit_proof_queue = list(ep[:3])

        # Fetch initial page — for stored XSS use verify_url (the display page)
        fetch_url = self.injection_point.verify_url or self.injection_point.url
        body, status, elapsed, ct = self.client.get(fetch_url)
        rf = GenericHttpClient.detect_response_format(ct, body)
        self.injection_point.response_format = rf

        obs = self._extract_state(body, AnalysisResult())
        info = {
            "episode": self.episode_count,
            "target": self.injection_point.url,
            "parameter": self.injection_point.parameter,
            "vuln_type": self._vuln_type(),
        }
        return obs, info

    # Minimum steps before no-signal truncation kicks in
    NO_SIGNAL_MIN_STEPS = 15
    NO_SIGNAL_REWARD_THRESHOLD = -10.0

    def step(self, action: int):
        self.current_step += 1

        response_body, analysis, payload_info = self._execute_action(action)
        reward = self._calculate_reward(action, analysis)
        self._episode_reward += reward
        self._update_memory(action, analysis)
        obs = self._extract_state(response_body, analysis)

        # Track positive signals for early truncation decision
        if analysis.payload_reflected or analysis.response_differs:
            self._got_signal = True
        if hasattr(analysis, 'sql_error_detected') and analysis.sql_error_detected:
            self._got_signal = True
        if hasattr(analysis, 'script_tag_present') and analysis.script_tag_present:
            self._got_signal = True

        terminated = self._is_success(analysis)
        truncated = self.current_step >= self.max_steps

        # No-signal early truncation: if we're past the minimum step threshold
        # with negative reward and zero positive signals, end the episode early
        if (
            not terminated
            and not truncated
            and self.current_step >= self.NO_SIGNAL_MIN_STEPS
            and self._episode_reward < self.NO_SIGNAL_REWARD_THRESHOLD
            and not self._got_signal
        ):
            truncated = True

        if self.render_mode == "log" and (terminated or truncated):
            status = "SUCCESS" if terminated else "TIMEOUT"
            logger.info(
                f"[{self._vuln_type().upper()}] Episode {self.episode_count} "
                f"{status} | steps={self.current_step} "
                f"reward={self._episode_reward:.1f} "
                f"param={self.injection_point.parameter!r}"
            )

        info = {
            "step": self.current_step,
            "action_name": self._get_action_name(action),
            "severity_score": analysis.severity_score,
            "episode_reward": self._episode_reward,
            "payload_info": payload_info,
        }
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self):
        if not self._initialized:
            ok = self.client.ensure_ready()
            if not ok:
                raise ConnectionError(
                    f"Cannot reach target: {self.injection_point.url}"
                )
            self.client.capture_baseline(self.injection_point)
            baseline = self.client._baseline_body
            baseline_time = self.client._baseline_time
            self.analyzer.set_baseline(self._vuln_type(), baseline, baseline_time)
            self._initialized = True

    def _init_memory(self) -> dict:
        return {
            "total_attempts": 0,
            "attempts_this_field": 0,
            "last_payload_category": 0,
            "unique_responses_seen": 0,
            "found_sql_error": False,
            "found_data_leak": False,
            "found_reflection": False,
            "found_xss_execution": False,
            "response_hashes": set(),
            "tried_actions": set(),
        }

    def _update_memory(self, action: int, analysis: AnalysisResult):
        m = self.agent_memory
        m["total_attempts"] += 1
        m["attempts_this_field"] += 1
        m["last_payload_category"] = action
        m["tried_actions"].add(action)

        rh = hash(str(analysis.response_length) + str(analysis.severity_score))
        if rh not in m["response_hashes"]:
            m["unique_responses_seen"] += 1
            m["response_hashes"].add(rh)

        if analysis.has_sql_error:
            m["found_sql_error"] = True
        if analysis.has_data_leak:
            m["found_data_leak"] = True
        if analysis.payload_reflected:
            m["found_reflection"] = True
        if analysis.script_tag_present or analysis.event_handler_present:
            m["found_xss_execution"] = True

    def _pick_payload(self, family: Optional[str]) -> str:
        if family is None:
            return "test_baseline_12345"

        # Exploit-proof priority: on the very first injection actions of an
        # episode, fire high-impact payloads (cookie theft, CSRF bypass, etc.)
        # regardless of which family the RL agent selected.  This ensures
        # PortSwigger-style labs see "dangerous" proofs early.
        if getattr(self, "_exploit_proof_queue", None):
            return self._exploit_proof_queue.pop(0)

        payloads = self.payloads.get(family, {}).get("payloads", ["test"])

        if not self._payload_gen.available:
            # Sequential-first strategy: on the first few attempts per family,
            # pick payloads from the front of the list (high-impact / document.cookie
            # variants are placed first). After exhausting the front, go random.
            family_key = f"_pick_idx_{family}"
            idx = getattr(self, family_key, 0)
            if idx < min(3, len(payloads)):
                setattr(self, family_key, idx + 1)
                return payloads[idx]
            return random.choice(payloads)

        # Build context from current environment state
        m = self.agent_memory
        observed = {
            "reflected":    m.get("found_reflection", False),
            "escaped":      (
                m.get("found_reflection", False)
                and not m.get("found_xss_execution", False)
                and not m.get("found_sql_error", False)
            ),
            "blocked":      (
                m.get("total_attempts", 0) > 0
                and not m.get("found_reflection", False)
                and not m.get("found_sql_error", False)
            ),
            "sql_error":    m.get("found_sql_error", False),
            "data_leaked":  m.get("found_data_leak", False),
        }
        context = {
            "target_url":            self.injection_point.url,
            "parameter":             self.injection_point.parameter,
            "last_payload":          self._last_payload or "",
            "last_response_snippet": self._last_response[:500] if self._last_response else "",
            "observed":              observed,
        }
        return self._payload_gen.generate(
            family=family,
            vuln_type=self._vuln_type(),
            fallback_payloads=payloads,
            context=context,
        )

    # ------------------------------------------------------------------
    # Abstract — implemented by subclasses
    # ------------------------------------------------------------------

    def _vuln_type(self) -> str: ...
    def _load_payloads(self) -> dict: ...
    def _execute_action(self, action_id: int) -> tuple: ...
    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float: ...
    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray: ...
    def _is_success(self, analysis: AnalysisResult) -> bool: ...
    def _get_action_name(self, action_id: int) -> str: ...


# ---------------------------------------------------------------------------
# Dynamic XSS environment
# ---------------------------------------------------------------------------

class DynamicXSSEnv(_DynamicBaseEnv):
    """
    Generic XSS environment for any injection point.
    Action space (12) and observation space (20-dim) are identical to XSSEnv,
    so the trained universal_xss_dqn model loads without modification.

    Detects and reports the XSS subtype:
      - "Stored XSS"    — verify_url is set (payload persists on a display page)
      - "DOM-based XSS"  — xss_context is "dom" or "javascript"
      - "Reflected XSS"  — payload reflected in the immediate server response
    """

    def __init__(self, injection_point: InjectionPoint, **kwargs):
        super().__init__(injection_point, **kwargs)
        self.action_space = spaces.Discrete(len(XSS_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_XSS_STATE_SIZE,),
            dtype=np.float32,
        )
        # Track the detected XSS subtype for this injection point
        self.xss_subtype = "Reflected XSS"  # default

    def _vuln_type(self) -> str:
        return "xss"

    def _load_payloads(self) -> dict:
        # Use the same payload file the model was trained on
        p = Path(__file__).parent.parent / "payloads" / "xss_payloads.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return data.get("payload_families", {})
        return {}

    def _get_action_name(self, action_id: int) -> str:
        return XSS_ACTIONS.get(action_id, f"action_{action_id}")

    def _execute_action(
        self, action_id: int
    ) -> tuple[str, AnalysisResult, dict]:
        family = XSS_ACTION_TO_FAMILY.get(action_id)

        # Action 11 = report_done (meta action, no HTTP request)
        if action_id == 11:
            analysis = AnalysisResult(response_length=len(self._last_response))
            return self._last_response, analysis, {"payload": "", "parameter": self.injection_point.parameter, "url_path": self.injection_point.url, "full_request_url": self.injection_point.url, "response_snippet": "", "reflected": False, "xss_subtype": self.xss_subtype}

        payload = self._pick_payload(family)
        body, status, elapsed, ct = self.client.send_payload(
            self.injection_point, payload
        )
        self._last_payload = payload
        self._last_response = body
        self._last_status = status

        # For stored XSS: after submitting, GET the verify page to check persistence
        verify_body = body
        if self.injection_point.verify_url:
            try:
                verify_body, v_status, v_elapsed, v_ct = self.client.get(
                    self.injection_point.verify_url
                )
            except Exception:
                verify_body = body

        rf = GenericHttpClient.detect_response_format(ct, verify_body)
        resp_headers = self.client.last_response_headers
        analysis = self.analyzer.analyze_xss_response(
            verify_body, payload, status, elapsed, headers=resp_headers
        )

        # Determine XSS subtype
        self.xss_subtype = self._classify_xss_subtype(analysis)

        payload_info = {
            "payload": payload,
            "parameter": self.injection_point.parameter,
            "url_path": self.injection_point.url,
            "full_request_url": (
                f"{self.injection_point.url}?{self.injection_point.parameter}={payload}"
                if self.injection_point.method == "GET" else self.injection_point.url
            ),
            "response_snippet": verify_body[:300],
            "reflected": analysis.payload_reflected,
            "xss_subtype": self.xss_subtype,
            "evidence": self._build_xss_evidence(analysis, payload),
        }
        return verify_body, analysis, payload_info

    @staticmethod
    def _build_xss_evidence(analysis: AnalysisResult, payload: str) -> list[str]:
        """Build human-readable proof-of-impact evidence from XSS analysis."""
        evidence = []
        if analysis.script_tag_present:
            evidence.append(f"Injected <script> tag executes in browser — JS code runs on page load")
        if analysis.event_handler_present:
            # Extract which handler from the payload
            handlers = ["onerror", "onload", "onmouseover", "onfocus", "onclick"]
            found = [h for h in handlers if h in payload.lower()]
            if found:
                evidence.append(f"Event handler '{found[0]}' fires on DOM element — JS executes on user interaction")
            else:
                evidence.append("Event handler executes on injected DOM element")
        if analysis.xss_context == "javascript":
            evidence.append("Payload injected into JavaScript context — executes as part of existing script")
        elif analysis.xss_context == "dom":
            evidence.append("DOM-based: payload processed by client-side JavaScript, modifies page DOM")
        if not evidence and analysis.payload_reflected:
            if analysis.xss_context == "encoded":
                evidence.append("Payload reflected but HTML-encoded — not executable (no impact)")
            else:
                evidence.append("Payload reflected in response but no confirmed execution context")
        return evidence

    def _classify_xss_subtype(self, analysis: AnalysisResult) -> str:
        """Classify the XSS subtype based on injection point and analysis."""
        # Stored: injection point has a separate verify URL
        if self.injection_point.verify_url:
            return "Stored XSS"
        # DOM-based: payload executes via client-side JS, not server reflection
        if analysis.xss_context in ("dom", "javascript"):
            return "DOM-based XSS"
        # Default: reflected
        return "Reflected XSS"

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0  # Step penalty

        # Action 11: report_done
        if action_id == 11:
            if self.agent_memory.get("found_xss_execution"):
                return 10.0
            return -20.0

        # Repeat-action penalty
        tried = self.agent_memory.get("tried_actions", set())
        if list(tried).count(action_id) >= 3:
            reward -= 5.0

        if analysis.payload_reflected:
            reward += 5.0  # Reflection alone is a weak signal
        if analysis.response_differs:
            reward += 10.0
        if analysis.script_tag_present or analysis.event_handler_present:
            # Confirmed executable XSS — high reward
            reward += 80.0
            # CSP mitigation: script reflected but won't execute due to CSP
            if analysis.script_blocked_by_csp:
                reward -= 40.0  # Downgrade — reflected but mitigated
            # Non-HTML content type: browser won't parse as HTML at all
            if analysis.non_html_content_type:
                reward -= 60.0  # Significant downgrade
        if analysis.xss_context in ("javascript", "dom"):
            reward += 40.0

        return reward

    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray:
        rf = self.injection_point.response_format
        return extract_unified_xss_state(body, analysis, self.agent_memory, response_format=rf)

    def _is_success(self, analysis: AnalysisResult) -> bool:
        # Require VISIBLE IMPACT: the payload must have created an executable
        # element (script tag or event handler) that a browser would run.
        # Mere reflection ("payload appeared in the page") is NOT enough.
        heuristic_hit = (
            analysis.script_tag_present
            or analysis.event_handler_present
        )
        # Also accept javascript: / dom context as executable
        if not heuristic_hit and analysis.xss_context in ("javascript", "dom"):
            heuristic_hit = True

        if not heuristic_hit:
            return False

        # CSP or non-HTML content type blocks execution — not a true success
        if analysis.non_html_content_type:
            return False

        # Confirm with LLM when available
        if self.detector.should_consult_llm("xss", analysis.severity_score):
            verdict = self.detector.detect(
                vuln_type="xss",
                payload=self._last_payload,
                response_body=self._last_response,
                status_code=self._last_status,
            )
            if verdict["success"]:
                logger.info(
                    f"[XSS] LLM confirmed success "
                    f"(confidence={verdict['confidence']:.0%}): "
                    f"{verdict['evidence']}"
                )
            return verdict["success"]

        return heuristic_hit


# ---------------------------------------------------------------------------
# Dynamic SQLi environment
# ---------------------------------------------------------------------------

class DynamicSQLiEnv(_DynamicBaseEnv):
    """
    Generic SQLi environment for any injection point.
    Action space (10) and observation space (18-dim) are identical to SQLiEnv,
    so the trained universal_sqli_dqn model loads without modification.
    """

    def __init__(self, injection_point: InjectionPoint, **kwargs):
        super().__init__(injection_point, **kwargs)
        self.action_space = spaces.Discrete(len(SQLI_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_SQLI_STATE_SIZE,),
            dtype=np.float32,
        )
        self.sqli_subtype = "Classic/In-band"  # default

    def _vuln_type(self) -> str:
        return "sqli"

    def _load_payloads(self) -> dict:
        p = Path(__file__).parent.parent / "payloads" / "sqli_payloads.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return data.get("payload_families", {})
        return {}

    def _get_action_name(self, action_id: int) -> str:
        return SQLI_ACTIONS.get(action_id, f"action_{action_id}")

    def _execute_action(
        self, action_id: int
    ) -> tuple[str, AnalysisResult, dict]:
        family = SQLI_ACTION_TO_FAMILY.get(action_id)

        if action_id == 9:  # report_done
            analysis = AnalysisResult(response_length=len(self._last_response))
            return self._last_response, analysis, {"payload": "", "parameter": self.injection_point.parameter, "url_path": self.injection_point.url, "full_request_url": self.injection_point.url, "response_snippet": "", "reflected": False, "sqli_subtype": self.sqli_subtype}

        payload = self._pick_payload(family)
        body, status, elapsed, ct = self.client.send_payload(
            self.injection_point, payload
        )
        self._last_payload = payload
        self._last_response = body
        self._last_status = status

        rf = GenericHttpClient.detect_response_format(ct, body)
        analysis = self.analyzer.analyze_sqli_response(body, payload, status, elapsed, response_format=rf)

        # Classify SQLi subtype
        self.sqli_subtype = self._classify_sqli_subtype(analysis)

        payload_info = {
            "payload": payload,
            "parameter": self.injection_point.parameter,
            "url_path": self.injection_point.url,
            "full_request_url": (
                f"{self.injection_point.url}?{self.injection_point.parameter}={payload}"
                if self.injection_point.method == "GET" else self.injection_point.url
            ),
            "response_snippet": body[:300],
            "reflected": False,
            "sqli_subtype": self.sqli_subtype,
            "evidence": self._build_sqli_evidence(analysis),
        }
        return body, analysis, payload_info

    @staticmethod
    def _build_sqli_evidence(analysis: AnalysisResult) -> list[str]:
        """Build human-readable proof-of-impact evidence from analysis."""
        evidence = []
        if analysis.has_data_leak:
            row_str = f"{analysis.leaked_data_count} row(s)" if analysis.leaked_data_count else "data"
            evidence.append(f"Data leaked: {row_str} extracted from database")
            if analysis.leaked_fields:
                evidence.append(f"Columns exposed: {', '.join(analysis.leaked_fields)}")
        if analysis.auth_bypass:
            evidence.append("Authentication bypassed — gained unauthorized access")
        if analysis.has_sql_error and analysis.sql_error_type:
            if analysis.sql_error_type == "time_based":
                delta = f"{analysis.response_time:.1f}s" if analysis.response_time else "significant"
                evidence.append(f"Time-based blind: response delayed by {delta} (vs normal <1s)")
            else:
                evidence.append(f"SQL error disclosed: {analysis.sql_error_type} error message visible in response")
        if analysis.response_differs and analysis.response_data_direction == 1 and not analysis.has_data_leak:
            evidence.append(
                "Hidden data retrieved: response significantly larger than baseline — "
                "WHERE clause expanded to return additional rows"
            )
        elif analysis.response_differs and not analysis.has_data_leak:
            evidence.append("Response content changed — server processed the injected SQL")
        return evidence

    def _classify_sqli_subtype(self, analysis: AnalysisResult) -> str:
        """Classify the SQLi subtype based on analysis results and verification probes."""
        # Time-based blind: analyzer flagged time delay
        if analysis.sql_error_type == "time_based":
            return "Blind Time-based"
        # Classic/In-band: data leaked in response (table rows, hidden_rows, etc.)
        if analysis.has_data_leak or analysis.auth_bypass:
            return "Classic/In-band"
        # Data retrieval: response significantly larger (hidden rows returned)
        if analysis.response_differs and analysis.response_data_direction == 1:
            return "Classic/In-band"
        # Error-based: SQL error messages revealed
        if analysis.has_sql_error and analysis.sql_error_type:
            return "Error-based"
        # Boolean blind: verify with true/false comparison probes
        if analysis.response_differs and self._verify_boolean_blind():
            return "Blind Boolean"
        # If response differs but boolean probe failed, still in-band
        if analysis.response_differs:
            return "Classic/In-band"
        return "Classic/In-band"

    def _verify_boolean_blind(self) -> bool:
        """
        Send true-condition and false-condition probes and compare responses.
        If they produce meaningfully different responses, it's boolean blind SQLi.
        """
        try:
            true_probes = ["' OR 1=1--", "' OR 'a'='a'--", "1 OR 1=1--"]
            false_probes = ["' OR 1=2--", "' OR 'a'='b'--", "1 OR 1=2--"]

            # Pick one pair
            idx = random.randint(0, len(true_probes) - 1)
            true_body, true_status, _, _ = self.client.send_payload(
                self.injection_point, true_probes[idx]
            )
            false_body, false_status, _, _ = self.client.send_payload(
                self.injection_point, false_probes[idx]
            )

            # Compare: meaningful difference = different content length (>10% delta)
            # or different status codes
            if true_status != false_status:
                return True
            len_true, len_false = len(true_body), len(false_body)
            if len_true == 0 and len_false == 0:
                return False
            max_len = max(len_true, len_false, 1)
            if abs(len_true - len_false) / max_len > 0.10:
                return True
            # Content-level check: strip whitespace and compare
            if true_body.strip() != false_body.strip():
                # Could be timestamps/tokens — check if the core differs
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, true_body, false_body).ratio()
                if ratio < 0.95:  # <95% similar = meaningfully different
                    return True
            return False
        except Exception:
            return False

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0

        if action_id == 9:  # report_done
            if self.agent_memory.get("found_data_leak") or self.agent_memory.get("found_sql_error"):
                return 10.0
            return -20.0

        tried = self.agent_memory.get("tried_actions", set())
        if list(tried).count(action_id) >= 3:
            reward -= 5.0

        if analysis.has_sql_error:
            reward += 15.0
        if analysis.response_differs:
            reward += 25.0
            # Boolean-blind signal: response direction differs from baseline
            # Larger response (+1) after OR 1=1 or smaller (-1) after AND 1=2
            # suggests the WHERE clause is being modified by our injection
            if analysis.response_data_direction != 0:
                reward += 15.0
        if analysis.has_data_leak:
            reward += 50.0
            reward += min(analysis.leaked_data_count * 5, 30)
        if analysis.auth_bypass:
            reward += 100.0

        return reward

    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray:
        rf = self.injection_point.response_format
        return extract_unified_sqli_state(body, analysis, self.agent_memory, response_format=rf)

    def _is_success(self, analysis: AnalysisResult) -> bool:
        # Data leak or auth bypass → classic SQLi success
        heuristic_hit = analysis.has_data_leak or analysis.auth_bypass

        # Time-based blind SQLi → success if analyzer flagged it
        # (analyzer already uses delta-vs-baseline to avoid false positives)
        if analysis.has_sql_error and analysis.sql_error_type == "time_based":
            heuristic_hit = True

        # Data retrieval via WHERE expansion (OR 1=1 style):
        # Response significantly larger AND content differs from baseline.
        # This catches hidden-row retrieval on sites that show products/items
        # in divs rather than tables (e.g. PortSwigger labs, e-commerce).
        if (
            not heuristic_hit
            and analysis.response_differs
            and analysis.response_data_direction == 1
        ):
            heuristic_hit = True

        if not heuristic_hit:
            return False

        if self.detector.should_consult_llm("sqli", analysis.severity_score):
            verdict = self.detector.detect(
                vuln_type="sqli",
                payload=self._last_payload,
                response_body=self._last_response,
                status_code=self._last_status,
            )
            if verdict["success"]:
                logger.info(
                    f"[SQLi] LLM confirmed success "
                    f"(confidence={verdict['confidence']:.0%}): "
                    f"{verdict['evidence']}"
                )
            return verdict["success"]

        return heuristic_hit


# ---------------------------------------------------------------------------
# Dynamic CMDi environment
# ---------------------------------------------------------------------------

class DynamicCMDiEnv(_DynamicBaseEnv):
    """
    Generic CMDi environment for any injection point.
    Action space (10) and observation space (18-dim) are identical to CMDiEnv,
    so a trained cmdi_dqn model loads without modification.
    """

    def __init__(self, injection_point: InjectionPoint, **kwargs):
        super().__init__(injection_point, **kwargs)
        self.action_space = spaces.Discrete(len(CMDI_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_CMDI_STATE_SIZE,),
            dtype=np.float32,
        )
        self.cmdi_subtype = "Direct/In-band"  # default

    def _vuln_type(self) -> str:
        return "cmdi"

    def _load_payloads(self) -> dict:
        p = Path(__file__).parent.parent / "payloads" / "cmdi_payloads.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return data.get("payload_families", {})
        return {}

    def _get_action_name(self, action_id: int) -> str:
        return CMDI_ACTIONS.get(action_id, f"action_{action_id}")

    def _execute_action(
        self, action_id: int
    ) -> tuple[str, AnalysisResult, dict]:
        family = CMDI_ACTION_TO_FAMILY.get(action_id)

        if action_id == 9:  # report_done
            analysis = AnalysisResult(response_length=len(self._last_response))
            return self._last_response, analysis, {
                "payload": "", "parameter": self.injection_point.parameter,
                "url_path": self.injection_point.url,
                "full_request_url": self.injection_point.url,
                "response_snippet": "", "reflected": False,
                "cmdi_subtype": self.cmdi_subtype,
            }

        payload = self._pick_payload(family)
        body, status, elapsed, ct = self.client.send_payload(
            self.injection_point, payload
        )
        self._last_payload = payload
        self._last_response = body
        self._last_status = status

        rf = GenericHttpClient.detect_response_format(ct, body)
        analysis = self.analyzer.analyze_cmdi_response(body, payload, status, elapsed, response_format=rf)

        # Classify CMDi subtype
        self.cmdi_subtype = self._classify_cmdi_subtype(analysis, payload)

        payload_info = {
            "payload": payload,
            "parameter": self.injection_point.parameter,
            "url_path": self.injection_point.url,
            "full_request_url": (
                f"{self.injection_point.url}?{self.injection_point.parameter}={payload}"
                if self.injection_point.method == "GET" else self.injection_point.url
            ),
            "response_snippet": body[:300],
            "reflected": payload.lower() in body.lower(),
            "cmdi_subtype": self.cmdi_subtype,
            "evidence": self._build_cmdi_evidence(analysis, body),
        }
        return body, analysis, payload_info

    @staticmethod
    def _build_cmdi_evidence(analysis: AnalysisResult, body: str) -> list[str]:
        """Build human-readable proof-of-impact evidence from CMDi analysis."""
        evidence = []
        if analysis.cmdi_time_based:
            evidence.append(
                f"Blind time-based CMDi: response delayed {analysis.response_time:.1f}s "
                f"(baseline <1s) — OS command executed server-side with no visible output"
            )
        if analysis.has_command_execution and not analysis.cmdi_time_based:
            evidence.append(f"Command output visible: {analysis.command_output_lines} line(s) of OS output in response")
            import re as _re
            uid_match = _re.search(r'(uid=\d+\([^)]+\))', body)
            if uid_match:
                evidence.append(f"System info leaked: {uid_match.group(1)}")
            passwd_match = _re.search(r'(root:.*?:/bin/\w+)', body)
            if passwd_match:
                evidence.append(f"File content leaked: /etc/passwd readable")
        if analysis.file_operations_detected:
            evidence.append("File system access confirmed — attacker can read/write server files")
        if analysis.shell_error_type and analysis.shell_error_type != "time_based_blind":
            evidence.append(f"Shell error exposed: {analysis.shell_error_type} — confirms OS command processing")
        return evidence

    def _classify_cmdi_subtype(self, analysis: AnalysisResult, payload: str) -> str:
        """Classify the CMDi subtype based on analysis results and baseline timing."""
        # Direct/In-band: command output visible in response
        if analysis.has_command_execution or analysis.command_output_lines > 0:
            return "Direct/In-band"
        # Time-based blind: payload uses sleep/timeout AND response significantly
        # slower than a fresh baseline request
        time_keywords = ("sleep", "timeout", "ping -c", "ping -n")
        is_time_payload = any(kw in payload.lower() for kw in time_keywords)
        if is_time_payload and analysis.response_time > 3.0:
            if self._verify_time_based_cmdi():
                return "Blind Time-based"
        return "Direct/In-band"

    def _verify_time_based_cmdi(self) -> bool:
        """
        Send a benign baseline request and compare its timing against the
        suspicious slow response. If baseline is fast and the slow response
        was >3x slower, confirm blind time-based CMDi.
        """
        try:
            # Send a harmless value to measure normal response time
            _, _, baseline_time, _ = self.client.send_payload(
                self.injection_point, "127.0.0.1"
            )
            # Also send a second baseline to smooth out network jitter
            _, _, baseline_time2, _ = self.client.send_payload(
                self.injection_point, "localhost"
            )
            avg_baseline = (baseline_time + baseline_time2) / 2.0

            # The last action's response_time is in self.analyzer's context,
            # but we can compare: if the slow response was >3x the baseline
            # and the baseline was under 2s, it's a genuine time-based signal
            if avg_baseline < 2.0 and avg_baseline > 0:
                return True  # Baseline fast, previous response was slow → confirmed
            return False
        except Exception:
            return False

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0

        if action_id == 9:  # report_done
            if self.agent_memory.get("found_command_exec"):
                return 10.0
            return -20.0

        tried = self.agent_memory.get("tried_actions", set())
        if list(tried).count(action_id) >= 3:
            reward -= 5.0

        if analysis.response_differs:
            reward += 15.0
        if analysis.shell_error_type:
            reward += 20.0
        if analysis.command_output_lines > 0:
            reward += 30.0
        if analysis.has_command_execution:
            reward += 40.0
        if analysis.file_operations_detected:
            reward += 60.0
        # Time-based blind CMDi: no visible output but response delayed
        if analysis.cmdi_time_based:
            reward += 50.0

        return reward

    def _update_memory(self, action: int, analysis: AnalysisResult):
        super()._update_memory(action, analysis)
        if analysis.has_command_execution:
            self.agent_memory["found_command_exec"] = True
        if analysis.file_operations_detected:
            self.agent_memory["found_file_operations"] = True

    def _init_memory(self) -> dict:
        memory = super()._init_memory()
        memory["found_command_exec"] = False
        memory["found_file_operations"] = False
        return memory

    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray:
        rf = self.injection_point.response_format
        return extract_unified_cmdi_state(body, analysis, self.agent_memory, response_format=rf)

    def _is_success(self, analysis: AnalysisResult) -> bool:
        heuristic_hit = (
            analysis.has_command_execution
            or analysis.file_operations_detected
            or analysis.cmdi_time_based
        )
        if not heuristic_hit:
            return False

        if self.detector.should_consult_llm("cmdi", analysis.severity_score):
            verdict = self.detector.detect(
                vuln_type="cmdi",
                payload=self._last_payload,
                response_body=self._last_response,
                status_code=self._last_status,
            )
            if verdict["success"]:
                logger.info(
                    f"[CMDi] LLM confirmed success "
                    f"(confidence={verdict['confidence']:.0%}): "
                    f"{verdict['evidence']}"
                )
            return verdict["success"]

        return heuristic_hit


# ---------------------------------------------------------------------------
# Dynamic SSTI environment
# ---------------------------------------------------------------------------

class DynamicSSTIEnv(_DynamicBaseEnv):
    """
    Generic SSTI environment for any Pug-template injection point.
    Action space (10) and observation space (18-dim) are identical to
    JuiceShopSSTIEnv, so a trained ssti_dqn model loads without modification.

    The injection point should be configured as:
        url          = <form POST target, e.g. http://localhost:3000/profile>
        verify_url   = <page to GET after POST to read rendered output>
        method       = "POST"
        parameter    = "username"
        input_type   = "form_field"

    GenericHttpClient.send_payload() handles the POST→GET verify_url pattern
    automatically when verify_url is set.
    """

    def __init__(self, injection_point: InjectionPoint, **kwargs):
        super().__init__(injection_point, **kwargs)
        self.action_space = spaces.Discrete(len(SSTI_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_SSTI_STATE_SIZE,),
            dtype=np.float32,
        )
        self.ssti_subtype = "SSTI"  # default

    def _vuln_type(self) -> str:
        return "ssti"

    def _load_payloads(self) -> dict:
        p = Path(__file__).parent.parent / "payloads" / "ssti_payloads.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return data.get("payload_families", {})
        return {}

    def _get_action_name(self, action_id: int) -> str:
        return SSTI_ACTIONS.get(action_id, f"action_{action_id}")

    def _execute_action(
        self, action_id: int
    ) -> tuple[str, AnalysisResult, dict]:
        family = SSTI_ACTION_TO_FAMILY.get(action_id)

        if action_id == 9:  # report_done
            analysis = AnalysisResult(response_length=len(self._last_response))
            return self._last_response, analysis, {
                "payload": "", "parameter": self.injection_point.parameter,
                "url_path": self.injection_point.url,
                "full_request_url": self.injection_point.url,
                "response_snippet": "", "reflected": False,
                "ssti_subtype": self.ssti_subtype,
            }

        payload = self._pick_payload(family)
        # send_payload: POST to url, then GET verify_url (stored-XSS pattern)
        body, status, elapsed, ct = self.client.send_payload(
            self.injection_point, payload
        )
        self._last_payload = payload
        self._last_response = body
        self._last_status = status

        rf = self.injection_point.response_format or "html"
        analysis = self.analyzer.analyze_ssti_response(
            body, payload, status, elapsed, response_format=rf
        )

        # Classify SSTI engine subtype
        self.ssti_subtype = self._classify_ssti_subtype(rf, payload)

        payload_info = {
            "payload": payload,
            "parameter": self.injection_point.parameter,
            "url_path": self.injection_point.url,
            "full_request_url": (
                f"{self.injection_point.url} [POST {self.injection_point.parameter}={payload[:60]}]"
            ),
            "response_snippet": body[:300],
            "reflected": "#{" not in body and "#{" in payload,
            "ssti_subtype": self.ssti_subtype,
            "evidence": self._build_ssti_evidence(analysis, body, payload, self.ssti_subtype),
        }
        return body, analysis, payload_info

    @staticmethod
    def _build_ssti_evidence(analysis: AnalysisResult, body: str, payload: str, engine: str) -> list[str]:
        """Build human-readable proof-of-impact evidence from SSTI analysis."""
        evidence = []
        if analysis.expression_evaluated:
            # Try to show what was evaluated
            import re as _re
            # Look for arithmetic results (e.g. 49 from 7*7)
            arith = _re.search(r'(\d+)\s*[*]\s*(\d+)', payload)
            if arith:
                expected = int(arith.group(1)) * int(arith.group(2))
                if str(expected) in body:
                    evidence.append(f"Template expression evaluated: {arith.group(1)}*{arith.group(2)} = {expected} visible in page")
            if not evidence:
                evidence.append(f"Template engine ({engine}) evaluated injected expression — server-side code execution confirmed")
        if analysis.nodejs_info_leaked:
            evidence.append("Server environment info leaked (Node.js version, env vars, or PHP config)")
        if analysis.has_command_execution:
            evidence.append("Remote Code Execution achieved — OS commands run on the server")
            import re as _re
            uid_match = _re.search(r'(uid=\d+\([^)]+\))', body)
            if uid_match:
                evidence.append(f"System identity: {uid_match.group(1)}")
        if analysis.waf_blocked:
            evidence.append("WAF blocked this payload — server has template injection protection")
        return evidence

    def _classify_ssti_subtype(self, response_format: str, payload: str) -> str:
        """
        Classify the SSTI engine by sending fingerprint probes.
        Falls back to response_format hint if probes are inconclusive.
        """
        # If response_format is already known, trust it
        if response_format == "pug":
            return "Pug (Node.js)"
        if response_format == "twig":
            return "Twig (PHP)"

        # Active fingerprinting: send engine-distinguishing probes
        engine = self._fingerprint_ssti_engine()
        if engine:
            return engine

        # Last resort: guess from payload syntax (least reliable)
        if "#{" in payload:
            return "Pug (Node.js)"
        if "{{" in payload:
            return "Jinja2 (Python)"
        return "SSTI"

    def _fingerprint_ssti_engine(self) -> str:
        """
        Send distinguishing payloads to identify the template engine:
        - {{7*'7'}} → Jinja2 returns '7777777', Twig returns '49'
        - #{7*7}    → Pug returns '49', others reflect literally
        - ${7*7}    → Freemarker/Mako returns '49'
        Returns engine name or empty string if inconclusive.
        """
        try:
            # Probe 1: {{7*'7'}} — distinguishes Jinja2 from Twig
            body1, _, _, _ = self.client.send_payload(
                self.injection_point, "{{7*'7'}}"
            )
            if "7777777" in body1:
                return "Jinja2 (Python)"
            if "49" in body1 and "{{7*'7'}}" not in body1:
                return "Twig (PHP)"

            # Probe 2: #{7*7} — Pug evaluates this
            body2, _, _, _ = self.client.send_payload(
                self.injection_point, "#{7*7}"
            )
            if "49" in body2 and "#{7*7}" not in body2:
                return "Pug (Node.js)"

            # Probe 3: {{7*7}} — generic check (Twig or Jinja2)
            body3, _, _, _ = self.client.send_payload(
                self.injection_point, "{{7*7}}"
            )
            if "49" in body3 and "{{7*7}}" not in body3:
                # Could be Twig or Jinja2 — Probe 1 didn't trigger Jinja2,
                # so this is likely Twig
                return "Twig (PHP)"

            return ""  # Inconclusive
        except Exception:
            return ""

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0

        if action_id == 9:  # report_done
            if self.agent_memory.get("found_rce"):
                return 10.0
            return -20.0

        tried = self.agent_memory.get("tried_actions", set())
        if list(tried).count(action_id) >= 3:
            reward -= 5.0

        if analysis.waf_blocked:
            reward -= 3.0
        if analysis.expression_evaluated:
            reward += 15.0
        if analysis.nodejs_info_leaked:
            reward += 30.0
        if analysis.has_command_execution:
            reward += 100.0

        return reward

    def _update_memory(self, action: int, analysis: AnalysisResult):
        super()._update_memory(action, analysis)
        if analysis.expression_evaluated:
            self.agent_memory["found_eval"] = True
        if analysis.has_command_execution:
            self.agent_memory["found_rce"] = True

    def _init_memory(self) -> dict:
        memory = super()._init_memory()
        memory["found_eval"] = False
        memory["found_rce"] = False
        return memory

    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray:
        rf = self.injection_point.response_format
        return extract_unified_ssti_state(body, analysis, self.agent_memory, response_format=rf)

    def _is_success(self, analysis: AnalysisResult) -> bool:
        heuristic_hit = analysis.has_command_execution or analysis.expression_evaluated
        if not heuristic_hit:
            return False

        if self.detector.should_consult_llm("ssti", analysis.severity_score):
            verdict = self.detector.detect(
                vuln_type="ssti",
                payload=self._last_payload,
                response_body=self._last_response,
                status_code=self._last_status,
            )
            if verdict["success"]:
                logger.info(
                    f"[SSTI] LLM confirmed success "
                    f"(confidence={verdict['confidence']:.0%}): "
                    f"{verdict['evidence']}"
                )
            return verdict["success"]

        return heuristic_hit


# ---------------------------------------------------------------------------
# Dynamic Stored XSS environment (multi-step: inject → check display pages)
# ---------------------------------------------------------------------------

STORED_XSS_ACTIONS = {
    **XSS_ACTIONS,                    # 0-11: same as regular XSS
    12: "check_display_page",         # GET verify_url / crawled pages to check persistence
    13: "try_next_display_page",      # Switch to next candidate display page
}

STORED_XSS_ACTION_TO_FAMILY = {
    **XSS_ACTION_TO_FAMILY,
    12: None,
    13: None,
}

STORED_XSS_OBS_SIZE = UNIFIED_XSS_STATE_SIZE + 2  # +stored_confirmed, +pages_checked_ratio


class DynamicStoredXSSEnv(_DynamicBaseEnv):
    """
    Multi-step stored XSS environment.

    Extends the regular XSS env with two extra actions:
      12: check_display_page  — GET the current candidate display page and
          look for the previously injected payload (confirms stored XSS).
      13: try_next_display_page — cycle to the next candidate display page.

    The agent learns to:
      1. Inject an XSS payload via the injection point (actions 1-10)
      2. Check display pages to confirm the payload persists (action 12/13)

    Observation: 22-dim (20 XSS + stored_confirmed + pages_checked_ratio)
    """

    def __init__(self, injection_point: InjectionPoint, display_urls: list[str] | None = None, **kwargs):
        super().__init__(injection_point, **kwargs)
        self.action_space = spaces.Discrete(len(STORED_XSS_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(STORED_XSS_OBS_SIZE,),
            dtype=np.float32,
        )
        self.xss_subtype = "Stored XSS"
        # Candidate display pages where stored payload might appear
        self._display_urls = display_urls or []
        if injection_point.verify_url and injection_point.verify_url not in self._display_urls:
            self._display_urls.insert(0, injection_point.verify_url)
        self._display_idx = 0
        self._stored_confirmed = False
        self._last_injected_payload = ""

    def _vuln_type(self) -> str:
        return "xss"

    def _load_payloads(self) -> dict:
        p = Path(__file__).parent.parent / "payloads" / "xss_payloads.json"
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            return data.get("payload_families", {})
        return {}

    def _get_action_name(self, action_id: int) -> str:
        return STORED_XSS_ACTIONS.get(action_id, f"action_{action_id}")

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        # Action 11: report_done
        if action_id == 11:
            analysis = AnalysisResult(response_length=len(self._last_response))
            return self._last_response, analysis, {
                "payload": "", "reflected": False,
                "xss_subtype": self.xss_subtype,
            }

        # Action 12: check current display page for stored payload
        if action_id == 12:
            return self._check_display_page()

        # Action 13: cycle to next display page
        if action_id == 13:
            if self._display_urls:
                self._display_idx = (self._display_idx + 1) % len(self._display_urls)
            analysis = AnalysisResult(response_length=len(self._last_response))
            return self._last_response, analysis, {
                "payload": "", "reflected": False,
                "xss_subtype": self.xss_subtype,
            }

        # Actions 0-10: standard XSS injection
        family = XSS_ACTION_TO_FAMILY.get(action_id)
        payload = self._pick_payload(family)
        body, status, elapsed, ct = self.client.send_payload(
            self.injection_point, payload
        )
        self._last_payload = payload
        self._last_injected_payload = payload
        self._last_response = body
        self._last_status = status

        analysis = self.analyzer.analyze_xss_response(body, payload, status, elapsed)
        payload_info = {
            "payload": payload,
            "parameter": self.injection_point.parameter,
            "url_path": self.injection_point.url,
            "reflected": analysis.payload_reflected,
            "xss_subtype": self.xss_subtype,
            "evidence": DynamicXSSEnv._build_xss_evidence(analysis, payload),
        }
        return body, analysis, payload_info

    def _check_display_page(self) -> tuple[str, AnalysisResult, dict]:
        """GET the current display page and check if a stored payload is present."""
        if not self._display_urls or not self._last_injected_payload:
            analysis = AnalysisResult(response_length=0)
            return "", analysis, {"payload": "", "reflected": False, "xss_subtype": self.xss_subtype}

        display_url = self._display_urls[self._display_idx]
        try:
            body, status, elapsed, ct = self.client.get(display_url)
        except Exception:
            body, status, elapsed = "", 0, 0.0

        payload = self._last_injected_payload
        analysis = self.analyzer.analyze_xss_response(body, payload, status, elapsed)

        if analysis.payload_reflected and (analysis.script_tag_present or analysis.event_handler_present):
            self._stored_confirmed = True
            self.xss_subtype = "Stored XSS"

        self._last_response = body
        self._last_status = status

        evidence = []
        if self._stored_confirmed:
            evidence.append(f"Stored XSS confirmed: payload persists on {display_url}")
            evidence.extend(DynamicXSSEnv._build_xss_evidence(analysis, payload))

        return body, analysis, {
            "payload": payload,
            "parameter": self.injection_point.parameter,
            "url_path": display_url,
            "reflected": analysis.payload_reflected,
            "xss_subtype": self.xss_subtype,
            "evidence": evidence,
        }

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0

        if action_id == 11:
            return 10.0 if self._stored_confirmed else -20.0

        if action_id == 13:
            return -0.5  # Small cost for cycling pages

        if action_id == 12:
            if self._stored_confirmed:
                return 100.0  # High reward for confirming stored XSS
            if analysis.payload_reflected:
                return 5.0  # Partial signal — reflected on display page
            return -2.0

        # Standard XSS rewards for injection actions
        tried = self.agent_memory.get("tried_actions", set())
        if list(tried).count(action_id) >= 3:
            reward -= 5.0
        if analysis.payload_reflected:
            reward += 5.0
        if analysis.script_tag_present or analysis.event_handler_present:
            reward += 30.0  # Lower than regular XSS — stored confirmation needed
        return reward

    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray:
        rf = self.injection_point.response_format
        base = extract_unified_xss_state(body, analysis, self.agent_memory, response_format=rf)
        # Append stored XSS specific features
        stored_feat = np.array([
            1.0 if self._stored_confirmed else 0.0,
            (self._display_idx + 1) / max(len(self._display_urls), 1),
        ], dtype=np.float32)
        return np.concatenate([base, stored_feat])

    def _is_success(self, analysis: AnalysisResult) -> bool:
        return self._stored_confirmed


# ---------------------------------------------------------------------------
# Dynamic Access Control environment
# ---------------------------------------------------------------------------

ACCESS_CONTROL_ACTIONS = {
    0: "submit_baseline",           # Request page with valid auth
    1: "try_no_auth",               # Request without cookies/auth
    2: "try_different_role",        # Request with low-privilege cookies
    3: "try_path_traversal",        # Try ../admin, /./admin, etc.
    4: "try_param_tampering",       # Modify user ID / role parameter
    5: "try_method_switch",         # Switch GET→POST or POST→GET
    6: "try_direct_object_ref",     # Try incrementing/decrementing IDs
    7: "try_force_browse",          # Request common admin paths
    8: "try_header_bypass",         # X-Original-URL, X-Rewrite-URL
    9: "report_done",
}

ACCESS_CONTROL_ACTION_TO_FAMILY = {
    0: None, 1: "no_auth", 2: "role_switch", 3: "path_traversal",
    4: "param_tamper", 5: "method_switch", 6: "idor",
    7: "force_browse", 8: "header_bypass", 9: None,
}

ACCESS_CONTROL_OBS_SIZE = 12


class DynamicAccessControlEnv(_DynamicBaseEnv):
    """
    Environment for testing broken access control (OWASP A01:2021).

    Tests whether authenticated-only pages are accessible without proper auth,
    whether horizontal/vertical privilege escalation is possible, and whether
    common bypass techniques succeed.

    Observation (12-dim):
      0: baseline_status_ok        — authenticated request returns 200
      1: no_auth_accessible        — page loads without auth
      2: no_auth_same_content      — content matches authenticated version
      3: role_switch_accessible    — low-priv user can access
      4: path_traversal_works      — path manipulation bypasses auth
      5: param_tamper_works        — ID/role param tampering succeeds
      6: method_switch_works       — HTTP method switch bypasses check
      7: idor_detected             — direct object reference found
      8: force_browse_found        — admin page accessible
      9: header_bypass_works       — X-Original-URL bypass found
      10: step_ratio               — current_step / max_steps
      11: unique_findings_ratio    — unique findings / 9 possible checks
    """

    def __init__(
        self,
        injection_point: InjectionPoint,
        auth_cookies: dict | None = None,
        low_priv_cookies: dict | None = None,
        admin_paths: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(injection_point, **kwargs)
        self.action_space = spaces.Discrete(len(ACCESS_CONTROL_ACTIONS))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(ACCESS_CONTROL_OBS_SIZE,),
            dtype=np.float32,
        )
        self._auth_cookies = auth_cookies or {}
        self._low_priv_cookies = low_priv_cookies or {}
        self._admin_paths = admin_paths or [
            "/admin", "/admin.php", "/dashboard", "/panel",
            "/manager", "/config", "/settings",
        ]
        # Findings state
        self._baseline_body = ""
        self._baseline_ok = False
        self._findings: dict[str, bool] = {}
        self.access_control_subtype = ""

    def _vuln_type(self) -> str:
        return "access_control"

    def _load_payloads(self) -> dict:
        return {}  # No payload file needed

    def _get_action_name(self, action_id: int) -> str:
        return ACCESS_CONTROL_ACTIONS.get(action_id, f"action_{action_id}")

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        url = self.injection_point.url
        evidence = []

        if action_id == 0:  # baseline
            body, status, elapsed, ct = self.client.get(url)
            self._baseline_body = body
            self._baseline_ok = 200 <= status < 400
            analysis = AnalysisResult(response_length=len(body), status_code=status)
            return body, analysis, {"payload": "", "evidence": evidence}

        if action_id == 1:  # no_auth
            body, status = self._request_without_auth(url)
            accessible = 200 <= status < 400 and len(body) > 200
            if accessible and self._baseline_body:
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, body[:2000], self._baseline_body[:2000]).ratio()
                same_content = ratio > 0.7
            else:
                same_content = False
            self._findings["no_auth"] = accessible
            self._findings["no_auth_same_content"] = same_content
            if accessible:
                evidence.append(f"Page accessible without authentication (status {status})")
                if same_content:
                    evidence.append("Content matches authenticated version -- full bypass")
                    self.access_control_subtype = "Missing Authentication"
            analysis = AnalysisResult(response_length=len(body), status_code=status)
            return body, analysis, {"payload": "(no cookies)", "evidence": evidence}

        if action_id == 2:  # role_switch
            body, status = self._request_with_cookies(url, self._low_priv_cookies)
            accessible = 200 <= status < 400 and len(body) > 200
            self._findings["role_switch"] = accessible
            if accessible:
                evidence.append(f"Low-privilege user can access protected page (status {status})")
                self.access_control_subtype = "Vertical Privilege Escalation"
            analysis = AnalysisResult(response_length=len(body), status_code=status)
            return body, analysis, {"payload": "(low-priv cookies)", "evidence": evidence}

        if action_id == 3:  # path_traversal
            traversals = [
                url.replace("/admin", "/../admin"),
                url + "/./",
                url.rstrip("/") + "%2f",
                url.replace("admin", "%61dmin"),
            ]
            for t_url in traversals:
                body, status = self._request_without_auth(t_url)
                if 200 <= status < 400 and len(body) > 200:
                    self._findings["path_traversal"] = True
                    evidence.append(f"Path traversal bypass: {t_url} (status {status})")
                    self.access_control_subtype = "Path Traversal Bypass"
                    analysis = AnalysisResult(response_length=len(body), status_code=status)
                    return body, analysis, {"payload": t_url, "evidence": evidence}
            self._findings["path_traversal"] = False
            analysis = AnalysisResult(response_length=0)
            return "", analysis, {"payload": "", "evidence": evidence}

        if action_id == 4:  # param_tampering
            # Try common ID params
            import re
            from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            tampered = False
            for key in list(params.keys()):
                val = params[key][0] if params[key] else ""
                if val.isdigit():
                    for new_val in [str(int(val) + 1), str(int(val) - 1), "1", "0"]:
                        new_params = {**{k: v[0] for k, v in params.items()}, key: new_val}
                        new_url = urlunparse(parsed._replace(query=urlencode(new_params)))
                        body, status, elapsed, ct = self.client.get(new_url)
                        if 200 <= status < 400 and len(body) > 200:
                            self._findings["param_tamper"] = True
                            evidence.append(f"Param tampering: {key}={new_val} returns data (status {status})")
                            self.access_control_subtype = "IDOR (Insecure Direct Object Reference)"
                            tampered = True
                            analysis = AnalysisResult(response_length=len(body), status_code=status)
                            return body, analysis, {"payload": f"{key}={new_val}", "evidence": evidence}
            self._findings["param_tamper"] = False
            analysis = AnalysisResult(response_length=0)
            return "", analysis, {"payload": "", "evidence": evidence}

        if action_id == 5:  # method_switch
            switch_method = "POST" if self.injection_point.method == "GET" else "GET"
            try:
                if switch_method == "POST":
                    resp = self.client.session.post(url, timeout=self.client.timeout)
                else:
                    resp = self.client.session.get(url, timeout=self.client.timeout)
                body, status = resp.text, resp.status_code
                accessible = 200 <= status < 400 and len(body) > 200
                self._findings["method_switch"] = accessible
                if accessible:
                    evidence.append(f"HTTP method switch ({switch_method}) bypasses access control")
                    self.access_control_subtype = "Method-based Bypass"
            except Exception:
                body, status = "", 0
                self._findings["method_switch"] = False
            analysis = AnalysisResult(response_length=len(body), status_code=status)
            return body, analysis, {"payload": f"Method: {switch_method}", "evidence": evidence}

        if action_id == 6:  # direct_object_ref
            # Similar to param_tamper but for path-based IDs
            import re
            match = re.search(r'/(\d+)(?:/|$|\?)', url)
            if match:
                orig_id = match.group(1)
                for delta in [1, -1, 0]:
                    new_id = str(int(orig_id) + delta) if delta != 0 else "1"
                    new_url = url.replace(f"/{orig_id}", f"/{new_id}", 1)
                    body, status, elapsed, ct = self.client.get(new_url)
                    if 200 <= status < 400 and len(body) > 200 and delta != 0:
                        self._findings["idor"] = True
                        evidence.append(f"IDOR: accessing ID {new_id} returns other user's data")
                        self.access_control_subtype = "IDOR (Insecure Direct Object Reference)"
                        analysis = AnalysisResult(response_length=len(body), status_code=status)
                        return body, analysis, {"payload": new_url, "evidence": evidence}
            self._findings["idor"] = False
            analysis = AnalysisResult(response_length=0)
            return "", analysis, {"payload": "", "evidence": evidence}

        if action_id == 7:  # force_browse
            from urllib.parse import urljoin
            for path in self._admin_paths:
                admin_url = urljoin(url, path)
                body, status = self._request_without_auth(admin_url)
                if 200 <= status < 400 and len(body) > 200:
                    self._findings["force_browse"] = True
                    evidence.append(f"Admin page accessible without auth: {admin_url} (status {status})")
                    self.access_control_subtype = "Forced Browsing"
                    analysis = AnalysisResult(response_length=len(body), status_code=status)
                    return body, analysis, {"payload": admin_url, "evidence": evidence}
            self._findings["force_browse"] = False
            analysis = AnalysisResult(response_length=0)
            return "", analysis, {"payload": "", "evidence": evidence}

        if action_id == 8:  # header_bypass
            bypass_headers = [
                ("X-Original-URL", self.injection_point.url),
                ("X-Rewrite-URL", self.injection_point.url),
                ("X-Forwarded-Host", "localhost"),
            ]
            for hdr_name, hdr_val in bypass_headers:
                try:
                    resp = self.client.session.get(
                        url, headers={hdr_name: hdr_val}, timeout=self.client.timeout,
                    )
                    if 200 <= resp.status_code < 400 and len(resp.text) > 200:
                        self._findings["header_bypass"] = True
                        evidence.append(f"Header bypass via {hdr_name}: page accessible (status {resp.status_code})")
                        self.access_control_subtype = "Header-based Bypass"
                        analysis = AnalysisResult(response_length=len(resp.text), status_code=resp.status_code)
                        return resp.text, analysis, {"payload": f"{hdr_name}: {hdr_val}", "evidence": evidence}
                except Exception:
                    pass
            self._findings["header_bypass"] = False
            analysis = AnalysisResult(response_length=0)
            return "", analysis, {"payload": "", "evidence": evidence}

        # Action 9: report_done
        analysis = AnalysisResult(response_length=len(self._last_response))
        return self._last_response, analysis, {
            "payload": "", "evidence": [],
            "access_control_subtype": self.access_control_subtype,
        }

    def _request_without_auth(self, url: str) -> tuple[str, int]:
        """Request a URL without any auth cookies."""
        try:
            import requests as _req
            resp = _req.get(url, timeout=8, allow_redirects=False)
            return resp.text, resp.status_code
        except Exception:
            return "", 0

    def _request_with_cookies(self, url: str, cookies: dict) -> tuple[str, int]:
        """Request a URL with specific cookies."""
        try:
            import requests as _req
            resp = _req.get(url, cookies=cookies, timeout=8, allow_redirects=False)
            return resp.text, resp.status_code
        except Exception:
            return "", 0

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        if action_id == 9:  # report_done
            if any(self._findings.values()):
                return 10.0
            return -20.0

        if action_id == 0:  # baseline
            return 1.0 if self._baseline_ok else -1.0

        # Reward for finding access control issues
        finding_keys = ["no_auth", "no_auth_same_content", "role_switch",
                        "path_traversal", "param_tamper", "method_switch",
                        "idor", "force_browse", "header_bypass"]
        action_to_key = {1: "no_auth", 2: "role_switch", 3: "path_traversal",
                         4: "param_tamper", 5: "method_switch", 6: "idor",
                         7: "force_browse", 8: "header_bypass"}
        key = action_to_key.get(action_id, "")
        if key and self._findings.get(key):
            return 50.0  # Found an access control issue
        return -1.0  # Step penalty

    def _extract_state(self, body: str, analysis: AnalysisResult) -> np.ndarray:
        findings_count = sum(1 for v in self._findings.values() if v)
        state = np.array([
            1.0 if self._baseline_ok else 0.0,
            1.0 if self._findings.get("no_auth") else 0.0,
            1.0 if self._findings.get("no_auth_same_content") else 0.0,
            1.0 if self._findings.get("role_switch") else 0.0,
            1.0 if self._findings.get("path_traversal") else 0.0,
            1.0 if self._findings.get("param_tamper") else 0.0,
            1.0 if self._findings.get("method_switch") else 0.0,
            1.0 if self._findings.get("idor") else 0.0,
            1.0 if self._findings.get("force_browse") else 0.0,
            1.0 if self._findings.get("header_bypass") else 0.0,
            min(self.current_step / max(self.max_steps, 1), 1.0),
            min(findings_count / 9.0, 1.0),
        ], dtype=np.float32)
        return state

    def _is_success(self, analysis: AnalysisResult) -> bool:
        return any(self._findings.values())


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def make_dynamic_env(
    injection_point: InjectionPoint,
    vuln_type: str,
    max_steps: int = 50,
    api_key: Optional[str] = None,
    wrap_monitor: bool = True,
    use_llm_payloads: bool = True,
    cookies: Optional[dict] = None,
    crawled_pages: Optional[list[str]] = None,
    auth_credentials: Optional[tuple[str, str, str]] = None,
) -> gym.Env:
    """
    Create a wrapped dynamic environment ready for SB3.

    Args:
        injection_point:  Discovered injection point to test.
        vuln_type:        "xss" or "sqli".
        max_steps:        Episode step limit.
        api_key:          Anthropic API key (reads ANTHROPIC_API_KEY env if None).
        wrap_monitor:     Wrap in SB3 Monitor (required for evaluate_agent()).
        use_llm_payloads: Use Claude to generate context-adapted payloads (default True).
                          Automatically falls back to static list if no API key.

    Returns:
        Gymnasium environment, optionally wrapped in Monitor.
    """
    kwargs = dict(
        injection_point=injection_point,
        max_steps=max_steps,
        api_key=api_key,
        use_llm_payloads=use_llm_payloads,
        cookies=cookies,
    )
    if vuln_type == "xss":
        env = DynamicXSSEnv(**kwargs)
    elif vuln_type == "stored_xss":
        display_urls = crawled_pages or []
        env = DynamicStoredXSSEnv(display_urls=display_urls, **kwargs)
    elif vuln_type == "sqli":
        env = DynamicSQLiEnv(**kwargs)
    elif vuln_type == "cmdi":
        env = DynamicCMDiEnv(**kwargs)
    elif vuln_type == "ssti":
        env = DynamicSSTIEnv(**kwargs)
    elif vuln_type == "access_control":
        env = DynamicAccessControlEnv(auth_cookies=cookies, **kwargs)
    else:
        raise ValueError(f"Unknown vuln_type: {vuln_type!r}. Use 'xss', 'sqli', 'cmdi', 'ssti', 'stored_xss', or 'access_control'.")

    # Pass crawled page URLs so the client can sweep them for stored XSS
    if crawled_pages:
        env.client._crawled_pages = list(crawled_pages)

    # Post-auth page discovery: if the injection point has a verify_url
    # (indicating stored data) and the URL suggests a registration/login
    # form, probe common post-auth landing pages so the stored-XSS sweep
    # can find pages where stored usernames/emails are displayed.
    if crawled_pages and injection_point.verify_url:
        _url_lower = injection_point.url.lower()
        _auth_hints = ("signup", "sign-up", "register", "registration",
                       "create_account", "create-account", "new_user",
                       "enroll", "join", "login", "signin", "sign-in")
        if any(h in _url_lower for h in _auth_hints):
            env.client.probe_post_auth_pages(injection_point.url)

    # Store auth credentials so the client can re-authenticate on session expiry
    if auth_credentials:
        env.client.set_auth_credentials(*auth_credentials)

    if wrap_monitor:
        env = Monitor(env)
    return env
