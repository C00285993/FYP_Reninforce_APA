"""
WebGoat XSS Environment
Gymnasium environment for training an RL agent to exploit Reflected XSS
on OWASP WebGoat's "Cross-Site Scripting" Shopping Cart exercise.

WebGoat's XSS lesson reflects the ``field2`` (credit card owner) value
from the purchase form back into the receipt page without sanitisation.
Injecting an XSS payload into ``field2`` triggers ``lessonCompleted: true``
in the JSON response, which serves as the definitive success signal.

This environment shares identical action / observation spaces with XSSEnv
(DVWA) so the universal XSS model can be fine-tuned on WebGoat via
transfer learning with no architecture changes.

Target lesson: Cross-Site Scripting (Intro) — exercise 5a (Reflected XSS)
Endpoint: GET /WebGoat/CrossSiteScripting/attack5a
Params: QTY1=1, QTY2=1, QTY3=1, QTY4=1, field1=<payload>, field2="John Doe"
Response: JSON {lessonCompleted, feedback, output (HTML receipt)}

Note: WebGoat lesson 5a checks XSS_PATTERN against field1 (credit card number),
NOT field2 (cardholder name). The pattern requires <script>alert(...)</script>
or <script>console.log(...)</script>. Injecting into field2 triggers the
"wrong field" failure path (lessonCompleted=False).

Action space:  12 discrete actions (identical to XSSEnv)
Observation space: 20-dimensional float vector (UNIFIED_XSS_STATE_SIZE)
"""

import json
import random
import logging
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from gymnasium import spaces

from environments.base_env import BasePentestEnv
from environments.feature_extractors import UNIFIED_XSS_STATE_SIZE, extract_unified_xss_state
from utils.response_analyzer import AnalysisResult
from utils.webgoat_client import WebGoatClient

logger = logging.getLogger(__name__)


class WebGoatXSSEnv(BasePentestEnv):
    """
    RL Environment for XSS testing on OWASP WebGoat.

    Identical action semantics and reward structure to XSSEnv (DVWA) so
    that the universal XSS model can be transferred and fine-tuned on
    WebGoat's Java/Spring Reflected XSS exercise.

    The ``field2`` parameter in WebGoat's Shopping Cart lesson mirrors
    DVWA's ``name`` field — both accept a value that is reflected into
    the response page without sanitisation.

    Actions:
        0:  submit_baseline       - Submit normal text (no injection)
        1:  inject_basic_script   - Basic <script>alert(1)</script>
        2:  inject_img_onerror    - <img src=x onerror=alert(1)>
        3:  inject_svg_onload     - <svg onload=alert(1)>
        4:  inject_event_handler  - HTML5 event handler payloads
        5:  inject_attr_escape    - Break out of HTML attributes
        6:  inject_case_bypass    - Case variation bypass
        7:  inject_encoding       - Encoding-based evasion
        8:  inject_nested_tags    - Nested/broken tags
        9:  inject_dom_based      - DOM-based vectors
        10: inject_polyglot       - Multi-context polyglots
        11: report_done           - Agent declares finished

    Rewards: mirrors XSSEnv
        -1.0  per step
        +10   payload reflected in response
        +10   response differs from baseline
        +40   unescaped script tag or event handler detected
        +40   XSS in html/javascript context
        -20   report_done without finding anything
        -5    repeating same action 3+ times in a row
        +2    exploration bonus for first use of each action type
    """

    ACTIONS = {
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

    ACTION_TO_FAMILY = {
        0: None,
        1: "basic_script",
        2: "img_onerror",
        3: "svg_onload",
        4: "event_handler",
        5: "attribute_escape",
        6: "case_bypass",
        7: "encoding_bypass",
        8: "nested_tags",
        9: "dom_based",
        10: "polyglot",
        11: None,
    }

    # ------------------------------------------------------------------ #
    #  Client & identity
    # ------------------------------------------------------------------ #

    def _create_client(self):
        """Use WebGoatClient instead of DVWAClient."""
        return WebGoatClient(base_url=self.dvwa_url)

    def _vuln_type(self) -> str:
        return "xss"

    # ------------------------------------------------------------------ #
    #  Spaces
    # ------------------------------------------------------------------ #

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_XSS_STATE_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Payload loading
    # ------------------------------------------------------------------ #

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "xss_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("xss")
        return html

    def _capture_baselines(self):
        """Submit empty field2 to capture WebGoat baseline (no XSS)."""
        response_text, status, _ = self.client.submit_xss("")
        self.analyzer.set_baseline("webgoat_xss", response_text)
        self._baseline_html = response_text

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute an XSS action against WebGoat."""
        action_name = self.ACTIONS[action_id]
        url_path = self.client._XSS_PATH

        if action_id == 0:
            # Baseline: submit a harmless credit card number (no XSS pattern)
            payload = random.choice(["4128 3214 0002 1999", "1234 5678 9012 3456",
                                     "4111 1111 1111 1111", "5500 0000 0000 0004"])
            response_text, status, elapsed = self.client.submit_xss(payload)
            analysis = self.analyzer.analyze_webgoat_xss_response(
                response_text, payload, status, elapsed
            )
            payload_info = self._build_payload_info(payload, url_path, response_text, analysis)
            return response_text, analysis, payload_info

        elif action_id == 11:
            # Report done — no HTTP request
            analysis = AnalysisResult()
            analysis.script_tag_present = self.agent_memory.get("found_xss_execution", False)
            analysis.event_handler_present = self.agent_memory.get("found_xss_execution", False)
            payload_info = {
                "payload": "",
                "parameter": "field2",
                "url_path": url_path,
                "full_request_url": "",
                "response_snippet": "",
                "reflected": False,
            }
            return self._baseline_html, analysis, payload_info

        else:
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                payload = random.choice(self.payloads[family_name]["payloads"])
            else:
                payload = "<script>alert(1)</script>"

            response_text, status, elapsed = self.client.submit_xss(payload)
            analysis = self.analyzer.analyze_webgoat_xss_response(
                response_text, payload, status, elapsed
            )

            logger.debug(
                f"WebGoat XSS action={action_name} payload={payload!r} "
                f"reflected={analysis.payload_reflected} "
                f"script={analysis.script_tag_present} "
                f"event={analysis.event_handler_present} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(payload, url_path, response_text, analysis)
            return response_text, analysis, payload_info

    def _build_payload_info(
        self, payload: str, url_path: str, response_text: str, analysis: AnalysisResult
    ) -> dict:
        params = {"QTY1": "1", "QTY2": "1", "QTY3": "1", "QTY4": "1",
                  "field1": payload, "field2": "John Doe"}
        full_url = f"{self.dvwa_url}{url_path}?{urlencode(params)}"

        # Extract snippet from WebGoat JSON response
        snippet = ""
        try:
            data = json.loads(response_text)
            output_html = data.get("output", "") or ""
            feedback = data.get("feedback", "") or ""
            snippet = (output_html or feedback)[:500]
        except (ValueError, TypeError):
            snippet = response_text[:500]

        return {
            "payload": payload,
            "parameter": "field1",
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": analysis.payload_reflected,
        }

    # ------------------------------------------------------------------ #
    #  Reward — mirrors XSSEnv._calculate_reward()
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0  # Step penalty

        if analysis.payload_reflected and not self.agent_memory.get("found_reflection"):
            reward += 10.0
            logger.debug("  +10: Payload reflected in response")

        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 10.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +10: Response differs from baseline")

        if analysis.script_tag_present:
            if not self.agent_memory.get("found_xss_execution"):
                reward += 40.0
                logger.debug("  +40: Script tag executed (lessonCompleted)!")
            if analysis.xss_context in ("html", "javascript"):
                reward += 40.0
                logger.debug(f"  +40: XSS in {analysis.xss_context} context!")

        if analysis.event_handler_present:
            if not self.agent_memory.get("_rewarded_event"):
                reward += 40.0
                self.agent_memory["_rewarded_event"] = True
                logger.debug("  +40: Event handler triggered!")

        if action_id == 11:
            if self.agent_memory.get("found_xss_execution"):
                reward += 20.0
            else:
                reward -= 20.0

        recent_actions = [s.get("action_id") for s in self.episode_log[-3:]]
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            reward -= 5.0

        if action_id not in self.agent_memory.get("tried_actions", set()):
            reward += 2.0

        return reward

    # ------------------------------------------------------------------ #
    #  Agent memory
    # ------------------------------------------------------------------ #

    def _init_agent_memory(self) -> dict:
        memory = super()._init_agent_memory()
        memory["found_reflection"] = False
        memory["found_xss_execution"] = False
        return memory

    def _update_agent_memory(self, action_id: int, analysis: AnalysisResult):
        super()._update_agent_memory(action_id, analysis)
        if analysis.payload_reflected:
            self.agent_memory["found_reflection"] = True
        if analysis.script_tag_present or analysis.event_handler_present:
            self.agent_memory["found_xss_execution"] = True

    # ------------------------------------------------------------------ #
    #  State extraction & termination
    # ------------------------------------------------------------------ #

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        return extract_unified_xss_state(
            html, analysis, self.agent_memory, response_format="webgoat"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when XSS payload is detected by WebGoat."""
        return analysis.script_tag_present or analysis.event_handler_present

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
