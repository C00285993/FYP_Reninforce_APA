"""
XSS (Cross-Site Scripting) Environment
Gymnasium environment for training an RL agent to discover and exploit
XSS vulnerabilities on DVWA's Reflected XSS page.

Action space: 12 discrete actions (payload categories + meta actions)
Observation space: 20-dimensional float vector
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import random
import logging
from pathlib import Path
from urllib.parse import urlencode

from bs4 import BeautifulSoup

from environments.base_env import BasePentestEnv
from utils.response_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class XSSEnv(BasePentestEnv):
    """
    RL Environment for XSS testing on DVWA.

    The agent learns to select XSS payloads to test against DVWA's
    Reflected XSS page. The goal is to inject a script that would execute
    in a victim's browser (detected by finding unescaped script/event tags
    in the response).

    Actions:
        0:  submit_baseline       - Submit normal text input
        1:  inject_basic_script   - Basic <script>alert(1)</script>
        2:  inject_img_onerror    - <img src=x onerror=alert(1)>
        3:  inject_svg_onload     - <svg onload=alert(1)>
        4:  inject_event_handler  - Various event handler payloads
        5:  inject_attr_escape    - Break out of HTML attributes
        6:  inject_case_bypass    - Case variation bypass
        7:  inject_encoding       - Encoding-based evasion
        8:  inject_nested_tags    - Nested/broken tags
        9:  inject_dom_based      - DOM-based vectors
        10: inject_polyglot       - Multi-context polyglots
        11: report_done           - Agent declares finished

    Rewards:
        -1.0  per step (efficiency penalty)
        +10   payload reflected in response (but escaped)
        +20   response differs from baseline
        +40   unescaped script/event handler in response
        +80   confirmed XSS execution context
        -20   report_done without finding anything
        -5    repeating same action 3+ times
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

    def _vuln_type(self) -> str:
        return "xss"

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(self.extractor.XSS_STATE_SIZE,),
            dtype=np.float32,
        )

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "xss_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("xss_reflected")
        return html

    def _capture_baselines(self):
        """Submit normal input to capture baseline response."""
        html, status, _ = self.client.submit_xss_reflected("test")
        self.analyzer.set_baseline("xss_reflected", html)
        self._baseline_html = html

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute an XSS action against DVWA."""
        action_name = self.ACTIONS[action_id]
        url_path = self.client.PAGES["xss_reflected"]

        if action_id == 0:
            # Baseline: submit harmless text
            payload = random.choice(["hello", "test", "world", "user123"])
            html, status, elapsed = self.client.submit_xss_reflected(payload)
            analysis = self.analyzer.analyze_xss_response(
                html, payload, status, elapsed
            )
            payload_info = self._build_payload_info(
                payload, "name", url_path, html, analysis
            )
            return html, analysis, payload_info

        elif action_id == 11:
            # Report done
            analysis = AnalysisResult()
            analysis.script_tag_present = self.agent_memory.get(
                "found_xss_execution", False
            )
            payload_info = {
                "payload": "",
                "parameter": "",
                "url_path": url_path,
                "full_request_url": "",
                "response_snippet": "",
                "reflected": False,
            }
            return self._baseline_html, analysis, payload_info

        else:
            # XSS payload injection
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                payload = random.choice(self.payloads[family_name]["payloads"])
            else:
                payload = "<script>alert(1)</script>"  # Fallback

            html, status, elapsed = self.client.submit_xss_reflected(payload)
            analysis = self.analyzer.analyze_xss_response(
                html, payload, status, elapsed
            )

            logger.debug(
                f"XSS action={action_name} payload='{payload[:50]}' "
                f"reflected={analysis.payload_reflected} "
                f"script={analysis.script_tag_present} "
                f"event={analysis.event_handler_present} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(
                payload, "name", url_path, html, analysis
            )
            return html, analysis, payload_info

    def _build_payload_info(self, payload: str, parameter: str,
                            url_path: str, html: str,
                            analysis: AnalysisResult) -> dict:
        """Build the payload_info dict for an XSS step."""
        params = {"name": payload}
        full_url = f"{self.dvwa_url}{url_path}?{urlencode(params)}"

        snippet = self._extract_response_snippet(html)
        reflected = analysis.payload_reflected

        return {
            "payload": payload,
            "parameter": parameter,
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    @staticmethod
    def _extract_response_snippet(html: str, max_len: int = 500) -> str:
        """Extract the main content area as a short HTML snippet."""
        soup = BeautifulSoup(html, "lxml")
        main = soup.find("div", {"class": "body_padded"})
        if main:
            # Look for the vulnerable output area
            pres = main.find_all("pre")
            if pres:
                return str(pres[0])[:max_len]
            return main.get_text(strip=True)[:max_len]
        return html[:max_len]

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        """
        Calculate reward for an XSS action.

        Reward structure encourages:
        1. Getting payloads reflected (shows input reaches output)
        2. Getting unescaped script/event handlers (actual XSS)
        3. Efficiency (fewer steps)
        """
        reward = -1.0  # Step penalty

        # Reflection detected (payload appears in response)
        if analysis.payload_reflected and not self.agent_memory.get("found_reflection"):
            reward += 10.0
            logger.debug("  +10: Payload reflected in response")

        # Response differs from baseline
        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 10.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +10: Response differs from baseline")

        # Unescaped script tag (actual XSS!)
        if analysis.script_tag_present:
            if not self.agent_memory.get("found_xss_execution"):
                reward += 40.0
                logger.debug("  +40: Script tag executed in response!")
            # Bonus for confirmed context
            if analysis.xss_context in ("html", "javascript"):
                reward += 40.0
                logger.debug(f"  +40: XSS in {analysis.xss_context} context!")

        # Event handler reflected unescaped
        if analysis.event_handler_present:
            if not self.agent_memory.get("_rewarded_event"):
                reward += 40.0
                self.agent_memory["_rewarded_event"] = True
                logger.debug("  +40: Event handler reflected!")

        # Report done
        if action_id == 11:
            if self.agent_memory.get("found_xss_execution"):
                reward += 20.0
            else:
                reward -= 20.0

        # Penalty for loops
        recent_actions = [s.get("action_id") for s in self.episode_log[-3:]]
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            reward -= 5.0

        # Exploration bonus
        if action_id not in self.agent_memory.get("tried_actions", set()):
            reward += 2.0

        return reward

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        """Extract observation from current state."""
        return self.extractor.extract_xss_state(
            html, analysis, self.agent_memory
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """
        Episode succeeds when an XSS payload executes
        (script tag or event handler reflected unescaped).
        """
        return analysis.script_tag_present or analysis.event_handler_present

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
