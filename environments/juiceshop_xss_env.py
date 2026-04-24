"""
Juice Shop XSS (Cross-Site Scripting) Environment
Gymnasium environment for training an RL agent to discover and exploit
XSS vulnerabilities on OWASP Juice Shop via the product search endpoint.

Attack surface: GET /rest/products/search?q=<payload>
Target challenge: "DOM XSS" — rendered via Angular innerHTML.

Action space: 12 discrete actions (same as DVWA XSSEnv)
Observation space: 20-dimensional float vector (unified XSS state)

DOM XSS detection modes
-----------------------
HTTP-only (default, use_playwright=False):
    Sends payloads to the REST API and analyses the JSON response for
    reflection heuristics.  Cannot confirm actual script execution because
    Angular's innerHTML binding runs entirely in the browser.

Playwright (use_playwright=True):
    Drives a real Chromium browser alongside the HTTP session.  The browser
    navigates to the SPA search route and Playwright intercepts any alert()
    call triggered by the injected payload — the definitive DOM XSS signal.
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
from utils.juiceshop_client import JuiceShopClient

logger = logging.getLogger(__name__)


class JuiceShopXSSEnv(BasePentestEnv):
    """
    RL Environment for XSS testing on OWASP Juice Shop.

    Juice Shop is an Angular SPA. XSS execution happens client-side when
    the search query is rendered via innerHTML. The agent targets the
    product search endpoint:
        GET /rest/products/search?q=<payload>

    Detection strategy:
    - HTTP heuristic: payload reflection + unescaped HTML tags in JSON values
    - Playwright (optional): real browser intercepts alert() — definitive signal
    - Scoreboard: challenge solve confirmed via API

    Actions:
        0:  submit_baseline       - Submit normal text input
        1:  inject_basic_script   - Basic <script>alert(1)</script>
        2:  inject_img_onerror    - <img src=x onerror=alert(1)>
        3:  inject_svg_onload     - <svg onload=alert(1)>
        4:  inject_event_handler  - Various event handler payloads
        5:  inject_attr_escape    - Break out of HTML attributes
        6:  inject_case_bypass    - Case variation bypass
        7:  inject_encoding       - Encoding-based evasion
        8:  inject_nested_tags    - Nested/broken tags (iframe vectors)
        9:  inject_dom_based      - DOM-based vectors
        10: inject_polyglot       - Multi-context polyglots
        11: report_done           - Agent declares finished

    Rewards:
        -1.0  per step (efficiency penalty)
        +10   payload reflected in response
        +10   response differs from baseline
        +40   unescaped script/event handler detected in response
        +40   confirmed XSS context
        +100  challenge solved on scoreboard OR browser alert fired
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

    def __init__(self, *args, use_playwright: bool = False,
                 headless: bool = True, **kwargs):
        """
        Args:
            use_playwright: Launch a real Chromium browser for DOM XSS
                            detection (slower but accurate).
            headless:       Run browser headlessly (default True).
                            Set False to watch the browser during debugging.
            *args / **kwargs: Forwarded to BasePentestEnv.
        """
        # Store before super().__init__() calls _create_client()
        self._use_playwright = use_playwright
        self._headless = headless
        super().__init__(*args, **kwargs)

    def _create_client(self):
        """Return a Playwright-capable client when requested."""
        if self._use_playwright:
            from utils.playwright_juiceshop_client import PlaywrightJuiceShopClient
            logger.info("JuiceShopXSSEnv: using PlaywrightJuiceShopClient")
            return PlaywrightJuiceShopClient(
                base_url=self.dvwa_url,
                headless=self._headless,
            )
        return JuiceShopClient(base_url=self.dvwa_url)

    def close(self):
        """Clean up Playwright browser if it was used."""
        if hasattr(self.client, "close_browser"):
            self.client.close_browser()
        super().close()

    def _vuln_type(self) -> str:
        return "juiceshop_xss"

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_XSS_STATE_SIZE,),
            dtype=np.float32,
        )

    def _load_payloads(self) -> dict:
        payload_file = (
            Path(__file__).parent.parent
            / "payloads"
            / "juiceshop_xss_payloads.json"
        )
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    def _get_initial_page(self) -> str:
        text, _ = self.client.get_page("juiceshop_xss")
        return text

    def _capture_baselines(self):
        """Submit a normal search to capture the baseline response."""
        text, status, _ = self.client.search_products("apple")
        self.analyzer.set_baseline("juiceshop_xss", text)
        self._baseline_text = text

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute an XSS action against Juice Shop's search endpoint."""
        action_name = self.ACTIONS[action_id]

        if action_id == 0:
            # Baseline: submit harmless text
            payload = random.choice(["apple", "banana", "juice", "orange", "test"])
            text, status, elapsed = self.client.search_products(payload)
            analysis = self.analyzer.analyze_juiceshop_xss_response(
                text, payload, status, elapsed
            )
            payload_info = self._build_payload_info(payload, text, status)
            return text, analysis, payload_info

        elif action_id == 11:
            # Report done
            analysis = AnalysisResult()
            analysis.script_tag_present = self.agent_memory.get(
                "found_xss_execution", False
            )
            payload_info = {
                "payload": "",
                "parameter": "",
                "url_path": "/rest/products/search",
                "full_request_url": "",
                "response_snippet": "",
                "reflected": False,
            }
            return self._baseline_text, analysis, payload_info

        else:
            # XSS payload injection
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                payload = random.choice(self.payloads[family_name]["payloads"])
            else:
                payload = "<script>alert(1)</script>"

            # HTTP request: gets the server's JSON response for heuristics
            text, status, elapsed = self.client.search_products(payload)

            # Playwright browser navigation: definitive DOM XSS detection
            browser_result = None
            if hasattr(self.client, "navigate_xss_payload"):
                browser_result = self.client.navigate_xss_payload(payload)

            # Scoreboard check (runs after browser navigation to catch solves)
            new_solves = self.client.check_new_solves()

            analysis = self.analyzer.analyze_juiceshop_xss_response(
                text, payload, status, elapsed,
                challenge_solves=new_solves,
            )

            # Browser alert overrides heuristics — this is ground-truth success
            if browser_result and browser_result["alert_fired"]:
                analysis.script_tag_present = True
                analysis.xss_context = "html"
                logger.info(
                    f"[PLAYWRIGHT] DOM XSS confirmed! "
                    f"payload='{payload[:60]}' "
                    f"alert='{browser_result['alert_message']}'"
                )

            logger.debug(
                f"JuiceShop XSS action={action_name} payload='{payload[:50]}' "
                f"reflected={analysis.payload_reflected} "
                f"script={analysis.script_tag_present} "
                f"event={analysis.event_handler_present} "
                f"browser_alert={browser_result['alert_fired'] if browser_result else 'N/A'} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(payload, text, status)
            if browser_result:
                payload_info["browser_url"] = browser_result["url"]
                payload_info["alert_fired"] = browser_result["alert_fired"]
            return text, analysis, payload_info

    def _build_payload_info(
        self, payload: str, response_text: str, status_code: int
    ) -> dict:
        """Build the payload_info dict for a Juice Shop XSS step."""
        url_path = "/rest/products/search"
        params = {"q": payload}
        full_url = f"{self.dvwa_url}{url_path}?{urlencode(params)}"

        snippet = response_text[:500]
        reflected = payload.lower() in response_text.lower()

        return {
            "payload": payload,
            "parameter": "q",
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        """
        Calculate reward for a Juice Shop XSS action.

        Same structure as DVWA XSSEnv + bonus for challenge solve.
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

        # Unescaped script tag detected
        if analysis.script_tag_present:
            if not self.agent_memory.get("found_xss_execution"):
                reward += 40.0
                logger.debug("  +40: Script tag / DOM XSS vector detected!")
            # Bonus for confirmed context
            if analysis.xss_context in ("html", "javascript"):
                reward += 40.0
                logger.debug(f"  +40: XSS in {analysis.xss_context} context!")

        # Event handler reflected
        if analysis.event_handler_present:
            if not self.agent_memory.get("_rewarded_event"):
                reward += 40.0
                self.agent_memory["_rewarded_event"] = True
                logger.debug("  +40: Event handler reflected!")

        # Challenge solve bonus (definitive success — scoreboard or browser alert)
        if analysis.script_tag_present and not self.agent_memory.get("_rewarded_challenge"):
            reward += 100.0
            self.agent_memory["_rewarded_challenge"] = True
            logger.debug("  +100: Challenge solve bonus!")

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

    def _extract_state(self, response_text: str, analysis: AnalysisResult) -> np.ndarray:
        """Extract observation from current state using unified XSS extractor."""
        return extract_unified_xss_state(
            response_text, analysis, self.agent_memory, response_format="json"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """
        Episode succeeds when an XSS payload is detected
        (script tag or event handler reflected unescaped, or browser alert fired).
        """
        return analysis.script_tag_present or analysis.event_handler_present

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
