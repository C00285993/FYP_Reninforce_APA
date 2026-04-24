"""
DVWA Stored (Persistent) XSS Environment
Gymnasium environment for training an RL agent to discover and exploit
stored XSS vulnerabilities on DVWA's Stored XSS (Guestbook) page.

Attack surface: POST /vulnerabilities/xss_s/
  - txtName: Name field (10-char limit on low security, bypassed on med/high)
  - mtxMessage: Message field (main injection target)

Stored XSS detection:
  After submitting a payload via POST, the env GETs the same page to check
  whether the payload persists and renders unescaped in the guestbook entries.
  The guestbook is cleared between episodes to prevent cross-contamination.

Action space: 12 discrete actions (same as reflected XSS)
Observation space: 20-dimensional float vector (unified XSS state)
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

logger = logging.getLogger(__name__)


class DVWAStoredXSSEnv(BasePentestEnv):
    """
    RL Environment for Stored XSS testing on DVWA's guestbook page.

    The agent POSTs a payload into the guestbook, then the env GETs the page
    to check if the payload is stored and rendered unescaped — which would
    execute in any visitor's browser (persistent XSS).

    Actions: identical to reflected XSSEnv (12 actions, same families)
    Observation: unified 20-dim XSS state vector

    Rewards:
        -1.0  per step (efficiency penalty)
        +10   payload reflected in stored page
        +20   response differs from baseline
        +40   unescaped script/event handler persists in guestbook
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
            shape=(UNIFIED_XSS_STATE_SIZE,),
            dtype=np.float32,
        )

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "xss_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("xss_stored")
        return html

    def _capture_baselines(self):
        """Clear guestbook and capture the clean page as baseline."""
        self.client.reset_xss_stored_db()
        html, _ = self.client.get_page("xss_stored")
        self.analyzer.set_baseline("xss_stored", html)
        self._baseline_html = html

    def reset(self, seed=None, options=None):
        """Reset: clear guestbook so each episode starts clean."""
        super().reset(seed=seed)
        # Clear guestbook entries from previous episode
        self.client.reset_xss_stored_db()

        html = self._get_initial_page()
        obs = self._extract_state(html, AnalysisResult())
        info = {
            "episode": self.episode_count,
            "target": self.dvwa_url,
            "vuln_type": "xss_stored",
        }
        return obs, info

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Submit payload to guestbook, then GET the page to check persistence."""
        url_path = self.client.PAGES["xss_stored"]

        if action_id == 0:
            # Baseline: submit harmless text
            payload = random.choice(["hello", "test", "world", "user123"])
            html, status, elapsed = self.client.submit_xss_stored(
                name="TestUser", message=payload
            )
            analysis = self.analyzer.analyze_xss_response(
                html, payload, status, elapsed
            )
            payload_info = self._build_payload_info(payload, url_path, html, analysis)
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
            # XSS payload injection into the guestbook message field
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                payload = random.choice(self.payloads[family_name]["payloads"])
            else:
                payload = "<script>alert(1)</script>"

            # POST the payload into the guestbook
            html, status, elapsed = self.client.submit_xss_stored(
                name="Agent", message=payload
            )

            # The POST response includes the updated guestbook page.
            # Analyze whether our payload persists and executes.
            analysis = self.analyzer.analyze_xss_response(
                html, payload, status, elapsed
            )

            logger.debug(
                f"Stored XSS action={self.ACTIONS[action_id]} "
                f"payload='{payload[:50]}' "
                f"reflected={analysis.payload_reflected} "
                f"script={analysis.script_tag_present} "
                f"event={analysis.event_handler_present} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(
                payload, url_path, html, analysis
            )
            return html, analysis, payload_info

    def _build_payload_info(self, payload: str, url_path: str,
                            html: str, analysis: AnalysisResult) -> dict:
        full_url = f"{self.dvwa_url}{url_path}"
        snippet = html[:500]
        return {
            "payload": payload,
            "parameter": "mtxMessage",
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": analysis.payload_reflected,
        }

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0  # Step penalty

        # Reflection detected (payload stored and rendered in page)
        if analysis.payload_reflected and not self.agent_memory.get("found_reflection"):
            reward += 10.0

        # Response differs from baseline
        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 20.0
            self.agent_memory["_rewarded_diff"] = True

        # Unescaped script tag persisted
        if analysis.script_tag_present:
            if not self.agent_memory.get("found_xss_execution"):
                reward += 40.0
            if analysis.xss_context in ("html", "javascript"):
                reward += 40.0

        # Event handler persisted
        if analysis.event_handler_present:
            if not self.agent_memory.get("_rewarded_event"):
                reward += 40.0
                self.agent_memory["_rewarded_event"] = True

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
        return extract_unified_xss_state(
            html, analysis, self.agent_memory, response_format="html"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Stored XSS succeeds when the payload persists and executes."""
        return analysis.script_tag_present or analysis.event_handler_present

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
