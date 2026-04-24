"""
SQL Injection Environment
Gymnasium environment for training an RL agent to discover and exploit
SQL injection vulnerabilities on DVWA's SQLi page.

Action space: 10 discrete actions (payload categories + meta actions)
Observation space: 18-dimensional float vector
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
from environments.feature_extractors import UNIFIED_SQLI_STATE_SIZE, extract_unified_sqli_state
from utils.response_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class SQLiEnv(BasePentestEnv):
    """
    RL Environment for SQL Injection testing on DVWA.

    The agent learns to select SQL injection payloads to test against
    DVWA's SQL Injection page. The goal is to extract data from the
    database (the 'users' table).

    Actions:
        0: submit_baseline      - Submit normal input (e.g., "1")
        1: inject_single_quote  - Test with single quote
        2: inject_or_true       - Boolean-based OR true
        3: inject_union_select  - UNION SELECT data extraction
        4: inject_comment_bypass - Comment-based bypass
        5: inject_time_based    - Time-based blind injection
        6: inject_error_based   - Error-based extraction
        7: inject_stacked       - Stacked queries
        8: inject_encoded       - URL-encoded variants
        9: report_done          - Agent declares finished

    Rewards:
        -1.0  per step (efficiency penalty)
        +15   SQL error detected
        +25   response differs from baseline
        +50   data leak detected
        +100  multiple user records extracted (full exploit)
        -20   report_done without finding anything
        -5    repeating exact same action 3+ times
    """

    # Action definitions
    ACTIONS = {
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

    # Map action IDs to payload family names (from sqli_payloads.json)
    ACTION_TO_FAMILY = {
        0: None,  # baseline
        1: "single_quote",
        2: "or_true",
        3: "union_select",
        4: "comment_bypass",
        5: "time_based",
        6: "error_based",
        7: "stacked",
        8: "encoded",
        9: None,  # meta action
    }

    def _vuln_type(self) -> str:
        return "sqli"

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_SQLI_STATE_SIZE,),
            dtype=np.float32,
        )

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "sqli_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("sqli")
        return html

    def _capture_baselines(self):
        """Submit a normal query to capture the baseline response."""
        html, status, _ = self.client.submit_sqli("1")
        self.analyzer.set_baseline("sqli", html)
        self._baseline_html = html

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute a SQLi action against DVWA."""
        action_name = self.ACTIONS[action_id]
        url_path = self.client.PAGES["sqli"]

        if action_id == 0:
            # Baseline: submit a normal numeric ID
            payload = str(random.randint(1, 5))
            html, status, elapsed = self.client.submit_sqli(payload)
            analysis = self.analyzer.analyze_sqli_response(
                html, payload, status, elapsed
            )
            payload_info = self._build_payload_info(
                payload, "id", url_path, html, analysis
            )
            return html, analysis, payload_info

        elif action_id == 9:
            # Report done — no HTTP request, just signal
            analysis = AnalysisResult()
            # Check if agent actually found something
            analysis.has_data_leak = self.agent_memory.get("found_data_leak", False)
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
            # Injection payload — use LLM generator when available, else random
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                fallback = self.payloads[family_name]["payloads"]
                payload = self._pick_payload(family_name, fallback)
            else:
                payload = "'"  # Fallback

            html, status, elapsed = self.client.submit_sqli(payload)
            self._last_payload = payload
            self._last_response = html
            self._last_status = status
            analysis = self.analyzer.analyze_sqli_response(
                html, payload, status, elapsed
            )

            logger.debug(
                f"SQLi action={action_name} payload='{payload}' "
                f"error={analysis.has_sql_error} leak={analysis.has_data_leak} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(
                payload, "id", url_path, html, analysis
            )
            return html, analysis, payload_info

    def _build_payload_info(self, payload: str, parameter: str,
                            url_path: str, html: str,
                            analysis: AnalysisResult) -> dict:
        """Build the payload_info dict for a SQLi step."""
        params = {"id": payload, "Submit": "Submit"}
        if self.security_level in ("medium", "high"):
            full_url = f"{self.dvwa_url}{url_path} [POST id={payload}]"
        else:
            full_url = f"{self.dvwa_url}{url_path}?{urlencode(params)}"

        # Extract a response snippet from the main content area
        snippet = self._extract_response_snippet(html)

        reflected = payload.lower() in html.lower()

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
        # DVWA wraps vuln output in <div class="body_padded"> or <pre> tags
        main = soup.find("div", {"class": "body_padded"})
        if main:
            # Look for <pre> tags (where query results appear)
            pres = main.find_all("pre")
            if pres:
                return str(pres[0])[:max_len]
            # Fall back to the full main content text
            return main.get_text(strip=True)[:max_len]
        return html[:max_len]

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        """
        Calculate reward for a SQLi action.

        Reward structure encourages:
        1. Finding SQL errors (evidence of injectability)
        2. Extracting data (actual exploitation)
        3. Efficiency (fewer steps is better)
        """
        reward = -1.0  # Step penalty

        # Detection rewards (only award once per type per episode)
        if analysis.has_sql_error and not self.agent_memory.get("found_sql_error"):
            reward += 15.0
            logger.debug("  +15: First SQL error detected")

        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 10.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +10: Response differs from baseline")

        # Data extraction rewards
        if analysis.has_data_leak:
            if not self.agent_memory.get("found_data_leak"):
                reward += 50.0
                logger.debug("  +50: First data leak!")
            if analysis.leaked_data_count > 1 and not self.agent_memory.get("_rewarded_multi"):
                reward += 50.0
                self.agent_memory["_rewarded_multi"] = True
                logger.debug("  +50: Multiple records extracted!")

        # Auth bypass
        if analysis.auth_bypass and not self.agent_memory.get("_rewarded_auth"):
            reward += 30.0
            self.agent_memory["_rewarded_auth"] = True
            logger.debug("  +30: Auth bypass detected!")

        # Report done action
        if action_id == 9:
            if self.agent_memory.get("found_data_leak"):
                reward += 20.0  # Good: reported after finding something
            else:
                reward -= 20.0  # Bad: gave up without finding anything

        # Penalty for repeating the same action excessively
        recent_actions = [s.get("action_id") for s in self.episode_log[-3:]]
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            reward -= 5.0  # Stuck in a loop

        # Small exploration bonus for trying new action types
        if action_id not in self.agent_memory.get("tried_actions", set()):
            reward += 2.0

        return reward

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        """Extract observation from current state."""
        return extract_unified_sqli_state(
            html, analysis, self.agent_memory, response_format="html"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when data is extracted from the database."""
        if not (analysis.has_data_leak and analysis.leaked_data_count >= 1):
            return False
        return self._llm_confirm_success(analysis)

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
