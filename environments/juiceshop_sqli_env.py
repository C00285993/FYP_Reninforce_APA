"""
Juice Shop SQL Injection Environment
Gymnasium environment for training an RL agent to discover and exploit
SQL injection vulnerabilities on OWASP Juice Shop (SQLite backend).

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

from environments.base_env import BasePentestEnv
from environments.feature_extractors import UNIFIED_SQLI_STATE_SIZE, extract_unified_sqli_state
from utils.response_analyzer import AnalysisResult
from utils.juiceshop_client import JuiceShopClient

logger = logging.getLogger(__name__)


class JuiceShopSQLiEnv(BasePentestEnv):
    """
    RL Environment for SQL Injection testing on OWASP Juice Shop.

    Juice Shop uses a SQLite backend and REST APIs that return JSON.
    The agent targets two attack surfaces:
      - Product search: GET /rest/products/search?q=<payload>
      - Login form:     POST /rest/user/login {email: <payload>}

    Actions:
        0: submit_baseline      - Submit normal search query
        1: inject_single_quote  - Test with single quote
        2: inject_or_true       - Boolean-based OR true
        3: inject_union_select  - UNION SELECT data extraction
        4: inject_login_bypass  - Login form SQLi bypass
        5: inject_time_based    - Time-based blind injection
        6: inject_error_based   - Error-based extraction
        7: inject_subquery      - Subquery-based extraction
        8: inject_encoded       - URL-encoded variants
        9: report_done          - Agent declares finished

    Rewards:
        -1.0  per step (efficiency penalty)
        +15   SQL error detected
        +25   response differs from baseline
        +50   data leak detected (products returned via injection)
        +100  challenge solved on scoreboard
        -20   report_done without finding anything
        -5    repeating exact same action 3+ times
    """

    ACTIONS = {
        0: "submit_baseline",
        1: "inject_single_quote",
        2: "inject_or_true",
        3: "inject_union_select",
        4: "inject_login_bypass",
        5: "inject_time_based",
        6: "inject_error_based",
        7: "inject_subquery",
        8: "inject_encoded",
        9: "report_done",
    }

    ACTION_TO_FAMILY = {
        0: None,
        1: "single_quote",
        2: "or_true",
        3: "union_select",
        4: "login_bypass",
        5: "time_based",
        6: "error_based",
        7: "subquery",
        8: "encoded",
        9: None,
    }

    def _create_client(self):
        """Override to use JuiceShopClient instead of DVWAClient."""
        return JuiceShopClient(base_url=self.dvwa_url)

    def _vuln_type(self) -> str:
        return "juiceshop_sqli"

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_SQLI_STATE_SIZE,),
            dtype=np.float32,
        )

    def _load_payloads(self) -> dict:
        payload_file = (
            Path(__file__).parent.parent
            / "payloads"
            / "juiceshop_sqli_payloads.json"
        )
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    def _get_initial_page(self) -> str:
        text, _ = self.client.get_page("juiceshop_sqli")
        return text

    def _capture_baselines(self):
        """Submit a normal search to capture the baseline response."""
        text, status, _ = self.client.search_products("apple")
        self.analyzer.set_baseline("juiceshop_sqli", text)
        self._baseline_text = text

    def _execute_action(
        self, action_id: int
    ) -> tuple[str, AnalysisResult, dict]:
        """Execute a SQLi action against Juice Shop."""
        action_name = self.ACTIONS[action_id]

        if action_id == 0:
            # Baseline: normal search query
            query = random.choice(["apple", "banana", "juice", "orange", "1"])
            text, status, elapsed = self.client.search_products(query)
            analysis = self.analyzer.analyze_juiceshop_sqli_response(
                text, query, status, elapsed
            )
            payload_info = self._build_payload_info(
                query, "q", "/rest/products/search", text, status
            )
            return text, analysis, payload_info

        elif action_id == 9:
            # Report done
            analysis = AnalysisResult()
            analysis.has_data_leak = self.agent_memory.get(
                "found_data_leak", False
            )
            payload_info = {
                "payload": "",
                "parameter": "",
                "url_path": "",
                "full_request_url": "",
                "response_snippet": "",
                "reflected": False,
            }
            return self._baseline_text, analysis, payload_info

        elif action_id == 4:
            # Login bypass — targets the login endpoint
            family_name = self.ACTION_TO_FAMILY[action_id]
            payload = random.choice(
                self.payloads[family_name]["payloads"]
            )
            text, status, elapsed = self.client.submit_login(
                payload, "anything"
            )
            analysis = self.analyzer.analyze_juiceshop_sqli_response(
                text, payload, status, elapsed
            )
            # Check scoreboard for new challenge solves
            new_solves = self.client.check_new_solves()
            if new_solves:
                analysis.auth_bypass = True

            payload_info = self._build_payload_info(
                payload, "email", "/rest/user/login", text, status
            )
            return text, analysis, payload_info

        else:
            # Search endpoint injection
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                payload = random.choice(
                    self.payloads[family_name]["payloads"]
                )
            else:
                payload = "'"

            text, status, elapsed = self.client.search_products(payload)
            analysis = self.analyzer.analyze_juiceshop_sqli_response(
                text, payload, status, elapsed
            )

            # Check scoreboard for new challenge solves
            new_solves = self.client.check_new_solves()
            if new_solves:
                # A challenge was solved — mark as significant data leak
                if not analysis.has_data_leak:
                    analysis.has_data_leak = True
                    analysis.leaked_data_count = max(
                        analysis.leaked_data_count, 1
                    )

            logger.debug(
                f"JuiceShop SQLi action={action_name} payload='{payload}' "
                f"error={analysis.has_sql_error} leak={analysis.has_data_leak} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(
                payload, "q", "/rest/products/search", text, status
            )
            return text, analysis, payload_info

    def _build_payload_info(
        self,
        payload: str,
        parameter: str,
        url_path: str,
        response_text: str,
        status_code: int,
    ) -> dict:
        """Build the payload_info dict for a Juice Shop SQLi step."""
        full_url = f"{self.dvwa_url}{url_path}?{parameter}={payload}"

        # Extract a short snippet from the response
        snippet = response_text[:500]

        reflected = payload.lower() in response_text.lower()

        return {
            "payload": payload,
            "parameter": parameter,
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    def _calculate_reward(
        self, action_id: int, analysis: AnalysisResult
    ) -> float:
        """
        Calculate reward for a Juice Shop SQLi action.
        Same reward structure as DVWA SQLi for consistency.
        """
        reward = -1.0  # Step penalty

        # SQL error detection
        if analysis.has_sql_error and not self.agent_memory.get(
            "found_sql_error"
        ):
            reward += 15.0
            logger.debug("  +15: First SQL error detected")

        # Response differs from baseline
        if analysis.response_differs and not self.agent_memory.get(
            "_rewarded_diff"
        ):
            reward += 10.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +10: Response differs from baseline")

        # Data extraction
        if analysis.has_data_leak:
            if not self.agent_memory.get("found_data_leak"):
                reward += 50.0
                logger.debug("  +50: First data leak!")
            if analysis.leaked_data_count > 5 and not self.agent_memory.get(
                "_rewarded_multi"
            ):
                reward += 50.0
                self.agent_memory["_rewarded_multi"] = True
                logger.debug("  +50: Large data extraction!")

        # Auth bypass (login SQLi)
        if analysis.auth_bypass and not self.agent_memory.get(
            "_rewarded_auth"
        ):
            reward += 30.0
            self.agent_memory["_rewarded_auth"] = True
            logger.debug("  +30: Auth bypass / challenge solved!")

        # Report done
        if action_id == 9:
            if self.agent_memory.get("found_data_leak"):
                reward += 20.0
            else:
                reward -= 20.0

        # Repetition penalty
        recent_actions = [s.get("action_id") for s in self.episode_log[-3:]]
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            reward -= 5.0

        # Exploration bonus
        if action_id not in self.agent_memory.get("tried_actions", set()):
            reward += 2.0

        return reward

    def _extract_state(
        self, response_text: str, analysis: AnalysisResult
    ) -> np.ndarray:
        """Extract observation from current state."""
        return extract_unified_sqli_state(
            response_text, analysis, self.agent_memory, response_format="json"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when data is extracted or auth is bypassed."""
        return (
            (analysis.has_data_leak and analysis.leaked_data_count >= 1)
            or analysis.auth_bypass
        )

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
