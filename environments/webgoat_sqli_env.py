"""
WebGoat SQL Injection Environment
Gymnasium environment for training an RL agent to exploit SQL injection
on OWASP WebGoat's "String SQL Injection" exercise.

WebGoat uses an H2 in-memory database (Java/Spring Boot), making it a
different technology stack from DVWA (MySQL/PHP) and Juice Shop (SQLite/Node).
This environment shares identical action / observation spaces with SQLiEnv
(DVWA) so a pre-trained model can be transferred via fine-tuning.

Target lesson: SQL Injection (Intro) — exercise 3 (String SQL Injection)
Endpoint: POST /WebGoat/SqlInjection/attack3  (parameter: ``account``)
Response: JSON {lessonCompleted, feedback, output (HTML table of users)}

Action space:  10 discrete actions (identical to SQLiEnv)
Observation space: 18-dimensional float vector (UNIFIED_SQLI_STATE_SIZE)
"""

import json
import random
import logging
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from gymnasium import spaces

from environments.base_env import BasePentestEnv
from environments.feature_extractors import UNIFIED_SQLI_STATE_SIZE, extract_unified_sqli_state
from utils.response_analyzer import AnalysisResult
from utils.webgoat_client import WebGoatClient

logger = logging.getLogger(__name__)


class WebGoatSQLiEnv(BasePentestEnv):
    """
    RL Environment for SQL Injection testing on OWASP WebGoat.

    Identical action semantics and reward structure to SQLiEnv (DVWA) so
    that a model trained on DVWA's SQL injection page can be transferred
    and fine-tuned on WebGoat's Java/Spring/H2 SQL injection exercise.

    The ``account`` field in WebGoat's String SQL Injection lesson mirrors
    DVWA's ``id`` field — both accept a value, run a SELECT, and return
    user records.  A successful injection returns all user rows.

    Actions:
        0: submit_baseline        - Submit normal account name ("Smith")
        1: inject_single_quote    - Single quote to break SQL context
        2: inject_or_true         - Boolean-based OR true (OR '1'='1)
        3: inject_union_select    - UNION SELECT data extraction
        4: inject_comment_bypass  - Comment-based bypass (--, #, /**/)
        5: inject_time_based      - Time-based blind injection
        6: inject_error_based     - Error-based extraction
        7: inject_stacked         - Stacked queries
        8: inject_encoded         - URL-encoded variants
        9: report_done            - Agent declares finished

    Rewards: identical to SQLiEnv
        -1.0  per step
        +15   SQL error detected
        +10   response differs from baseline
        +50   first data leak (rows extracted)
        +50   multiple records extracted
        +30   auth bypass (lessonCompleted)
        -20   report_done without finding anything
        -5    repeating same action 3+ times in a row
        +2    exploration bonus for first use of each action type
    """

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

    ACTION_TO_FAMILY = {
        0: None,
        1: "single_quote",
        2: "or_true",
        3: "union_select",
        4: "comment_bypass",
        5: "time_based",
        6: "error_based",
        7: "stacked",
        8: "encoded",
        9: None,
    }

    # ------------------------------------------------------------------ #
    #  Client & identity
    # ------------------------------------------------------------------ #

    def _create_client(self):
        """Use WebGoatClient instead of DVWAClient."""
        return WebGoatClient(base_url=self.dvwa_url)

    def _vuln_type(self) -> str:
        return "sqli"

    # ------------------------------------------------------------------ #
    #  Spaces
    # ------------------------------------------------------------------ #

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_SQLI_STATE_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Payload loading
    # ------------------------------------------------------------------ #

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "sqli_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("sqli")
        return html

    def _capture_baselines(self):
        """Submit an empty injection to capture the WebGoat baseline (no data returned)."""
        response_text, status, _ = self.client.submit_sqli("")
        self.analyzer.set_baseline("webgoat_sqli", response_text)
        self._baseline_html = response_text

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute a SQLi action against WebGoat."""
        action_name = self.ACTIONS[action_id]
        url_path = self.client._SQLI_PATH

        if action_id == 0:
            # Baseline: empty injection → no data returned
            payload = ""
            response_text, status, elapsed = self.client.submit_sqli(payload)
            analysis = self.analyzer.analyze_webgoat_sqli_response(
                response_text, payload, status, elapsed
            )
            payload_info = self._build_payload_info(payload, url_path, response_text, analysis)
            return response_text, analysis, payload_info

        elif action_id == 9:
            # Report done — no HTTP request
            analysis = AnalysisResult()
            analysis.has_data_leak = self.agent_memory.get("found_data_leak", False)
            payload_info = {
                "payload": "",
                "parameter": "account",
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
                payload = "'"

            response_text, status, elapsed = self.client.submit_sqli(payload)
            analysis = self.analyzer.analyze_webgoat_sqli_response(
                response_text, payload, status, elapsed
            )

            logger.debug(
                f"WebGoat SQLi action={action_name} payload={payload!r} "
                f"error={analysis.has_sql_error} leak={analysis.has_data_leak} "
                f"rows={analysis.leaked_data_count} severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(payload, url_path, response_text, analysis)
            return response_text, analysis, payload_info

    def _build_payload_info(
        self, payload: str, url_path: str, response_text: str, analysis: AnalysisResult
    ) -> dict:
        full_url = f"{self.dvwa_url}{url_path} [POST account=Smith&operator==&injection={payload}]"

        # Extract output snippet from WebGoat JSON response
        snippet = ""
        try:
            data = json.loads(response_text)
            output_html = data.get("output", "") or ""
            feedback = data.get("feedback", "") or ""
            if output_html:
                soup = BeautifulSoup(output_html, "lxml")
                snippet = soup.get_text(strip=True)[:500]
            elif feedback:
                snippet = feedback[:500]
        except (ValueError, TypeError):
            snippet = response_text[:500]

        reflected = payload.lower() in response_text.lower()
        return {
            "payload": payload,
            "parameter": "account",
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    # ------------------------------------------------------------------ #
    #  Reward — mirrors SQLiEnv
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0  # Step penalty

        if analysis.has_sql_error and not self.agent_memory.get("found_sql_error"):
            reward += 15.0
            logger.debug("  +15: First SQL error detected")

        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 10.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +10: Response differs from baseline")

        if analysis.has_data_leak:
            if not self.agent_memory.get("found_data_leak"):
                reward += 50.0
                logger.debug("  +50: First data leak!")
            if analysis.leaked_data_count > 1 and not self.agent_memory.get("_rewarded_multi"):
                reward += 50.0
                self.agent_memory["_rewarded_multi"] = True
                logger.debug("  +50: Multiple records extracted!")

        if analysis.auth_bypass and not self.agent_memory.get("_rewarded_auth"):
            reward += 30.0
            self.agent_memory["_rewarded_auth"] = True
            logger.debug("  +30: lessonCompleted (auth bypass)")

        if action_id == 9:
            if self.agent_memory.get("found_data_leak"):
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
        memory["found_sql_error"] = False
        memory["found_data_leak"] = False
        return memory

    def _update_agent_memory(self, action_id: int, analysis: AnalysisResult):
        super()._update_agent_memory(action_id, analysis)
        if analysis.has_sql_error:
            self.agent_memory["found_sql_error"] = True
        if analysis.has_data_leak:
            self.agent_memory["found_data_leak"] = True

    # ------------------------------------------------------------------ #
    #  State extraction & termination
    # ------------------------------------------------------------------ #

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        return extract_unified_sqli_state(
            html, analysis, self.agent_memory, response_format="webgoat"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when user data is extracted from the database."""
        return analysis.has_data_leak and analysis.leaked_data_count >= 1

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
