"""
WebGoat Command Injection Environment
Gymnasium environment for training an RL agent to exploit OS command injection
on OWASP WebGoat's "Operating System Command Injection" exercise.

The action / observation spaces are identical to CMDiEnv (DVWA) so a model
pre-trained on DVWA can be fine-tuned on WebGoat via transfer learning.

Action space:  10 discrete actions (identical to CMDiEnv)
Observation space: 18-dimensional float vector (UNIFIED_CMDI_STATE_SIZE)
"""

import json
import random
import logging
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from gymnasium import spaces

from environments.base_env import BasePentestEnv
from environments.feature_extractors import UNIFIED_CMDI_STATE_SIZE, extract_unified_cmdi_state
from utils.response_analyzer import AnalysisResult
from utils.webgoat_client import WebGoatClient

logger = logging.getLogger(__name__)


class WebGoatCMDiEnv(BasePentestEnv):
    """
    RL Environment for OS Command Injection testing on OWASP WebGoat.

    Identical action semantics and reward structure to CMDiEnv (DVWA) so that
    a model trained on DVWA's command execution page can be transferred and
    fine-tuned on WebGoat's Java/Spring OS command injection exercise.

    Actions:
        0: submit_baseline        - Submit normal IP input ("127.0.0.1")
        1: inject_separator       - Semicolon command separator (;id)
        2: inject_pipe            - Pipe operator (|id)
        3: inject_logical_and     - Logical AND (&& id)
        4: inject_logical_or      - Logical OR (|| id)
        5: inject_backtick        - Backtick substitution (`id`)
        6: inject_dollar_paren    - Dollar-paren substitution ($(id))
        7: inject_newline         - Newline injection (%0aid)
        8: inject_encoded         - URL-encoded variants
        9: report_done            - Agent declares finished

    Rewards: identical to CMDiEnv
        -1.0  per step
        +15   response differs from baseline
        +20   shell error detected
        +30   command output lines detected
        +40   has_command_execution confirmed
        +60   file operations detected
        -20   report_done without finding anything
        -5    repeating same action 3+ times in a row
        +2    exploration bonus for first use of each action type
    """

    ACTIONS = {
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

    ACTION_TO_FAMILY = {
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

    # ------------------------------------------------------------------ #
    #  Client & identity
    # ------------------------------------------------------------------ #

    def _create_client(self):
        """Use WebGoatClient instead of DVWAClient."""
        return WebGoatClient(base_url=self.dvwa_url)

    def _vuln_type(self) -> str:
        return "cmdi"

    # ------------------------------------------------------------------ #
    #  Spaces
    # ------------------------------------------------------------------ #

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_CMDI_STATE_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Payload loading
    # ------------------------------------------------------------------ #

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "cmdi_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("cmdi")
        return html

    def _capture_baselines(self):
        """Submit a normal IP to capture the baseline WebGoat response."""
        html, status, _ = self.client.submit_cmdi("127.0.0.1")
        self.analyzer.set_baseline("cmdi", html)
        self._baseline_html = html

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute a CMDi action against WebGoat."""
        action_name = self.ACTIONS[action_id]
        url_path = self.client._CMDI_PATH

        if action_id == 0:
            html, status, elapsed = self.client.submit_cmdi("127.0.0.1")
            analysis = self.analyzer.analyze_cmdi_response(
                html, "127.0.0.1", status, elapsed
            )
            payload_info = self._build_payload_info(
                "127.0.0.1", url_path, html, analysis
            )
            return html, analysis, payload_info

        elif action_id == 9:
            analysis = AnalysisResult()
            analysis.has_command_execution = self.agent_memory.get(
                "found_command_exec", False
            )
            payload_info = {
                "payload": "",
                "parameter": "ipAddress",
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
                payload = "; id"

            html, status, elapsed = self.client.submit_cmdi(payload)
            analysis = self.analyzer.analyze_cmdi_response(
                html, payload, status, elapsed
            )

            logger.debug(
                f"WebGoat CMDi action={action_name} payload={payload!r} "
                f"cmd_exec={analysis.has_command_execution} "
                f"file_ops={analysis.file_operations_detected} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(payload, url_path, html, analysis)
            return html, analysis, payload_info

    def _build_payload_info(
        self, payload: str, url_path: str, html: str, analysis: AnalysisResult
    ) -> dict:
        full_url = f"{self.dvwa_url}{url_path} [POST ipAddress={payload}]"
        soup = BeautifulSoup(html, "lxml")
        output_area = (
            soup.find("div", {"class": "lesson-content"})
            or soup.find("div", {"id": "output"})
        )
        snippet = (
            output_area.get_text(strip=True)[:500]
            if output_area
            else soup.get_text(strip=True)[:500]
        )
        reflected = payload.lower() in html.lower()
        return {
            "payload": payload,
            "parameter": "ipAddress",
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    # ------------------------------------------------------------------ #
    #  Reward — identical to CMDiEnv
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0

        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 15.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +15: Response differs from baseline")

        if analysis.shell_error_type and not self.agent_memory.get("_rewarded_shell_err"):
            reward += 20.0
            self.agent_memory["_rewarded_shell_err"] = True
            logger.debug("  +20: Shell error detected")

        if analysis.command_output_lines > 0 and not self.agent_memory.get("_rewarded_lines"):
            reward += 30.0
            self.agent_memory["_rewarded_lines"] = True
            logger.debug(f"  +30: Command output lines ({analysis.command_output_lines})")

        if analysis.has_command_execution and not self.agent_memory.get("found_command_exec"):
            reward += 40.0
            logger.debug("  +40: Command execution confirmed!")

        if analysis.file_operations_detected and not self.agent_memory.get("found_file_operations"):
            reward += 60.0
            logger.debug("  +60: File operations detected!")

        if action_id == 9:
            if self.agent_memory.get("found_command_exec"):
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
        memory["found_command_exec"] = False
        memory["found_file_operations"] = False
        return memory

    def _update_agent_memory(self, action_id: int, analysis: AnalysisResult):
        super()._update_agent_memory(action_id, analysis)
        if analysis.has_command_execution:
            self.agent_memory["found_command_exec"] = True
        if analysis.file_operations_detected:
            self.agent_memory["found_file_operations"] = True

    # ------------------------------------------------------------------ #
    #  State extraction & termination
    # ------------------------------------------------------------------ #

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        return extract_unified_cmdi_state(
            html, analysis, self.agent_memory, response_format="webgoat"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        return analysis.has_command_execution

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
