"""
Command Injection Environment
Gymnasium environment for training an RL agent to discover and exploit
OS command injection vulnerabilities on DVWA's Command Execution page.

Action space: 10 discrete actions (payload families + meta actions)
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
from environments.feature_extractors import UNIFIED_CMDI_STATE_SIZE, extract_unified_cmdi_state
from utils.response_analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class CMDiEnv(BasePentestEnv):
    """
    RL Environment for OS Command Injection testing on DVWA.

    The agent learns to select command injection payloads to test against
    DVWA's Command Execution page (/vulnerabilities/exec/). The goal is
    to execute arbitrary OS commands on the server (e.g. id, whoami,
    cat /etc/passwd).

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

    Rewards:
        -1.0  per step (efficiency penalty)
        +15   response differs from baseline (evidence of code path change)
        +20   shell error detected (injection reached the shell)
        +30   command output lines detected (cmd executed, partial output)
        +40   has_command_execution confirmed
        +60   file operations detected (/etc/passwd, ls output, etc.)
        -20   report_done without finding command execution
        -5    repeating exact same action 3+ times in a row
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
        0: None,                  # baseline
        1: "command_separator",
        2: "pipe_operator",
        3: "logical_and",
        4: "logical_or",
        5: "backtick_exec",
        6: "dollar_paren",
        7: "newline_inject",
        8: "encoded",
        9: None,                  # meta action
    }

    def _vuln_type(self) -> str:
        return "cmdi"

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_CMDI_STATE_SIZE,),
            dtype=np.float32,
        )

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "cmdi_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    def _get_initial_page(self) -> str:
        html, _ = self.client.get_page("cmdi")
        return html

    def _capture_baselines(self):
        """Submit a normal ping target to capture the baseline response."""
        html, status, _ = self.client.submit_cmdi("127.0.0.1")
        self.analyzer.set_baseline("cmdi", html)
        self._baseline_html = html

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute a CMDi action against DVWA."""
        action_name = self.ACTIONS[action_id]
        url_path = self.client.PAGES["cmdi"]

        if action_id == 0:
            # Baseline: submit a normal IP address
            html, status, elapsed = self.client.submit_cmdi("127.0.0.1")
            analysis = self.analyzer.analyze_cmdi_response(
                html, "127.0.0.1", status, elapsed
            )
            payload_info = self._build_payload_info(
                "127.0.0.1", "ip", url_path, html, analysis
            )
            return html, analysis, payload_info

        elif action_id == 9:
            # Report done — no HTTP request, just signal
            analysis = AnalysisResult()
            analysis.has_command_execution = self.agent_memory.get(
                "found_command_exec", False
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
            # Injection payload
            family_name = self.ACTION_TO_FAMILY[action_id]
            if family_name and family_name in self.payloads:
                payload = random.choice(self.payloads[family_name]["payloads"])
            else:
                payload = "; id"  # Fallback

            html, status, elapsed = self.client.submit_cmdi(payload)
            analysis = self.analyzer.analyze_cmdi_response(
                html, payload, status, elapsed
            )

            logger.debug(
                f"CMDi action={action_name} payload={payload!r} "
                f"cmd_exec={analysis.has_command_execution} "
                f"file_ops={analysis.file_operations_detected} "
                f"severity={analysis.severity_score}"
            )

            payload_info = self._build_payload_info(
                payload, "ip", url_path, html, analysis
            )
            return html, analysis, payload_info

    def _build_payload_info(self, payload: str, parameter: str,
                            url_path: str, html: str,
                            analysis: AnalysisResult) -> dict:
        """Build the payload_info dict for a CMDi step."""
        full_url = f"{self.dvwa_url}{url_path} [POST ip={payload}]"
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
        """Extract the main content area (command output) as a short snippet."""
        soup = BeautifulSoup(html, "lxml")
        main = soup.find("div", {"class": "body_padded"})
        if main:
            pres = main.find_all("pre")
            if pres:
                return str(pres[0])[:max_len]
            return main.get_text(strip=True)[:max_len]
        return html[:max_len]

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        """
        Calculate reward for a CMDi action.

        Reward structure:
        - Encourages detecting that the shell executed a command
        - Rewards escalating evidence: diff → shell error → cmd output → file ops
        - Penalises giving up without finding execution
        """
        reward = -1.0  # Step penalty

        # One-time rewards for each level of evidence
        if analysis.response_differs and not self.agent_memory.get("_rewarded_diff"):
            reward += 15.0
            self.agent_memory["_rewarded_diff"] = True
            logger.debug("  +15: Response differs from baseline")

        if analysis.shell_error_type and not self.agent_memory.get("_rewarded_shell_err"):
            reward += 20.0
            self.agent_memory["_rewarded_shell_err"] = True
            logger.debug("  +20: Shell error detected (injection reached shell)")

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

        # Report done action
        if action_id == 9:
            if self.agent_memory.get("found_command_exec"):
                reward += 20.0  # Good: reported after confirmed execution
            else:
                reward -= 20.0  # Bad: gave up without finding anything

        # Penalty for repeating the same action 3+ times in a row
        recent_actions = [s.get("action_id") for s in self.episode_log[-3:]]
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            reward -= 5.0

        # Exploration bonus for first use of each action type
        if action_id not in self.agent_memory.get("tried_actions", set()):
            reward += 2.0

        return reward

    def _init_agent_memory(self) -> dict:
        """Extend base memory with CMDi-specific fields."""
        memory = super()._init_agent_memory()
        memory["found_command_exec"] = False
        memory["found_file_operations"] = False
        return memory

    def _update_agent_memory(self, action_id: int, analysis: AnalysisResult):
        """Update persistent agent state after each step."""
        super()._update_agent_memory(action_id, analysis)
        if analysis.has_command_execution:
            self.agent_memory["found_command_exec"] = True
        if analysis.file_operations_detected:
            self.agent_memory["found_file_operations"] = True

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        """Extract observation from current state."""
        return extract_unified_cmdi_state(
            html, analysis, self.agent_memory, response_format="html"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when command execution is confirmed."""
        return analysis.has_command_execution

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
