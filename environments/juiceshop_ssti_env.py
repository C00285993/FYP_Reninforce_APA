"""
Juice Shop SSTI (Server-Side Template Injection) Environment
Gymnasium environment for training an RL agent to discover and exploit
Server-Side Template Injection via Pug template injection on the /profile page.

Attack surface: POST /profile  { username: <payload> }, then GET /profile
Target challenge: "SSTi" (sstiChallenge) — Node.js eval() triggered by #{...} in username.
Success: expression evaluated (#{7*7}→49) OR RCE confirmed.

Action space: 10 discrete actions (payload families + meta actions)
Observation space: 18-dimensional float vector (UNIFIED_SSTI_STATE_SIZE)

WAF note
--------
Juice Shop blocks payloads containing 'this.constructor', 'require', 'process'
as literal strings. The agent must learn WAF bypass techniques:
  - Hex encoding:   global['\\x70\\x72\\x6f\\x63\\x65\\x73\\x73'].version
  - Concat/join:    global[['pro','cess'].join('')].version
  - Template literal: global[`proc${'ess'}`].version
"""

import json
import random
import logging
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from gymnasium import spaces

from environments.base_env import BasePentestEnv
from environments.feature_extractors import UNIFIED_SSTI_STATE_SIZE, extract_unified_ssti_state
from utils.response_analyzer import AnalysisResult
from utils.juiceshop_client import JuiceShopClient

logger = logging.getLogger(__name__)


class JuiceShopSSTIEnv(BasePentestEnv):
    """
    RL Environment for SSTI testing on OWASP Juice Shop.

    The agent targets the Pug template engine on the profile page:
      1. POST /profile  multipart username=<payload>
      2. GET  /profile  — read what Pug rendered (expression eval evidence)
      3. check_new_solves() detects sstiChallenge → success

    Actions:
        0:  submit_baseline         - Submit a benign username
        1:  probe_arithmetic        - #{7*7} — confirm Pug eval
        2:  probe_string            - String operation probes
        3:  inject_global           - global.process object access
        4:  inject_env              - process.env exfiltration
        5:  inject_waf_encoding     - Hex/base64 WAF bypass
        6:  inject_waf_concat       - Array join / concat WAF bypass
        7:  inject_rce_direct       - require('child_process') — may be WAF-blocked
        8:  inject_rce_indirect     - mainModule.require bypass chain
        9:  report_done             - Agent declares finished

    Rewards:
        -1.0  per step (efficiency penalty)
        +15   expression evaluated (e.g. 49 found after #{7*7})
        +30   Node.js info leaked (process.version / env path in response)
        +100  sstiChallenge solved on scoreboard
        -20   report_done without finding anything
        -5    repeating same action 3+ times in a row
        +2    exploration bonus (first use of each action)
    """

    # Challenge display names as returned by /api/Challenges/ .name field
    # Actual API name is "SSTi"; keep "Server-side Template Injection" as alias.
    SSTI_CHALLENGE_NAMES = {
        "SSTi",
        "Server-side Template Injection",
    }

    ACTIONS = {
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

    ACTION_TO_FAMILY = {
        0: None,                   # baseline
        1: "arithmetic_probe",
        2: "string_ops",
        3: "global_object",
        4: "env_access",
        5: "waf_bypass_encoding",
        6: "waf_bypass_concat",
        7: "rce_direct",
        8: "rce_indirect",
        9: None,                   # meta action
    }

    # ------------------------------------------------------------------ #
    #  Client & identity
    # ------------------------------------------------------------------ #

    def _create_client(self):
        """Use JuiceShopClient instead of DVWAClient."""
        return JuiceShopClient(base_url=self.dvwa_url)

    def _vuln_type(self) -> str:
        return "juiceshop_ssti"

    # ------------------------------------------------------------------ #
    #  Spaces
    # ------------------------------------------------------------------ #

    def _define_action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.ACTIONS))

    def _define_observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(UNIFIED_SSTI_STATE_SIZE,),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ #
    #  Payload loading
    # ------------------------------------------------------------------ #

    def _load_payloads(self) -> dict:
        payload_file = Path(__file__).parent.parent / "payloads" / "ssti_payloads.json"
        with open(payload_file) as f:
            data = json.load(f)
        return data["payload_families"]

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def _get_initial_page(self) -> str:
        """Fetch the rendered profile page as the initial observation."""
        html, _ = self.client.get_page("juiceshop_ssti")
        return html

    def _capture_baselines(self):
        """Set a benign username and record the rendered profile as baseline."""
        self.client.update_username("PentestAgent")
        html, status, _ = self.client.get_profile()
        self.analyzer.set_baseline("juiceshop_ssti", html)
        self._baseline_html = html

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute one SSTI step: POST username payload, GET profile, check scoreboard."""
        action_name = self.ACTIONS[action_id]
        url_path = "/profile"

        if action_id == 0:
            # Baseline: reset to a benign username
            self.client.update_username("PentestAgent")
            html, status, elapsed = self.client.get_profile()
            new_solves = self.client.check_new_solves()
            analysis = self.analyzer.analyze_ssti_response(
                html, "PentestAgent", status, elapsed, challenge_solves=new_solves
            )
            payload_info = self._build_payload_info(
                "PentestAgent", "username", url_path, html, analysis
            )
            return html, analysis, payload_info

        elif action_id == 9:
            # report_done — no HTTP request, just signal termination
            analysis = AnalysisResult()
            analysis.has_command_execution = self.agent_memory.get("found_rce", False)
            payload_info = {
                "payload": "",
                "parameter": "username",
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
                payload = "#{7*7}"  # safe fallback

            # POST the payload as the username, then read the rendered profile
            self.client.update_username(payload)
            html, status, elapsed = self.client.get_profile()
            new_solves = self.client.check_new_solves()

            analysis = self.analyzer.analyze_ssti_response(
                html, payload, status, elapsed, challenge_solves=new_solves
            )

            logger.debug(
                f"SSTI action={action_name!r} payload={payload!r} "
                f"eval={analysis.expression_evaluated} "
                f"nodejs={analysis.nodejs_info_leaked} "
                f"rce={analysis.has_command_execution} "
                f"waf={analysis.waf_blocked} "
                f"new_solves={new_solves}"
            )

            payload_info = self._build_payload_info(
                payload, "username", url_path, html, analysis
            )
            return html, analysis, payload_info

    def _build_payload_info(
        self,
        payload: str,
        parameter: str,
        url_path: str,
        html: str,
        analysis: AnalysisResult,
    ) -> dict:
        """Build the payload_info dict for this SSTI step."""
        full_url = (
            f"{self.dvwa_url}{url_path} "
            f"[POST username={payload[:60]}]"
        )
        soup = BeautifulSoup(html, "lxml")
        snippet = soup.get_text(separator=" ", strip=True)[:500]
        reflected = payload.lower() in html.lower()
        return {
            "payload": payload,
            "parameter": parameter,
            "url_path": url_path,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    # ------------------------------------------------------------------ #
    #  Reward
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        """
        Tiered reward structure encouraging escalating SSTI exploitation evidence.

        Rewards fire once per discovery level (tracked in agent_memory) so the
        agent isn't incentivised to repeat the same successful payload family.
        """
        reward = -1.0  # Step penalty

        # One-time reward: template expression evaluated
        if analysis.expression_evaluated and not self.agent_memory.get("_rewarded_eval"):
            reward += 15.0
            self.agent_memory["_rewarded_eval"] = True
            logger.debug("  +15: Pug expression evaluated (arithmetic/string result confirmed)")

        # One-time reward: Node.js info leaked
        if analysis.nodejs_info_leaked and not self.agent_memory.get("_rewarded_nodejs"):
            reward += 30.0
            self.agent_memory["_rewarded_nodejs"] = True
            logger.debug("  +30: Node.js info leaked (version / env / platform)")

        # Large reward: RCE confirmed / sstiChallenge solved
        if analysis.has_command_execution and not self.agent_memory.get("found_rce"):
            reward += 100.0
            logger.debug("  +100: RCE / sstiChallenge solved!")

        # Small penalty: WAF blocked (first time only — penalise blocked family choice)
        if analysis.waf_blocked and not self.agent_memory.get("_saw_waf"):
            reward -= 5.0
            self.agent_memory["_saw_waf"] = True
            logger.debug("  -5: WAF block detected (first time)")

        # report_done action
        if action_id == 9:
            if self.agent_memory.get("found_rce"):
                reward += 20.0
            else:
                reward -= 20.0

        # Repeat-action penalty (same action 3+ times in a row)
        recent_actions = [s.get("action_id") for s in self.episode_log[-3:]]
        if len(recent_actions) >= 3 and len(set(recent_actions)) == 1:
            reward -= 5.0

        # Exploration bonus: first use of each action type
        if action_id not in self.agent_memory.get("tried_actions", set()):
            reward += 2.0

        return reward

    # ------------------------------------------------------------------ #
    #  Agent memory
    # ------------------------------------------------------------------ #

    def _init_agent_memory(self) -> dict:
        memory = super()._init_agent_memory()
        memory["found_eval"] = False
        memory["found_rce"] = False
        memory["_rewarded_eval"] = False
        memory["_rewarded_nodejs"] = False
        memory["_saw_waf"] = False
        return memory

    def _update_agent_memory(self, action_id: int, analysis: AnalysisResult):
        super()._update_agent_memory(action_id, analysis)
        if analysis.expression_evaluated:
            self.agent_memory["found_eval"] = True
        if analysis.has_command_execution:
            self.agent_memory["found_rce"] = True

    # ------------------------------------------------------------------ #
    #  State extraction & termination
    # ------------------------------------------------------------------ #

    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        return extract_unified_ssti_state(html, analysis, self.agent_memory)

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when template expression is evaluated OR RCE confirmed.

        The Juice Shop sstiChallenge is solved as soon as ANY #{...} expression is
        eval'd server-side (via Node.js eval() in userProfile.js).  Arithmetic probes
        like #{7*7} are sufficient — RCE is not required.
        """
        return analysis.expression_evaluated or analysis.has_command_execution

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
