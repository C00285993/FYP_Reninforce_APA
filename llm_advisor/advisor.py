"""
LLM Advisor (Phase 3 - Optional Extension)
Uses a local LLM (via Ollama) to provide strategic advice when the RL agent
is uncertain about which action to take.

Requirements:
    - Ollama installed: https://ollama.ai
    - Model pulled: ollama pull mistral

Usage:
    advisor = LLMAdvisor()
    suggestion = advisor.get_advice(observation_summary, available_actions)
"""

import requests
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMAdvisor:
    """
    LLM-based strategic advisor for the pentest agent.
    Runs locally via Ollama — no data leaves the machine.
    """

    def __init__(self, model: str = "mistral",
                 ollama_url: str = "http://localhost:11434"):
        self.model = model
        self.ollama_url = ollama_url.rstrip("/")
        self._available = None

        # Load system prompt
        prompt_file = Path(__file__).parent / "prompts" / "pentest_system.md"
        if prompt_file.exists():
            self.system_prompt = prompt_file.read_text()
        else:
            self.system_prompt = self._default_system_prompt()

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                self._available = any(self.model in m for m in models)
                if not self._available:
                    logger.warning(
                        f"Ollama running but model '{self.model}' not found. "
                        f"Pull it with: ollama pull {self.model}"
                    )
            else:
                self._available = False
        except requests.ConnectionError:
            logger.warning("Ollama not running. Start with: ollama serve")
            self._available = False
        return self._available

    def get_advice(self, observation: str, available_actions: list[str],
                   history: str = "") -> str | None:
        """
        Ask the LLM for a strategic recommendation.

        Args:
            observation: Description of current state
            available_actions: List of action names the agent can take
            history: Summary of recent actions and results

        Returns:
            Recommended action name, or None if LLM unavailable
        """
        if not self.is_available():
            return None

        actions_str = "\n".join(f"  - {a}" for a in available_actions)

        prompt = f"""Current observation:
{observation}

Recent history:
{history if history else "No actions taken yet."}

Available actions:
{actions_str}

Based on this information, which SINGLE action should the agent take next?
Reply with ONLY the action name, nothing else."""

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "system": self.system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 50,
                    }
                },
                timeout=30,
            )

            if resp.status_code == 200:
                response_text = resp.json().get("response", "").strip()
                # Try to match response to an available action
                for action in available_actions:
                    if action.lower() in response_text.lower():
                        logger.info(f"LLM recommends: {action}")
                        return action

                logger.debug(f"LLM response didn't match actions: {response_text}")
                return None
            else:
                logger.warning(f"Ollama returned status {resp.status_code}")
                return None

        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="LLM advisor", once_flag_obj=self)
            return None

    def analyze_response(self, html_snippet: str, payload: str) -> str:
        """
        Ask the LLM to interpret a response (is this a vulnerability?).

        Args:
            html_snippet: Relevant portion of the HTTP response
            payload: The payload that was sent

        Returns:
            LLM's analysis as a string
        """
        if not self.is_available():
            return "LLM unavailable"

        prompt = f"""Analyze this web application response.

Payload sent: {payload}

Response snippet (first 500 chars):
{html_snippet[:500]}

Questions:
1. Does this response indicate a vulnerability? (yes/no)
2. What type of vulnerability? (SQLi, XSS, none)
3. What should be tried next?

Be concise (3 lines max)."""

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "system": self.system_prompt,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 150},
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
        except Exception as e:
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="LLM analysis", once_flag_obj=self,
                             once_flag_attr="_analysis_error_shown")

        return "Analysis unavailable"

    @staticmethod
    def _default_system_prompt() -> str:
        return """You are a penetration testing advisor for an educational AI security project.
You assist an RL agent in testing DELIBERATELY VULNERABLE web applications (DVWA)
in an isolated lab environment. This is an authorized academic exercise.

Your role:
- Suggest which action to try next based on observations
- Interpret HTTP responses for evidence of vulnerabilities
- Focus on SQL injection and Cross-Site Scripting (XSS)
- Be concise and actionable

Always recommend only from the available action list provided."""
