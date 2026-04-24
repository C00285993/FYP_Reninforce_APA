"""
Base Pentest Environment
Abstract Gymnasium environment that provides shared logic for all
vulnerability-specific environments (SQLi, XSS, etc.).

Handles: episode management, step counting, logging, DVWA client lifecycle.
Subclasses implement: action execution, reward calculation, state extraction.
"""

import random
from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import logging
from abc import abstractmethod
from pathlib import Path
from datetime import datetime

from utils.dvwa_client import DVWAClient
from utils.response_analyzer import ResponseAnalyzer, AnalysisResult
from utils.llm_success_detector import LLMSuccessDetector
from utils.llm_payload_generator import LLMPayloadGenerator
from environments.feature_extractors import FeatureExtractor

logger = logging.getLogger(__name__)


class BasePentestEnv(gym.Env):
    """
    Abstract base class for pentest Gymnasium environments.

    Subclasses must implement:
        - _define_action_space() -> spaces.Discrete
        - _execute_action(action_id) -> (html, analysis)
        - _calculate_reward(action_id, analysis) -> float
        - _extract_state(html, analysis) -> np.ndarray
        - _is_success(analysis) -> bool
        - _get_action_name(action_id) -> str
    """

    metadata = {"render_modes": ["human", "log"]}

    def __init__(
        self,
        dvwa_url: str = "http://localhost:8080",
        security_level: str = "low",
        max_steps: int = 50,
        log_dir: str = "./logs",
        render_mode: str = "log",
        api_key: Optional[str] = None,
    ):
        super().__init__()

        self.dvwa_url = dvwa_url
        self.security_level = security_level
        self.max_steps = max_steps
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.render_mode = render_mode

        # Core components (subclasses can override _create_client)
        self.client = self._create_client()
        self.analyzer = ResponseAnalyzer()
        self.extractor = FeatureExtractor()

        # LLM components (both degrade gracefully when api_key is absent)
        self.detector = LLMSuccessDetector(api_key=api_key)
        self._payload_gen = LLMPayloadGenerator(api_key=api_key)
        if api_key and self._payload_gen.available:
            logger.info(f"{self.__class__.__name__}: LLM payload generation enabled")

        # Tracks last HTTP interaction for LLM context
        self._last_payload: str = ""
        self._last_response: str = ""
        self._last_status: int = 200

        # Subclass defines these
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

        # Episode state
        self.current_step = 0
        self.episode_count = 0
        self.agent_memory = {}
        self.episode_log = []
        self._episode_reward = 0.0
        self._dvwa_initialized = False

        # Payload tracking
        self.payloads = self._load_payloads()

    def _create_client(self):
        """Create the HTTP client. Override in subclasses for non-DVWA targets."""
        return DVWAClient(base_url=self.dvwa_url)

    def _ensure_dvwa_ready(self):
        """Lazy initialization of DVWA connection (first episode only)."""
        if not self._dvwa_initialized:
            logger.info("Initializing DVWA connection...")
            success = self.client.ensure_ready(self.security_level)
            if not success:
                raise ConnectionError(
                    f"Cannot connect to DVWA at {self.dvwa_url}. "
                    "Is Docker running? Try: docker compose up -d"
                )
            # Capture baseline responses
            self._capture_baselines()
            self._dvwa_initialized = True

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Returns:
            observation: Initial state vector
            info: Dict with episode metadata
        """
        super().reset(seed=seed)
        self._ensure_dvwa_ready()

        # Reset episode state
        self.current_step = 0
        self.episode_count += 1
        self._episode_reward = 0.0
        self.episode_log = []
        self.agent_memory = self._init_agent_memory()

        # Reset DVWA session (lightweight)
        self.client.reset_for_episode()

        # Get initial observation
        initial_html = self._get_initial_page()
        initial_analysis = AnalysisResult()  # Empty analysis at start
        observation = self._extract_state(initial_html, initial_analysis)

        info = {
            "episode": self.episode_count,
            "security_level": self.security_level,
            "vuln_type": self._vuln_type(),
        }

        if self.render_mode == "human":
            self._render_human(f"=== Episode {self.episode_count} START ===")

        return observation, info

    def step(self, action: int):
        """
        Execute one action in the environment.

        Args:
            action: Integer action ID from the action space.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1

        # Execute the action (subclass handles specifics)
        response_html, analysis, payload_info = self._execute_action(action)

        # Calculate reward
        reward = self._calculate_reward(action, analysis)
        self._episode_reward += reward

        # Update agent memory
        self._update_agent_memory(action, analysis)

        # Extract new state
        observation = self._extract_state(response_html, analysis)

        # Check termination conditions
        terminated = self._is_success(analysis)  # Goal reached
        truncated = self.current_step >= self.max_steps  # Time limit

        # Log this step
        action_name = self._get_action_name(action)
        step_log = {
            "step": self.current_step,
            "action": action_name,
            "action_id": int(action),
            "reward": reward,
            "cumulative_reward": self._episode_reward,
            "severity_score": analysis.severity_score,
            "terminated": terminated,
            "truncated": truncated,
            "payload": payload_info.get("payload", ""),
            "parameter": payload_info.get("parameter", ""),
            "url_path": payload_info.get("url_path", ""),
            "full_request_url": payload_info.get("full_request_url", ""),
            "response_snippet": payload_info.get("response_snippet", ""),
            "reflected": payload_info.get("reflected", False),
        }
        self.episode_log.append(step_log)

        if self.render_mode == "human":
            self._render_human(
                f"  Step {self.current_step}: {action_name} -> "
                f"reward={reward:+.1f} severity={analysis.severity_score}"
            )

        # On episode end, save log
        if terminated or truncated:
            self._save_episode_log(terminated)
            if self.render_mode == "human":
                status = "SUCCESS" if terminated else "TIMEOUT"
                self._render_human(
                    f"=== Episode {self.episode_count} END ({status}) "
                    f"Total Reward: {self._episode_reward:.1f} ===\n"
                )

        info = {
            "step": self.current_step,
            "action_name": action_name,
            "severity_score": analysis.severity_score,
            "episode_reward": self._episode_reward,
            "payload_info": payload_info,
        }

        return observation, reward, terminated, truncated, info

    def _save_episode_log(self, success: bool):
        """Save structured episode log as JSON."""
        log_entry = {
            "episode": self.episode_count,
            "vuln_type": self._vuln_type(),
            "security_level": self.security_level,
            "total_steps": self.current_step,
            "total_reward": self._episode_reward,
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "steps": self.episode_log,
        }

        log_file = self.log_dir / f"{self._vuln_type()}_episodes.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _render_human(self, message: str):
        """Print human-readable output."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self._vuln_type().upper()}] {message}")

    def _init_agent_memory(self) -> dict:
        """Initialize agent memory for a new episode."""
        return {
            "total_attempts": 0,
            "attempts_this_field": 0,
            "last_payload_category": 0,
            "unique_responses_seen": 0,
            "found_sql_error": False,
            "found_data_leak": False,
            "found_reflection": False,
            "found_xss_execution": False,
            "response_hashes": set(),
            "tried_actions": set(),
        }

    def _update_agent_memory(self, action: int, analysis: AnalysisResult):
        """Update agent memory after an action."""
        self.agent_memory["total_attempts"] += 1
        self.agent_memory["attempts_this_field"] += 1
        self.agent_memory["last_payload_category"] = action
        self.agent_memory["tried_actions"].add(action)

        # Track unique responses
        resp_hash = hash(str(analysis.response_length) + str(analysis.severity_score))
        if resp_hash not in self.agent_memory["response_hashes"]:
            self.agent_memory["unique_responses_seen"] += 1
            self.agent_memory["response_hashes"].add(resp_hash)

        # Track findings
        if analysis.has_sql_error:
            self.agent_memory["found_sql_error"] = True
        if analysis.has_data_leak:
            self.agent_memory["found_data_leak"] = True
        if analysis.payload_reflected:
            self.agent_memory["found_reflection"] = True
        if analysis.script_tag_present or analysis.event_handler_present:
            self.agent_memory["found_xss_execution"] = True

    def set_security_level(self, level: str):
        """Change DVWA security level (for curriculum learning)."""
        self.security_level = level
        if self._dvwa_initialized:
            self.client.set_security_level(level)
            self._capture_baselines()

    # ------------------------------------------------------------------
    # LLM helper methods (used by subclass _execute_action / _is_success)
    # ------------------------------------------------------------------

    def _pick_payload(
        self,
        family: Optional[str],
        fallback_payloads: list,
        context: Optional[dict] = None,
    ) -> str:
        """
        Select a payload for the given attack family.

        Uses LLMPayloadGenerator when an API key was provided (adapts the
        payload to observed target behaviour). Falls back to random.choice
        from the static payload list when the LLM is unavailable.

        Args:
            family:            Attack family name (e.g. "or_true").
            fallback_payloads: Static payload list for this family.
            context:           Optional runtime context dict with keys:
                               target_url, parameter, last_payload,
                               last_response_snippet, observed.

        Returns:
            A payload string.
        """
        if not fallback_payloads:
            return "test"
        if not self._payload_gen.available:
            return random.choice(fallback_payloads)

        ctx = context or {
            "target_url": self.dvwa_url,
            "last_payload": self._last_payload,
            "last_response_snippet": self._last_response[:300],
            "observed": {
                "reflected":   self.agent_memory.get("found_reflection", False),
                "sql_error":   self.agent_memory.get("found_sql_error", False),
                "data_leaked": self.agent_memory.get("found_data_leak", False),
                "blocked": (
                    self.agent_memory.get("total_attempts", 0) > 0
                    and not self.agent_memory.get("found_reflection", False)
                    and not self.agent_memory.get("found_sql_error", False)
                ),
            },
        }
        return self._payload_gen.generate(
            family=family,
            vuln_type=self._vuln_type(),
            fallback_payloads=fallback_payloads,
            context=ctx,
        )

    def _llm_confirm_success(
        self,
        analysis: AnalysisResult,
    ) -> bool:
        """
        Ask the LLM Success Detector to confirm whether the attack succeeded.

        Only calls the API when the heuristic severity score is above the
        detector's threshold (controlled inside LLMSuccessDetector).
        Returns the heuristic result when LLM is unavailable.

        Args:
            analysis: AnalysisResult from the most recent step.

        Returns:
            True if the LLM (or heuristic fallback) confirms success.
        """
        vuln = self._vuln_type()
        heuristic = (
            analysis.has_data_leak
            or analysis.auth_bypass
            or analysis.script_tag_present
            or analysis.event_handler_present
            or analysis.has_command_execution
        )
        if not heuristic:
            return False

        if self.detector.should_consult_llm(vuln, analysis.severity_score):
            verdict = self.detector.detect(
                vuln_type=vuln,
                payload=self._last_payload,
                response_body=self._last_response,
                status_code=self._last_status,
            )
            if verdict["success"]:
                logger.info(
                    f"[LLM] Confirmed {vuln.upper()} success "
                    f"(confidence={verdict['confidence']:.0%}): {verdict['evidence']}"
                )
            return verdict["success"]

        return heuristic

    # ---- Abstract methods (subclasses must implement) ----

    @abstractmethod
    def _define_action_space(self) -> spaces.Discrete:
        """Define the discrete action space."""
        ...

    @abstractmethod
    def _define_observation_space(self) -> spaces.Box:
        """Define the observation space."""
        ...

    @abstractmethod
    def _execute_action(self, action_id: int) -> tuple[str, AnalysisResult, dict]:
        """Execute an action and return (html, analysis, payload_info).

        payload_info is a dict with keys:
            payload, parameter, url_path, full_request_url,
            response_snippet, reflected
        """
        ...

    @abstractmethod
    def _calculate_reward(self, action_id: int, analysis: AnalysisResult) -> float:
        """Calculate reward for this step."""
        ...

    @abstractmethod
    def _extract_state(self, html: str, analysis: AnalysisResult) -> np.ndarray:
        """Convert current state to observation vector."""
        ...

    @abstractmethod
    def _is_success(self, analysis: AnalysisResult) -> bool:
        """Check if the vulnerability was successfully exploited."""
        ...

    @abstractmethod
    def _get_action_name(self, action_id: int) -> str:
        """Get human-readable name for an action."""
        ...

    @abstractmethod
    def _vuln_type(self) -> str:
        """Return the vulnerability type identifier."""
        ...

    @abstractmethod
    def _load_payloads(self) -> dict:
        """Load payload definitions."""
        ...

    @abstractmethod
    def _get_initial_page(self) -> str:
        """Fetch the initial page for this vulnerability module."""
        ...

    @abstractmethod
    def _capture_baselines(self):
        """Capture baseline responses for comparison."""
        ...
