"""
LLM-Guided Training Callback

Monitors RL training progress and calls Claude Haiku when the agent plateaus.
Detects stagnation via a rolling reward window and asks the LLM which attack
family to explore next, logging the suggestion for the researcher.

Usage:
    from agents.llm_guided_callback import LLMGuidedCallback

    llm_cb = LLMGuidedCallback(
        api_key="sk-ant-...",
        vuln_type="sqli",
        action_names=["submit_baseline", "inject_or_true", ...],
    )
    callbacks = CallbackList([checkpoint_cb, metrics_cb, llm_cb])
    model.learn(total_timesteps=50000, callback=callbacks)
"""

import logging
import os
from collections import Counter, deque
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

try:
    from utils.api_error_handler import handle_api_error
except ImportError:
    handle_api_error = None

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"

_SYSTEM_PROMPT = """\
You are advising a Reinforcement Learning agent that is testing web applications
for security vulnerabilities in an authorized academic lab environment.
The agent has stalled and needs guidance on which attack family to focus on next.
Reply with ONLY the exact action name from the list provided — nothing else.
"""


class LLMGuidedCallback(BaseCallback):
    """
    Detects training plateaus and asks Claude Haiku for the next attack family.

    A plateau is declared when the mean episode reward over the last
    `plateau_window` episodes hasn't changed by more than `plateau_threshold`
    compared to the preceding window of the same size.

    On plateau detection the callback:
      1. Summarises the recent action distribution and reward history
      2. Calls Claude Haiku with the available action names
      3. Logs the suggestion at INFO level
      4. Stores it in `self.last_suggestion` for inspection

    The callback does NOT force the agent to take the suggested action —
    it acts as an advisory signal to the researcher watching training.
    A `cooldown_episodes` guard prevents spamming the API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        vuln_type: str = "sqli",
        action_names: Optional[list] = None,
        plateau_window: int = 20,
        plateau_threshold: float = 3.0,
        cooldown_episodes: int = 15,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.vuln_type = vuln_type
        self.action_names = action_names or []
        if not self.action_names:
            logger.warning("LLMGuidedCallback: action_names is empty — LLM suggestions will be disabled")
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.cooldown_episodes = cooldown_episodes

        # Ring buffer — keeps 2× window so we can compare two halves
        self._episode_rewards: deque = deque(maxlen=plateau_window * 2)
        self._recent_actions: deque = deque(maxlen=100)
        self._total_episodes = 0
        self._episodes_since_suggestion = cooldown_episodes  # allow check immediately

        self.last_suggestion: Optional[str] = None
        self.suggestion_history: list = []

        self._client = None
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
                if self.verbose >= 1:
                    logger.info("LLMGuidedCallback: Claude Haiku advisor active")
            except Exception as e:
                logger.warning("LLMGuidedCallback: Claude unavailable — %s", e)
        else:
            logger.info("LLMGuidedCallback: no API key — running in passive mode (no LLM calls)")

    # ------------------------------------------------------------------
    # SB3 callback interface
    # ------------------------------------------------------------------

    def _on_step(self) -> bool:
        # SB3 Monitor sets info["episode"] when an episode ends
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        ep_info = info.get("episode")

        if ep_info is not None:
            self._total_episodes += 1
            self._episodes_since_suggestion += 1
            self._episode_rewards.append(float(ep_info.get("r", 0.0)))

        # Track the action that was just taken
        actions = self.locals.get("actions")
        if actions is not None:
            action_id = int(actions[0]) if hasattr(actions, "__len__") else int(actions)
            name = (
                self.action_names[action_id]
                if self.action_names and action_id < len(self.action_names)
                else str(action_id)
            )
            self._recent_actions.append(name)

        # Check for plateau every episode
        if (
            self._client is not None
            and ep_info is not None
            and self._episodes_since_suggestion >= self.cooldown_episodes
            and len(self._episode_rewards) >= self.plateau_window
            and self._is_plateau()
        ):
            suggestion = self._consult_llm()
            if suggestion:
                self.last_suggestion = suggestion
                self.suggestion_history.append({
                    "episode": self._total_episodes,
                    "suggestion": suggestion,
                    "mean_reward": float(np.mean(list(self._episode_rewards)[-self.plateau_window:])),
                })
                self._episodes_since_suggestion = 0
                logger.info(
                    f"\n{'='*60}\n"
                    f"[LLM Advisor] Plateau at episode {self._total_episodes} "
                    f"(mean reward {np.mean(list(self._episode_rewards)):.1f})\n"
                    f"  Suggestion: try '{suggestion}' attack family\n"
                    f"{'='*60}\n"
                )

        return True

    # ------------------------------------------------------------------
    # Plateau detection
    # ------------------------------------------------------------------

    def _is_plateau(self) -> bool:
        rewards = list(self._episode_rewards)
        half = len(rewards) // 2
        if half < 3:
            return False
        older_mean = float(np.mean(rewards[:half]))
        newer_mean = float(np.mean(rewards[half:]))
        return abs(newer_mean - older_mean) < self.plateau_threshold

    # ------------------------------------------------------------------
    # LLM consultation
    # ------------------------------------------------------------------

    def _consult_llm(self) -> Optional[str]:
        if not self.action_names:
            return None
        rewards = list(self._episode_rewards)
        action_freq = Counter(list(self._recent_actions))
        top_used = action_freq.most_common(5)

        action_summary = "\n".join(f"  - {a}: {c} times" for a, c in top_used)
        available = "\n".join(f"  - {a}" for a in self.action_names)

        prompt = (
            f"An RL agent testing {self.vuln_type.upper()} vulnerabilities has plateaued.\n\n"
            f"Recent episode statistics ({len(rewards)} episodes):\n"
            f"  mean reward = {np.mean(rewards):.1f}, std = {np.std(rewards):.1f}\n\n"
            f"Most-used attack families:\n{action_summary}\n\n"
            f"Available attack families:\n{available}\n\n"
            f"Which single attack family should the agent focus on more "
            f"to break out of the plateau? "
            f"Output ONLY the exact action name as listed above."
        )

        try:
            response = self._client.messages.create(
                model=_MODEL,
                max_tokens=50,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip().lower().strip("'\"")

            # Exact match first
            if raw in [a.lower() for a in self.action_names]:
                for a in self.action_names:
                    if a.lower() == raw:
                        return a

            # Fuzzy: substring match
            for a in self.action_names:
                if a.lower() in raw or raw in a.lower():
                    return a

            logger.debug(f"LLMGuidedCallback: unrecognised suggestion '{raw}'")
            return raw  # Return as-is for logging even if not matched

        except Exception as e:
            if handle_api_error is not None:
                handle_api_error(e, logger, context="LLM guided callback", once_flag_obj=self)
            else:
                logger.warning("LLM consultation failed: %s", e)
            return None
