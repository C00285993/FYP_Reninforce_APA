"""
Logger
Configures structured logging for both training metrics and episode details.
Provides a custom SB3 callback for TensorBoard logging of pentest-specific metrics.
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime

from stable_baselines3.common.callbacks import BaseCallback


def setup_logging(log_dir: str = "./logs", level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the pentest assistant.

    Args:
        log_dir: Directory for log files
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Root logger configured with console and file handlers
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))

    # Console handler (concise)
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(console)

    # File handler (detailed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_dir / f"pentest_{timestamp}.log"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s:%(lineno)d: %(message)s"
    ))
    logger.addHandler(file_handler)

    return logger


class PentestMetricsCallback(BaseCallback):
    """
    Custom Stable-Baselines3 callback that logs pentest-specific metrics
    to TensorBoard during training.

    Tracks:
        - Success rate (rolling window)
        - Average episode length for successes vs failures
        - Unique actions used per episode
        - Vulnerability detection milestones
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        """
        Args:
            log_freq: Log metrics every N steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self._current_episode_reward = 0.0
        self._current_episode_length = 0

    def _on_step(self) -> bool:
        """Called at every step."""
        # Accumulate episode stats
        self._current_episode_reward += self.locals.get("rewards", [0])[0]
        self._current_episode_length += 1

        # Check for episode end
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])

        if dones[0]:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self._current_episode_length)

            # Check if episode was successful.
            # SB3's DummyVecEnv sets info["TimeLimit.truncated"]=True when
            # the episode hit the step limit instead of a natural terminal
            # state.  Natural termination means _is_success() returned True,
            # i.e. the agent actually exploited the vulnerability.
            # Using severity_score as a proxy breaks for Playwright-detected
            # DOM XSS where the HTTP response has no reflected payload
            # (severity=30 exactly, but the threshold was > 30).
            info = infos[0] if infos else {}
            truncated = info.get("TimeLimit.truncated", False)
            success = dones[0] and not truncated
            self.episode_successes.append(float(success))

            # Reset counters
            self._current_episode_reward = 0.0
            self._current_episode_length = 0

        # Log periodically
        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            window = min(50, len(self.episode_rewards))
            recent_rewards = self.episode_rewards[-window:]
            recent_successes = self.episode_successes[-window:]
            recent_lengths = self.episode_lengths[-window:]

            self.logger.record("pentest/mean_reward",
                             sum(recent_rewards) / len(recent_rewards))
            self.logger.record("pentest/success_rate",
                             sum(recent_successes) / len(recent_successes))
            self.logger.record("pentest/mean_episode_length",
                             sum(recent_lengths) / len(recent_lengths))
            self.logger.record("pentest/total_episodes",
                             len(self.episode_rewards))

            if self.verbose > 0:
                sr = sum(recent_successes) / len(recent_successes)
                mr = sum(recent_rewards) / len(recent_rewards)
                print(f"  [Metrics] Episodes: {len(self.episode_rewards)} | "
                      f"Success Rate: {sr:.1%} | Mean Reward: {mr:.1f}")

        return True  # Continue training
