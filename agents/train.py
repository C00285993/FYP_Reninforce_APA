"""
Training Script
Trains RL agents (DQN or PPO) on vulnerability environments.

Usage:
    python -m agents.train --vuln sqli --algo dqn --timesteps 50000
    python -m agents.train --vuln xss --algo ppo --timesteps 50000
    python -m agents.train --vuln sqli --algo dqn --security low --timesteps 50000
    python -m agents.train --vuln both --algo ppo --timesteps 50000

Multi-target transfer learning (trains sequentially, carrying model forward):
    python -m agents.train --targets dvwa juiceshop --vuln sqli --algo dqn --timesteps 50000

Resume from checkpoint (picks up latest checkpoint in the directory):
    python -m agents.train --vuln sqli --algo dqn --security medium \
        --resume models/sqli_dqn_medium_20260214_112115 --timesteps 50000

Curriculum training (progressive difficulty):
    python -m agents.train --vuln sqli --algo dqn --curriculum

Curriculum resume (skip low, start from existing model at medium):
    python -m agents.train --vuln xss --algo dqn --timesteps 100000 \
        --curriculum --curriculum-start medium \
        --resume models/xss_dqn_low_20260216_103148
"""

import argparse
import re
import sys
import os
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from environments.sqli_env import SQLiEnv
from environments.xss_env import XSSEnv
from environments.juiceshop_sqli_env import JuiceShopSQLiEnv
from environments.juiceshop_xss_env import JuiceShopXSSEnv
from environments.cmdi_env import CMDiEnv
from environments.juiceshop_ssti_env import JuiceShopSSTIEnv
from environments.dvwa_ssti_env import DVWASSTIEnv
from environments.webgoat_cmdi_env import WebGoatCMDiEnv
from environments.webgoat_sqli_env import WebGoatSQLiEnv
from environments.webgoat_xss_env import WebGoatXSSEnv
from environments.dvwa_stored_xss_env import DVWAStoredXSSEnv
from utils.logger import setup_logging, PentestMetricsCallback
from agents.llm_guided_callback import LLMGuidedCallback

import logging

logger = logging.getLogger(__name__)


def _resolve_target_url(base_url: str, target: str) -> str:
    """Map target name to its default port when using the DVWA default URL."""
    if base_url == "http://localhost:8080":
        return {
            "juiceshop": "http://localhost:3000",
            "juiceshop_dom": "http://localhost:3000",
            "webgoat": "http://localhost:8081",
        }.get(target, base_url)
    return base_url


def create_environment(vuln_type: str, dvwa_url: str, security: str,
                       max_steps: int, log_dir: str, render_mode: str,
                       target: str = "dvwa", use_playwright: bool = False,
                       headless: bool = True, api_key: Optional[str] = None):
    """Create and wrap the appropriate Gym environment."""
    if target == "juiceshop":
        if vuln_type == "sqli":
            env = JuiceShopSQLiEnv(
                dvwa_url=dvwa_url,
                security_level=security,
                max_steps=max_steps,
                log_dir=log_dir,
                render_mode=render_mode,
            )
        elif vuln_type == "xss":
            env = JuiceShopXSSEnv(
                dvwa_url=dvwa_url,
                security_level=security,
                max_steps=max_steps,
                log_dir=log_dir,
                render_mode=render_mode,
                use_playwright=use_playwright,
                headless=headless,
            )
        elif vuln_type == "ssti":
            env = JuiceShopSSTIEnv(
                dvwa_url=dvwa_url,
                security_level=security,
                max_steps=max_steps,
                log_dir=log_dir,
                render_mode=render_mode,
            )
        else:
            raise ValueError(
                f"Juice Shop supports sqli, xss, and ssti, got: {vuln_type}"
            )
    elif target == "webgoat":
        if vuln_type == "sqli":
            env = WebGoatSQLiEnv(
                dvwa_url=dvwa_url,
                security_level=security,
                max_steps=max_steps,
                log_dir=log_dir,
                render_mode=render_mode,
            )
        elif vuln_type == "cmdi":
            env = WebGoatCMDiEnv(
                dvwa_url=dvwa_url,
                security_level=security,
                max_steps=max_steps,
                log_dir=log_dir,
                render_mode=render_mode,
            )
        elif vuln_type == "xss":
            env = WebGoatXSSEnv(
                dvwa_url=dvwa_url,
                security_level=security,
                max_steps=max_steps,
                log_dir=log_dir,
                render_mode=render_mode,
            )
        else:
            raise ValueError(
                f"WebGoat target supports sqli, cmdi, and xss, got: {vuln_type}"
            )
    elif vuln_type == "sqli":
        env = SQLiEnv(
            dvwa_url=dvwa_url,
            security_level=security,
            max_steps=max_steps,
            log_dir=log_dir,
            render_mode=render_mode,
            api_key=api_key,
        )
    elif target == "dvwa_stored" and vuln_type == "xss":
        env = DVWAStoredXSSEnv(
            dvwa_url=dvwa_url,
            security_level=security,
            max_steps=max_steps,
            log_dir=log_dir,
            render_mode=render_mode,
            api_key=api_key,
        )
    elif target == "juiceshop_dom" and vuln_type == "xss":
        env = JuiceShopXSSEnv(
            dvwa_url=dvwa_url,
            security_level=security,
            max_steps=max_steps,
            log_dir=log_dir,
            render_mode=render_mode,
            use_playwright=use_playwright,
            headless=headless,
        )
    elif vuln_type == "xss":
        env = XSSEnv(
            dvwa_url=dvwa_url,
            security_level=security,
            max_steps=max_steps,
            log_dir=log_dir,
            render_mode=render_mode,
            api_key=api_key,
        )
    elif vuln_type == "cmdi":
        env = CMDiEnv(
            dvwa_url=dvwa_url,
            security_level=security,
            max_steps=max_steps,
            log_dir=log_dir,
            render_mode=render_mode,
            api_key=api_key,
        )
    elif vuln_type == "ssti":
        env = DVWASSTIEnv(
            dvwa_url=dvwa_url,
            security_level=security,
            max_steps=max_steps,
            log_dir=log_dir,
            render_mode=render_mode,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Unknown vuln type: {vuln_type}")

    # Wrap with Monitor for SB3 compatibility
    monitor_name = f"{target}_{vuln_type}" if target != "dvwa" else vuln_type
    env = Monitor(env, filename=str(Path(log_dir) / f"{monitor_name}_monitor"))
    return env


def create_agent(algo: str, env, tensorboard_log: str, config: dict):
    """Create the RL agent with appropriate hyperparameters."""
    if algo == "dqn":
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.get("learning_rate", 1e-3),
            buffer_size=config.get("buffer_size", 50000),
            learning_starts=config.get("learning_starts", 500),
            batch_size=config.get("batch_size", 64),
            gamma=config.get("gamma", 0.99),
            exploration_fraction=config.get("exploration_fraction", 0.3),
            exploration_initial_eps=config.get("exploration_initial_eps", 1.0),
            exploration_final_eps=config.get("exploration_final_eps", 0.05),
            target_update_interval=config.get("target_update_interval", 250),
            train_freq=config.get("train_freq", 4),
            policy_kwargs=dict(
                net_arch=config.get("net_arch", [128, 128]),
            ),
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    elif algo == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.get("learning_rate", 3e-4),
            n_steps=config.get("n_steps", 256),
            batch_size=config.get("batch_size", 64),
            n_epochs=config.get("n_epochs", 10),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            ent_coef=config.get("ent_coef", 0.01),
            policy_kwargs=dict(
                net_arch=dict(
                    pi=config.get("net_arch", [128, 128]),
                    vf=config.get("net_arch", [128, 128]),
                ),
            ),
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    return model


def find_latest_checkpoint(model_dir: str, vuln_type: str, algo: str) -> tuple[str | None, int]:
    """Find the latest checkpoint in a model directory.

    Returns:
        Tuple of (checkpoint_path, timestep) or (None, 0) if no checkpoint found.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return None, 0

    # Look for checkpoint files like sqli_dqn_10000_steps.zip
    best_path = None
    best_step = 0

    for f in model_path.glob(f"{vuln_type}_{algo}_*_steps.zip"):
        match = re.search(r"_(\d+)_steps\.zip$", f.name)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = str(f).replace(".zip", "")  # SB3 expects path without .zip

    # Also check for a final model
    final = model_path / f"{vuln_type}_{algo}_final.zip"
    if final.exists():
        # Final model exists, but we don't know the step count
        # Prefer checkpoint with known step count unless no checkpoint exists
        if best_path is None:
            return str(final).replace(".zip", ""), best_step

    return best_path, best_step


def train_single(vuln_type: str, algo: str, args,
                  model_path: str | None = None,
                  model_dir_override: str | None = None,
                  timesteps_override: int | None = None,
                  target_override: str | None = None):
    """Train a single agent on one vulnerability type.

    Args:
        vuln_type: Vulnerability type (sqli, xss).
        algo: RL algorithm (dqn, ppo).
        args: Parsed CLI arguments.
        model_path: Optional path to a pre-trained model to continue from
                     (used by multi-target training).
        model_dir_override: Optional override for model save directory
                             (used by multi-target training for the universal dir).
        timesteps_override: Optional override for total timesteps this call.
        target_override: Optional override for the target application.
    """
    target = target_override or args.target
    timesteps = timesteps_override or args.timesteps

    # When resuming, reuse the specified model directory; otherwise create a new one
    if args.resume and not model_path:
        resume_dir = Path(args.resume)
        if resume_dir.is_dir():
            model_dir = str(resume_dir)
        else:
            # User passed a model file path — use its parent directory
            model_dir = str(resume_dir.parent)
        run_name = Path(model_dir).name
        log_dir = str(Path(args.log_dir) / run_name)
    elif model_dir_override:
        model_dir = model_dir_override
        run_name = Path(model_dir).name
        log_dir = str(Path(args.log_dir) / run_name)
    else:
        run_name = f"{vuln_type}_{algo}_{args.security}_{datetime.now():%Y%m%d_%H%M%S}"
        log_dir = str(Path(args.log_dir) / run_name)
        model_dir = str(Path(args.model_dir) / run_name)

    tb_dir = str(Path(args.log_dir) / "tensorboard")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    logger.info(f"{'='*60}")
    logger.info(f"Training: {algo.upper()} on {vuln_type.upper()}")
    logger.info(f"Target: {target}")
    logger.info(f"Security Level: {args.security}")
    logger.info(f"Total Timesteps: {timesteps}")
    logger.info(f"Max Steps/Episode: {args.max_steps}")
    logger.info(f"Logs: {log_dir}")
    logger.info(f"Models: {model_dir}")
    logger.info(f"TensorBoard: {tb_dir}")
    if model_path:
        logger.info(f"Continuing from model: {model_path}")
    elif args.resume:
        logger.info(f"Resuming from: {args.resume}")
    logger.info(f"{'='*60}")

    # Resolve target URL
    target_url = _resolve_target_url(args.dvwa_url, target)

    # Create environment
    env = create_environment(
        vuln_type=vuln_type,
        dvwa_url=target_url,
        security=args.security,
        max_steps=args.max_steps,
        log_dir=log_dir,
        render_mode="human" if args.verbose else "log",
        target=target,
        use_playwright=getattr(args, "playwright", False),
        headless=not getattr(args, "no_headless", False),
        api_key=getattr(args, "api_key", None),
    )

    # Hyperparameter config (can be loaded from YAML later)
    config = {
        "learning_rate": args.lr,
        "gamma": 0.99,
        "net_arch": [128, 128],
    }

    # Load from checkpoint or create new agent
    reset_timesteps = True
    if model_path:
        # Continue from a model produced by a previous multi-target phase.
        # Reset exploration so the agent re-explores on the new target instead
        # of blindly exploiting strategies it learned elsewhere (e.g. DVWA).
        exploration_reset = getattr(args, "exploration_reset_eps", 0.8)
        logger.info(f"Loading model from previous phase: {model_path}")
        if algo == "dqn":
            model = DQN.load(
                model_path, env=env,
                exploration_initial_eps=exploration_reset,
                exploration_fraction=0.3,
            )
            logger.info(
                f"Exploration reset: eps={exploration_reset}, "
                f"fraction=0.3 of {timesteps} steps"
            )
        else:
            model = PPO.load(model_path, env=env)
        model.tensorboard_log = tb_dir
        reset_timesteps = True   # reset step counter so exploration schedule applies
    elif args.resume:
        checkpoint_path, resumed_step = find_latest_checkpoint(
            model_dir, vuln_type, algo
        )
        if checkpoint_path:
            logger.info(f"Loading checkpoint: {checkpoint_path} (step {resumed_step})")
            if algo == "dqn":
                model = DQN.load(checkpoint_path, env=env)
            else:
                model = PPO.load(checkpoint_path, env=env)
            model.tensorboard_log = tb_dir
            reset_timesteps = False
            logger.info(f"Resumed from step {resumed_step}. "
                        f"Will train for {timesteps} additional timesteps.")
        else:
            logger.warning(f"No checkpoint found in {model_dir}, starting fresh.")
            model = create_agent(algo, env, tb_dir, config)
    else:
        model = create_agent(algo, env, tb_dir, config)

    # Callbacks
    cb_list = [
        PentestMetricsCallback(log_freq=100, verbose=1),
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=model_dir,
            name_prefix=f"{vuln_type}_{algo}",
        ),
    ]

    # Optional LLM advisor — fires suggestions when agent plateaus
    api_key = getattr(args, "api_key", None)
    if getattr(args, "llm_advisor", False):
        action_names = []
        raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
        if hasattr(raw_env, "ACTIONS"):
            action_names = list(raw_env.ACTIONS.values())
        cb_list.append(LLMGuidedCallback(
            api_key=api_key,
            vuln_type=vuln_type,
            action_names=action_names,
            plateau_window=20,
            plateau_threshold=3.0,
            cooldown_episodes=15,
        ))
        logger.info("LLM Advisor callback enabled")

    callbacks = CallbackList(cb_list)

    # Train
    logger.info("Starting training...")
    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=reset_timesteps,
    )

    # Save final model
    final_path = str(Path(model_dir) / f"{vuln_type}_{algo}_final")
    model.save(final_path)
    logger.info(f"Final model saved: {final_path}")

    env.close()
    return final_path


def train_multi_target(vuln_type: str, algo: str, targets: list[str], args):
    """
    Train a single model sequentially across multiple targets.

    The model is saved to a universal directory (models/universal_{vuln}_{algo}/)
    and carried forward between targets so it accumulates experience.

    Args:
        vuln_type: Vulnerability type (e.g. "sqli").
        algo: RL algorithm ("dqn" or "ppo").
        targets: List of target names (e.g. ["dvwa", "juiceshop"]).
        args: Parsed CLI arguments.
    """
    steps_per_target = args.timesteps // len(targets)
    universal_dir = str(
        Path(args.model_dir) / f"universal_{vuln_type}_{algo}"
    )
    os.makedirs(universal_dir, exist_ok=True)

    model_path = None

    # If resuming, look for an existing model in the universal dir
    if args.resume:
        checkpoint, _ = find_latest_checkpoint(universal_dir, vuln_type, algo)
        if checkpoint:
            model_path = checkpoint
            logger.info(f"Resuming multi-target from: {model_path}")

    for i, target in enumerate(targets):
        logger.info(f"\n{'#'*60}")
        logger.info(
            f"MULTI-TARGET Phase {i+1}/{len(targets)}: "
            f"{target.upper()} ({steps_per_target} timesteps)"
        )
        logger.info(f"{'#'*60}\n")

        model_path = train_single(
            vuln_type,
            algo,
            args,
            model_path=model_path,
            model_dir_override=universal_dir,
            timesteps_override=steps_per_target,
            target_override=target,
        )

    logger.info(f"\nMulti-target training complete. Universal model: {model_path}")
    return model_path


def train_curriculum(vuln_type: str, algo: str, args):
    """
    Curriculum training: start at Low difficulty, progress to Medium then High.
    The model is carried over between difficulty levels.

    Supports --curriculum-start to skip already-completed levels and
    --resume to load an existing model as the starting point.
    When entering a new curriculum level, exploration is reset so the
    agent re-explores under the harder filter.
    """
    all_levels = ["low", "medium", "high"]
    start = getattr(args, "curriculum_start", "low") or "low"
    start_idx = all_levels.index(start)
    levels = all_levels[start_idx:]

    steps_per_level = args.timesteps // len(levels)

    # Resolve starting model: --resume points at an existing model/dir
    model_path = None
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            # Look for a final model inside the directory
            checkpoint, _ = find_latest_checkpoint(
                str(resume_path), vuln_type, algo
            )
            if checkpoint:
                model_path = checkpoint
            else:
                final = resume_path / f"{vuln_type}_{algo}_final"
                if (resume_path / f"{vuln_type}_{algo}_final.zip").exists():
                    model_path = str(final)
        else:
            # Direct path to a model file
            model_path = str(resume_path)
        if model_path:
            logger.info(f"Curriculum resuming from model: {model_path}")
        else:
            logger.warning(f"No model found at {args.resume}, starting fresh.")

    exploration_reset_eps = getattr(args, "exploration_reset_eps", 0.8)

    for i, level in enumerate(levels):
        logger.info(f"\n{'#'*60}")
        logger.info(f"CURRICULUM Phase {i+1}/{len(levels)}: Security={level.upper()}")
        logger.info(f"Timesteps this phase: {steps_per_level}")
        logger.info(f"{'#'*60}\n")

        args.security = level

        run_name = f"{vuln_type}_{algo}_curriculum_{level}_{datetime.now():%Y%m%d_%H%M%S}"
        log_dir = str(Path(args.log_dir) / run_name)
        model_dir = str(Path(args.model_dir) / run_name)
        tb_dir = str(Path(args.log_dir) / "tensorboard")

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        env = create_environment(
            vuln_type=vuln_type,
            dvwa_url=args.dvwa_url,
            security=level,
            max_steps=args.max_steps,
            log_dir=log_dir,
            render_mode="human" if args.verbose else "log",
        )

        if model_path:
            # Load previous model (from prior phase or --resume)
            if algo == "dqn":
                model = DQN.load(
                    model_path,
                    env=env,
                    exploration_initial_eps=exploration_reset_eps,
                    exploration_fraction=0.3,
                )
            else:
                model = PPO.load(model_path, env=env)
            model.tensorboard_log = tb_dir
            logger.info(f"Loaded model from: {model_path}")
            if algo == "dqn":
                logger.info(
                    f"Exploration reset: eps={exploration_reset_eps}, "
                    f"fraction=0.3 of {steps_per_level} steps"
                )
        else:
            config = {"learning_rate": args.lr, "gamma": 0.99, "net_arch": [128, 128]}
            model = create_agent(algo, env, tb_dir, config)

        callbacks = CallbackList([
            PentestMetricsCallback(log_freq=100, verbose=1),
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=model_dir,
                name_prefix=f"{vuln_type}_{algo}_{level}",
            ),
        ])

        model.learn(
            total_timesteps=steps_per_level,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=True,
        )

        model_path = str(Path(model_dir) / f"{vuln_type}_{algo}_{level}_final")
        model.save(model_path)
        logger.info(f"Phase {i+1} complete. Model: {model_path}")
        env.close()

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="AI Pentest Assistant - RL Agent Training"
    )
    parser.add_argument("--target", type=str, default="dvwa",
                        choices=["dvwa", "juiceshop", "webgoat", "dvwa_stored", "juiceshop_dom"],
                        help="Target application for single-target training")
    parser.add_argument("--targets", type=str, nargs="+", default=None,
                        choices=["dvwa", "juiceshop", "webgoat", "dvwa_stored", "juiceshop_dom"],
                        help="Multiple targets for transfer learning "
                             "(trains sequentially, carrying model forward)")
    parser.add_argument("--vuln", type=str, default="sqli",
                        choices=["sqli", "xss", "cmdi", "ssti", "both", "sqli+xss"],
                        help="Vulnerability type to train on ('both'/'sqli+xss' = SQLi then XSS)")
    parser.add_argument("--algo", type=str, default="dqn",
                        choices=["dqn", "ppo"],
                        help="RL algorithm")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Max steps per episode")
    parser.add_argument("--security", type=str, default="low",
                        choices=["low", "medium", "high"],
                        help="DVWA security level")
    parser.add_argument("--dvwa-url", type=str,
                        default="http://localhost:8080",
                        help="DVWA base URL")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="Log directory")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="Model checkpoint directory")
    parser.add_argument("--save-freq", type=int, default=10000,
                        help="Checkpoint save frequency (steps)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from a checkpoint directory "
                             "(e.g., models/sqli_dqn_medium_20260214_112115)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Use curriculum learning (low->medium->high)")
    parser.add_argument("--curriculum-start", type=str, default="low",
                        choices=["low", "medium", "high"],
                        help="Starting security level for curriculum "
                             "(skip already-trained levels)")
    parser.add_argument("--exploration-reset-eps", type=float, default=0.8,
                        help="Initial exploration epsilon when entering a new "
                             "curriculum level (default: 0.8)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed episode output")
    parser.add_argument("--llm-advisor", action="store_true",
                        help="Enable LLM Advisor callback — asks Claude Haiku for "
                             "suggestions when agent plateaus (~$2/50k steps)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key for LLM features "
                             "(default: ANTHROPIC_API_KEY env var)")
    parser.add_argument("--playwright", action="store_true",
                        help="Use Playwright browser for Juice Shop XSS DOM detection")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show the Playwright browser window (debug mode)")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_dir, level="DEBUG" if args.verbose else "INFO")

    logger.info("=" * 60)
    logger.info("AI-Driven Penetration Testing Assistant")
    logger.info("Training Configuration:")
    if args.targets:
        logger.info(f"  Targets:       {', '.join(args.targets)} (multi-target)")
    else:
        logger.info(f"  Target:        {args.target}")
    logger.info(f"  Vulnerability: {args.vuln}")
    logger.info(f"  Algorithm:     {args.algo}")
    logger.info(f"  Timesteps:     {args.timesteps}")
    logger.info(f"  Security:      {args.security}")
    logger.info(f"  Curriculum:    {args.curriculum}")
    logger.info("=" * 60)

    vuln_types = ["sqli", "xss"] if args.vuln in ("both", "sqli+xss") else [args.vuln]

    for vuln in vuln_types:
        if args.targets:
            model_path = train_multi_target(vuln, args.algo, args.targets, args)
        elif args.curriculum:
            model_path = train_curriculum(vuln, args.algo, args)
        else:
            model_path = train_single(vuln, args.algo, args)

        logger.info(f"\nTraining complete for {vuln.upper()}!")
        logger.info(f"Final model: {model_path}")

    logger.info("\nAll training complete!")
    logger.info(f"View TensorBoard: tensorboard --logdir {args.log_dir}/tensorboard")


if __name__ == "__main__":
    main()
