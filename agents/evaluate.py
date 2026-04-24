"""
Evaluation Script
Runs a trained RL agent against DVWA / Juice Shop and measures performance.
Produces structured results for the project report.

Usage:
    python -m agents.evaluate --model models/sqli_dqn_final --vuln sqli --episodes 50
    python -m agents.evaluate --model models/xss_ppo_final --vuln xss --episodes 50
    python -m agents.evaluate --compare models/sqli_dqn_final models/sqli_ppo_final --vuln sqli

Multi-target evaluation (same model across targets):
    python -m agents.evaluate --targets dvwa juiceshop \
        --model models/universal_sqli_dqn/sqli_dqn_final --vuln sqli --episodes 50
"""

import argparse
import sys
import json
import os
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3.common.monitor import Monitor

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
from utils.logger import setup_logging
from utils.model_loader import load_model
from utils.report_generator import PentestReportGenerator
from utils.narrative_generator import NarrativeGenerator

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


def evaluate_agent(model, env, algo_name: str, num_episodes: int = 50,
                   deterministic: bool = True) -> dict:
    """
    Run the agent for N episodes and collect performance metrics.

    Returns:
        Dict with detailed evaluation results.
    """
    results = {
        "algo": algo_name,
        "num_episodes": num_episodes,
        "deterministic": deterministic,
        "episodes": [],
        "summary": {},
    }

    total_rewards = []
    total_steps = []
    successes = 0
    steps_to_success = []
    action_counts = defaultdict(int)

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_actions = []
        step_details = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            episode_steps += 1
            episode_actions.append(int(action))
            action_counts[int(action)] += 1

            # Collect payload info from the step
            payload_info = info.get("payload_info", {})
            step_details.append({
                "step": episode_steps,
                "action": info.get("action_name", ""),
                "action_id": int(action),
                "reward": reward,
                "severity_score": info.get("severity_score", 0),
                "payload": payload_info.get("payload", ""),
                "parameter": payload_info.get("parameter", ""),
                "url_path": payload_info.get("url_path", ""),
                "full_request_url": payload_info.get("full_request_url", ""),
                "response_snippet": payload_info.get("response_snippet", ""),
                "reflected": payload_info.get("reflected", False),
            })

            done = terminated or truncated

        ep_success = terminated  # terminated = goal reached
        if ep_success:
            successes += 1
            steps_to_success.append(episode_steps)

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)

        results["episodes"].append({
            "episode": ep + 1,
            "reward": episode_reward,
            "steps": episode_steps,
            "success": ep_success,
            "actions": episode_actions,
            "step_details": step_details,
        })

        if (ep + 1) % 10 == 0:
            logger.info(
                f"  Eval [{ep+1}/{num_episodes}] "
                f"Success Rate: {successes/(ep+1):.1%} "
                f"Avg Reward: {np.mean(total_rewards):.1f}"
            )

    # Compute summary statistics
    results["summary"] = {
        "success_rate": successes / num_episodes,
        "total_successes": successes,
        "mean_reward": float(np.mean(total_rewards)),
        "std_reward": float(np.std(total_rewards)),
        "median_reward": float(np.median(total_rewards)),
        "mean_steps": float(np.mean(total_steps)),
        "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        "min_steps_to_success": int(min(steps_to_success)) if steps_to_success else None,
        "action_distribution": dict(action_counts),
    }

    return results


def run_random_baseline(env, num_episodes: int = 50) -> dict:
    """Run a random agent as baseline comparison."""
    logger.info("Running random baseline agent...")

    total_rewards = []
    total_steps = []
    successes = 0
    steps_to_success = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
            done = terminated or truncated

        if terminated:
            successes += 1
            steps_to_success.append(episode_steps)

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)

    return {
        "algo": "Random",
        "summary": {
            "success_rate": successes / num_episodes,
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_steps": float(np.mean(total_steps)),
            "mean_steps_to_success": float(np.mean(steps_to_success)) if steps_to_success else None,
        }
    }


def print_comparison_table(all_results: list[dict]):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Agent':<15} {'Success%':>10} {'Avg Reward':>12} {'Avg Steps':>10} "
          f"{'Steps->Win':>10}")
    print("-" * 70)

    for r in all_results:
        s = r["summary"]
        steps_win = f"{s['mean_steps_to_success']:.1f}" if s.get('mean_steps_to_success') else "N/A"
        print(f"{r['algo']:<15} {s['success_rate']:>9.1%} {s['mean_reward']:>12.1f} "
              f"{s['mean_steps']:>10.1f} {steps_win:>10}")

    print("=" * 70)


def print_cross_target_table(target_results: dict[str, dict]):
    """Print a comparison table showing the same model's performance across targets."""
    print("\n" + "=" * 70)
    print("CROSS-TARGET EVALUATION")
    print("=" * 70)
    print(f"{'Target':<15} {'Success%':>10} {'Avg Reward':>12} {'Avg Steps':>10} "
          f"{'Steps->Win':>10}")
    print("-" * 70)

    for target_name, r in target_results.items():
        s = r["summary"]
        steps_win = (
            f"{s['mean_steps_to_success']:.1f}"
            if s.get("mean_steps_to_success") else "N/A"
        )
        print(
            f"{target_name:<15} {s['success_rate']:>9.1%} "
            f"{s['mean_reward']:>12.1f} "
            f"{s['mean_steps']:>10.1f} {steps_win:>10}"
        )

    print("=" * 70)


def create_env(vuln_type: str, dvwa_url: str, security: str, max_steps: int,
               target: str = "dvwa", use_playwright: bool = False,
               headless: bool = True):
    """Create evaluation environment."""
    if target == "juiceshop":
        if vuln_type == "sqli":
            env = JuiceShopSQLiEnv(dvwa_url=dvwa_url, security_level=security,
                                   max_steps=max_steps, render_mode="log")
        elif vuln_type == "xss":
            env = JuiceShopXSSEnv(dvwa_url=dvwa_url, security_level=security,
                                  max_steps=max_steps, render_mode="log",
                                  use_playwright=use_playwright,
                                  headless=headless)
        elif vuln_type == "ssti":
            env = JuiceShopSSTIEnv(dvwa_url=dvwa_url, security_level=security,
                                   max_steps=max_steps, render_mode="log")
        else:
            raise ValueError(f"Juice Shop supports sqli, xss, and ssti, got: {vuln_type}")
    elif target == "webgoat":
        if vuln_type == "sqli":
            env = WebGoatSQLiEnv(dvwa_url=dvwa_url, security_level=security,
                                 max_steps=max_steps, render_mode="log")
        elif vuln_type == "cmdi":
            env = WebGoatCMDiEnv(dvwa_url=dvwa_url, security_level=security,
                                 max_steps=max_steps, render_mode="log")
        elif vuln_type == "xss":
            env = WebGoatXSSEnv(dvwa_url=dvwa_url, security_level=security,
                                max_steps=max_steps, render_mode="log")
        else:
            raise ValueError(f"WebGoat target supports sqli, cmdi, and xss, got: {vuln_type}")
    elif vuln_type == "sqli":
        env = SQLiEnv(dvwa_url=dvwa_url, security_level=security,
                      max_steps=max_steps, render_mode="log")
    elif target == "dvwa_stored" and vuln_type == "xss":
        env = DVWAStoredXSSEnv(dvwa_url=dvwa_url, security_level=security,
                               max_steps=max_steps, render_mode="log")
    elif target == "juiceshop_dom" and vuln_type == "xss":
        env = JuiceShopXSSEnv(dvwa_url=dvwa_url, security_level=security,
                              max_steps=max_steps, render_mode="log",
                              use_playwright=use_playwright, headless=headless)
    elif vuln_type == "xss":
        env = XSSEnv(dvwa_url=dvwa_url, security_level=security,
                     max_steps=max_steps, render_mode="log")
    elif vuln_type == "cmdi":
        env = CMDiEnv(dvwa_url=dvwa_url, security_level=security,
                      max_steps=max_steps, render_mode="log")
    elif vuln_type == "ssti":
        env = DVWASSTIEnv(dvwa_url=dvwa_url, security_level=security,
                          max_steps=max_steps, render_mode="log")
    else:
        raise ValueError(f"Unknown vuln: {vuln_type}")
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained pentest agents"
    )
    parser.add_argument("--target", type=str, default="dvwa",
                        choices=["dvwa", "juiceshop", "webgoat", "dvwa_stored", "juiceshop_dom"],
                        help="Target application for single-target evaluation")
    parser.add_argument("--targets", type=str, nargs="+", default=None,
                        choices=["dvwa", "juiceshop", "webgoat", "dvwa_stored", "juiceshop_dom"],
                        help="Evaluate the same model across multiple targets")
    parser.add_argument("--model", type=str, nargs="+",
                        help="Path(s) to trained model(s)")
    parser.add_argument("--vuln", type=str, required=True,
                        choices=["sqli", "xss", "cmdi", "ssti"])
    parser.add_argument("--episodes", type=int, default=50,
                        help="Number of evaluation episodes")
    parser.add_argument("--security", type=str, default="low",
                        choices=["low", "medium", "high"])
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--dvwa-url", type=str,
                        default="http://localhost:8080")
    parser.add_argument("--include-random", action="store_true",
                        help="Include random baseline")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--report", action="store_true", default=True,
                        help="Generate pentest report after evaluation (default: on)")
    parser.add_argument("--no-report", action="store_false", dest="report",
                        help="Disable pentest report generation")
    parser.add_argument("--narrative", action="store_true", default=False,
                        help="Generate LLM narrative report after evaluation (~$0.03/report)")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key for narrative generation "
                             "(default: ANTHROPIC_API_KEY env var)")
    parser.add_argument("--playwright", action="store_true",
                        help="Use Playwright browser for Juice Shop XSS DOM detection")
    parser.add_argument("--no-headless", action="store_true",
                        help="Show the Playwright browser window (debug mode)")

    args = parser.parse_args()
    setup_logging(level="INFO")

    use_playwright = getattr(args, "playwright", False)
    headless = not getattr(args, "no_headless", False)

    # --targets mode: evaluate the same model across multiple targets
    if args.targets:
        if not args.model or len(args.model) != 1:
            parser.error("--targets requires exactly one --model path")

        cross_target_results = {}
        for target in args.targets:
            target_url = _resolve_target_url(args.dvwa_url, target)

            logger.info(f"\nEvaluating on target: {target.upper()}")
            env = create_env(args.vuln, target_url, args.security,
                             args.max_steps, target=target,
                             use_playwright=use_playwright, headless=headless)
            model, algo_name = load_model(args.model[0], env)
            results = evaluate_agent(model, env, algo_name, args.episodes)
            results["target"] = target
            cross_target_results[target] = results
            env.close()

        # Print cross-target comparison
        print_cross_target_table(cross_target_results)

        # Generate reports per target
        if args.report:
            narrative_gen = NarrativeGenerator(api_key=getattr(args, "api_key", None)) if args.narrative else None
            for target, r in cross_target_results.items():
                target_url = _resolve_target_url(args.dvwa_url, target)
                report_gen = PentestReportGenerator(
                    eval_results=r,
                    vuln_type=args.vuln,
                    target_url=target_url,
                    security_level=args.security,
                )
                report = report_gen.generate()
                report_path = report_gen.save_json(report)
                report_gen.print_console_summary(report)
                logger.info(f"Pentest report ({target}) saved to {report_path}")
                if narrative_gen and narrative_gen.available:
                    narrative = narrative_gen.generate(
                        report=report,
                        vuln_type=args.vuln,
                        target_url=target_url,
                        security_level=args.security,
                    )
                    nb = str(report_path.stem)
                    narrative_gen.save_markdown(narrative, str(report_path.parent), nb)
                    print(f"\n--- LLM Narrative ({target}) ---\n{narrative}\n")

        # Save combined results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(cross_target_results, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")

        logger.info("\nCross-target evaluation complete!")
        return

    # Single-target mode (original behaviour)
    target_url = _resolve_target_url(args.dvwa_url, args.target)

    all_results = []

    # Random baseline
    if args.include_random:
        env = create_env(args.vuln, target_url, args.security, args.max_steps,
                         target=args.target,
                         use_playwright=use_playwright, headless=headless)
        baseline = run_random_baseline(env, args.episodes)
        all_results.append(baseline)
        env.close()

    # Evaluate each model
    for model_path in (args.model or []):
        logger.info(f"\nEvaluating model: {model_path}")
        env = create_env(args.vuln, target_url, args.security, args.max_steps,
                         target=args.target,
                         use_playwright=use_playwright, headless=headless)
        model, algo_name = load_model(model_path, env)
        results = evaluate_agent(model, env, algo_name, args.episodes)
        all_results.append(results)
        env.close()

    # Print comparison
    if all_results:
        print_comparison_table(all_results)

    # Generate pentest report for each evaluated model
    if args.report:
        narrative_gen = NarrativeGenerator(api_key=getattr(args, "api_key", None)) if args.narrative else None
        for r in all_results:
            if r["algo"] == "Random":
                continue  # Skip random baseline for report
            report_gen = PentestReportGenerator(
                eval_results=r,
                vuln_type=args.vuln,
                target_url=target_url,
                security_level=args.security,
            )
            report = report_gen.generate()
            report_path = report_gen.save_json(report)
            report_gen.print_console_summary(report)
            logger.info(f"Pentest report saved to {report_path}")
            if narrative_gen and narrative_gen.available:
                narrative = narrative_gen.generate(
                    report=report,
                    vuln_type=args.vuln,
                    target_url=target_url,
                    security_level=args.security,
                )
                nb = str(report_path.stem)
                narrative_gen.save_markdown(narrative, str(report_path.parent), nb)
                print(f"\n--- LLM Narrative ---\n{narrative}\n")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to {output_path}")

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
