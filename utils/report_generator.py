"""
Pentest Report Generator
Produces structured JSON reports and console summaries from enriched
evaluation data collected during RL agent testing against DVWA.

Generates:
    - JSON report (machine-readable, saved to results/)
    - Console summary (printed to terminal after evaluation)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class PentestReportGenerator:
    """Generates penetration testing reports from evaluation results."""

    def __init__(self, eval_results: dict, vuln_type: str,
                 target_url: str, security_level: str,
                 output_dir: str = "./results"):
        self.eval_results = eval_results
        self.vuln_type = vuln_type
        self.target_url = target_url.rstrip("/")
        self.security_level = security_level
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> dict:
        """Generate the full report and return it as a dict."""
        report = {
            "header": self._build_header(),
            "executive_summary": self._build_executive_summary(),
            "findings": self._build_findings(),
            "all_payloads_tested": self._build_all_payloads(),
            "vulnerable_parameters": self._build_vulnerable_parameters(),
            "request_response_evidence": self._build_evidence(),
        }
        return report

    def save_json(self, report: dict) -> Path:
        """Save the report as a JSON file in the output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo = self.eval_results.get("algo", "unknown")
        filename = f"pentest_report_{self.vuln_type}_{algo}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Report saved to {filepath}")
        return filepath

    def print_console_summary(self, report: dict):
        """Print a readable summary to the terminal."""
        header = report["header"]
        summary = report["executive_summary"]
        findings = report["findings"]

        print("\n" + "=" * 72)
        print("  PENETRATION TEST REPORT")
        print("=" * 72)

        # Header
        print(f"  Target:           {header['target_url']}")
        print(f"  Vulnerability:    {header['vulnerability_type']}")
        print(f"  Security Level:   {header['security_level']}")
        print(f"  Algorithm:        {header['algorithm']}")
        print(f"  Timestamp:        {header['timestamp']}")
        print("-" * 72)

        # Executive Summary
        print("\n  EXECUTIVE SUMMARY")
        print(f"  Total Episodes:       {summary['total_episodes']}")
        print(f"  Success Rate:         {summary['success_rate']:.1%}")
        print(f"  Mean Steps:           {summary['mean_steps']:.1f}")
        print(f"  Unique Payloads:      {summary['unique_payloads_tried']}")

        if summary.get("mean_steps_to_success"):
            print(f"  Avg Steps to Success: {summary['mean_steps_to_success']:.1f}")

        # Findings
        if findings:
            print(f"\n  FINDINGS ({len(findings)} successful exploits)")
            print("-" * 72)
            for i, finding in enumerate(findings, 1):
                print(f"\n  [{i}] Episode {finding['episode']}, "
                      f"Step {finding['step']}")
                print(f"      Payload:    {finding['payload']}")
                print(f"      Parameter:  {finding['parameter']}")
                print(f"      Severity:   {finding['severity_score']}/100")
                print(f"      Request:    {finding['full_request_url'][:100]}")
                if finding.get("response_snippet"):
                    snippet = finding["response_snippet"][:200]
                    print(f"      Response:   {snippet}")
                if finding.get("reflected"):
                    print(f"      Reflected:  Yes")
        else:
            print("\n  FINDINGS: None - no successful exploits detected.")

        # Vulnerable Parameters
        vuln_params = report["vulnerable_parameters"]
        if vuln_params:
            print(f"\n  VULNERABLE PARAMETERS")
            print("-" * 72)
            for param in vuln_params:
                status = "VULNERABLE" if param["confirmed_vulnerable"] else "tested"
                print(f"    {param['parameter']:>12} : {status} "
                      f"({param['total_attempts']} attempts, "
                      f"{param['successful_attempts']} successful)")

        print("\n" + "=" * 72)
        print()

    def _build_header(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "target_url": self.target_url,
            "vulnerability_type": self.vuln_type,
            "security_level": self.security_level,
            "algorithm": self.eval_results.get("algo", "unknown"),
        }

    def _build_executive_summary(self) -> dict:
        summary = self.eval_results.get("summary", {})
        episodes = self.eval_results.get("episodes", [])

        # Collect all unique payloads across all episodes
        all_payloads = set()
        for ep in episodes:
            for step in ep.get("step_details", []):
                payload = step.get("payload", "")
                if payload:
                    all_payloads.add(payload)

        return {
            "total_episodes": summary.get("total_successes", 0) + (
                len(episodes) - summary.get("total_successes", 0)
            ),
            "success_rate": summary.get("success_rate", 0.0),
            "mean_steps": summary.get("mean_steps", 0.0),
            "mean_reward": summary.get("mean_reward", 0.0),
            "mean_steps_to_success": summary.get("mean_steps_to_success"),
            "unique_payloads_tried": len(all_payloads),
        }

    def _build_findings(self) -> list[dict]:
        """Build a list of findings, one per successful episode."""
        findings = []
        episodes = self.eval_results.get("episodes", [])

        for ep in episodes:
            if not ep.get("success"):
                continue

            # The last step that caused termination (success) is the finding
            step_details = ep.get("step_details", [])
            if not step_details:
                continue

            # Find the step that triggered success (last step)
            winning_step = step_details[-1]

            findings.append({
                "episode": ep.get("episode", 0),
                "step": winning_step.get("step", 0),
                "action": winning_step.get("action", ""),
                "payload": winning_step.get("payload", ""),
                "parameter": winning_step.get("parameter", ""),
                "url_path": winning_step.get("url_path", ""),
                "full_request_url": winning_step.get("full_request_url", ""),
                "response_snippet": winning_step.get("response_snippet", ""),
                "reflected": winning_step.get("reflected", False),
                "severity_score": winning_step.get("severity_score", 0),
            })

        return findings

    def _build_all_payloads(self) -> dict:
        """Deduplicated list of all payloads, grouped by action category."""
        episodes = self.eval_results.get("episodes", [])
        grouped = defaultdict(list)

        seen = set()
        for ep in episodes:
            for step in ep.get("step_details", []):
                payload = step.get("payload", "")
                action = step.get("action", "unknown")
                if not payload:
                    continue
                key = (action, payload)
                if key in seen:
                    continue
                seen.add(key)

                grouped[action].append({
                    "payload": payload,
                    "full_request_url": step.get("full_request_url", ""),
                    "reflected": step.get("reflected", False),
                })

        return dict(grouped)

    def _build_vulnerable_parameters(self) -> list[dict]:
        """Which parameters were tested and which had confirmed vulnerabilities."""
        episodes = self.eval_results.get("episodes", [])
        param_stats = defaultdict(lambda: {
            "total_attempts": 0,
            "successful_attempts": 0,
            "confirmed_vulnerable": False,
        })

        for ep in episodes:
            for step in ep.get("step_details", []):
                param = step.get("parameter", "")
                if not param:
                    continue
                param_stats[param]["total_attempts"] += 1
                if step.get("reflected", False):
                    param_stats[param]["successful_attempts"] += 1
                    param_stats[param]["confirmed_vulnerable"] = True

        return [
            {"parameter": param, **stats}
            for param, stats in param_stats.items()
        ]

    def _build_evidence(self) -> list[dict]:
        """Request/response evidence for each finding."""
        findings = self._build_findings()
        evidence = []

        for finding in findings:
            evidence.append({
                "episode": finding["episode"],
                "step": finding["step"],
                "payload": finding["payload"],
                "request_url": finding["full_request_url"],
                "response_snippet": finding["response_snippet"],
                "reflected": finding["reflected"],
                "severity_score": finding["severity_score"],
            })

        return evidence
