"""
LLM Narrative Generator
Uses Claude Opus to produce a professional human-readable penetration testing
report from the structured JSON output of PentestReportGenerator.

Usage:
    gen = NarrativeGenerator(api_key="sk-ant-...")
    narrative = gen.generate(report, vuln_type="sqli", target_url="http://localhost:8080")
    path = gen.save_markdown(narrative, output_dir="results/", basename="report_sqli_DQN")
"""

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_VULN_CONTEXT = {
    "sqli": "SQL Injection (CWE-89, OWASP A03:2021 — Injection)",
    "xss":  "Cross-Site Scripting (CWE-79, OWASP A03:2021 — Injection)",
    "cmdi": "OS Command Injection (CWE-78, OWASP A03:2021 — Injection)",
    "ssti": "Server-Side Template Injection (CWE-94, OWASP A03:2021 — Injection)",
}

# Realistic attacker capabilities per vuln type — grounds the Attack Scenarios section
_ATTACKER_IMPACT = {
    "sqli": (
        "dump the entire database (usernames, hashed passwords, PII), "
        "bypass authentication by injecting always-true conditions, "
        "read sensitive files via LOAD_FILE(), or in some configurations "
        "write a web shell using INTO OUTFILE"
    ),
    "xss": (
        "steal session cookies and hijack authenticated accounts, "
        "redirect users to phishing pages, log keystrokes, "
        "perform actions on behalf of the victim (CSRF-style), "
        "or deliver malware through a trusted domain"
    ),
    "cmdi": (
        "execute arbitrary OS commands as the web server user, "
        "read /etc/passwd and other sensitive system files, "
        "establish a reverse shell for persistent access, "
        "pivot to internal network services, or exfiltrate data"
    ),
    "ssti": (
        "read server-side files and environment variables (including API keys and secrets), "
        "execute arbitrary code on the server through template engine sandbox escapes, "
        "achieve full Remote Code Execution (RCE), "
        "or enumerate internal infrastructure"
    ),
}

# Random style seed picked per run — prevents every report reading identically
_STYLE_SEEDS = [
    "Write directly and confidently, like a senior consultant presenting findings to a board.",
    "Be thorough but conversational — explain the impact as if talking to a developer who hasn't seen this before.",
    "Be precise and evidence-led. Let the payload data do the talking; add minimal editorial commentary.",
    "Write with urgency where it's warranted — if something is critical, say so plainly without hedging.",
    "Be clear and practical. Prioritise the 'so what?' — what does this mean for the business, what should they do first.",
]

_SYSTEM_PROMPT = """\
You are an experienced penetration tester writing a security assessment report.
Your reports are known for being clear, specific, and genuinely useful to the team receiving them.
You base everything strictly on the evidence provided — you never invent vulnerabilities or speculate beyond the data.
Write in flowing prose where appropriate; avoid bullet-point lists unless they genuinely aid clarity.
Do NOT use filler phrases like "it is important to note" or "in conclusion".
"""


class NarrativeGenerator:
    """
    Generates human-readable pentest narrative reports using Claude Opus.
    Each run varies in tone via a random style seed; temperature=1.0 ensures
    the prose never reads identically across runs.
    The vulnerable page URL and attacker impact scenarios are always surfaced.
    Falls back gracefully to a plain message if the API is unavailable.
    """

    MODEL = "claude-opus-4-6"

    def __init__(self, api_key: Optional[str] = None):
        self._available = False
        self._client = None
        try:
            import anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                logger.warning("NarrativeGenerator: ANTHROPIC_API_KEY not set — narrative disabled")
                return
            self._client = anthropic.Anthropic(api_key=key)
            self._available = True
            logger.debug("NarrativeGenerator: Claude Opus ready")
        except Exception as e:
            logger.warning(f"NarrativeGenerator unavailable: {e}")

    @property
    def available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        report: dict,
        vuln_type: str,
        target_url: str,
        security_level: str = "low",
    ) -> str:
        """
        Generate a markdown narrative for the given JSON report.

        Args:
            report:         Structured dict from PentestReportGenerator.generate()
            vuln_type:      "sqli", "xss", "cmdi", or "ssti"
            target_url:     URL that was tested
            security_level: Security level tested (low/medium/high)

        Returns:
            Markdown string. Returns a fallback notice if LLM is unavailable.
        """
        if not self._available:
            return (
                "# Narrative Unavailable\n\n"
                "Set `ANTHROPIC_API_KEY` to enable LLM narrative generation.\n"
            )

        vuln_desc = _VULN_CONTEXT.get(vuln_type, vuln_type.upper())
        attacker_impact = _ATTACKER_IMPACT.get(
            vuln_type, "exploit the vulnerability for malicious gain"
        )
        style_seed = random.choice(_STYLE_SEEDS)

        summary = report.get("executive_summary", {})
        vuln_params = report.get("vulnerable_parameters", [])
        findings = report.get("findings", [])

        # Build deduplicated top findings with their exact page URLs
        seen_payloads: set = set()
        top_findings = []
        for f in findings:
            p = f.get("payload", "")
            if p and p not in seen_payloads:
                seen_payloads.add(p)
                top_findings.append({
                    "payload": p,
                    "parameter": f.get("parameter", ""),
                    "page": f.get("url_path") or f.get("full_request_url", target_url),
                    "full_request": f.get("full_request_url", ""),
                    "response_snippet": f.get("response_snippet", "")[:300],
                    "reflected": f.get("reflected", False),
                    "severity": f.get("severity_score", 0),
                })
            if len(top_findings) >= 5:
                break

        # Distinct vulnerable pages — shown in the Executive Summary
        vulnerable_pages = list(dict.fromkeys(
            f["page"] for f in top_findings if f["page"]
        ))

        prompt = f"""Generate a penetration testing report from the assessment data below.

Tone/style instruction: {style_seed}

---

## Vulnerability
{vuln_desc}

## Target
- Base URL: {target_url}
- Security Level: {security_level}
- Vulnerable page(s) confirmed: {", ".join(vulnerable_pages) if vulnerable_pages else target_url}

## Assessment Statistics
- Total test episodes: {summary.get('total_episodes', 'N/A')}
- Success rate: {summary.get('success_rate', summary.get('success_rate_pct', 'N/A'))}
- Average steps to exploit: {summary.get('mean_steps_to_exploit', summary.get('mean_steps', 'N/A'))}

## Evidence — Top Exploits Found
```json
{json.dumps(top_findings, indent=2, default=str)[:2000]}
```

## Confirmed Vulnerable Parameters
```json
{json.dumps([p for p in vuln_params if p.get('confirmed_vulnerable')], indent=2, default=str)[:600]}
```

---

Write the report using the headings below exactly as shown. No section numbers.

**Executive Summary**
One paragraph. State the risk level (Critical / High / Medium / Low), name the exact page(s) where the vulnerability was confirmed, and describe the business impact plainly.

**Technical Findings**
What was found, on which exact page (URL) and parameter, and how it was confirmed. Reference specific payloads and what the server response revealed.

**Attack Scenarios**
Describe concretely what an attacker who found this could do next — be specific to this vuln type. Known realistic options include: {attacker_impact}. Write 2-3 realistic attack chains in prose, not a generic list.

**Proof of Concept**
Show 2-3 payloads from the evidence. For each, explain in one sentence what it does and why it worked on this endpoint.

**Risk Rating**
Estimated CVSS v3.1 base score. One-line justification per metric (AV, AC, PR, UI, S, C, I, A). State the final numeric score and severity label.

**Remediation**
3-5 specific fixes tailored to the confirmed parameter type. Name the function/library/approach — not just "sanitise inputs".

**References**
CWE link, OWASP Top 10 page, one further resource directly relevant to this vulnerability.

Keep the total under 900 words. Do not pad."""

        try:
            _approx_tokens = len(prompt) // 4
            _est_s = max(3, _approx_tokens // 600)
            _use_spinner = sys.stdout.isatty()
            if _use_spinner:
                from rich.console import Console as _Console
                from rich.status import Status as _Status
                _con = _Console(highlight=False)
                _spinner = _Status(
                    f"[bold cyan]Generating narrative report...[/bold cyan] [dim](~{_est_s}s estimated)[/dim]",
                    console=_con,
                    spinner="dots",
                )
                _spinner.start()
            _t0 = time.monotonic()
            response = self._client.messages.create(
                model=self.MODEL,
                max_tokens=2200,
                temperature=1.0,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            if _use_spinner:
                _spinner.stop()
                _con.print(f"[dim]  Narrative generated in {time.monotonic() - _t0:.1f}s[/dim]")
            narrative = response.content[0].text.strip()
            logger.info("Narrative generated successfully")
            return narrative
        except Exception as e:
            logger.warning("Narrative generation failed — check API key and network connectivity.")
            return "# Narrative Generation Failed\n\nThe report could not be generated. Check your API key and that the service is reachable.\n"

    def save_markdown(
        self,
        narrative: str,
        output_dir: str = "./results",
        basename: str = "pentest_report",
    ) -> Path:
        """
        Save the narrative to a .md file alongside the JSON report.

        Args:
            narrative:   Markdown string returned by generate()
            output_dir:  Directory to save into (created if missing)
            basename:    Filename without extension (e.g. "pentest_report_sqli_DQN_20260303")

        Returns:
            Path to the saved .md file
        """
        path = Path(output_dir) / f"{basename}_narrative.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(narrative, encoding="utf-8")
        logger.info(f"Narrative saved to {path}")
        return path
