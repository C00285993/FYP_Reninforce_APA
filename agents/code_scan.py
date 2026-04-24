"""
AI Pentest Assistant — Static Source Code Scanner

Analyses source code files for security vulnerabilities using Claude.
Works on any language: PHP, Python, JavaScript, Java, C#, Ruby, Go, etc.

Why this matters for the dynamic scanner:
  - The HTTP scanner can only test what it can reach (forms, URLs, APIs).
  - Source code exposes ALL vulnerabilities — including ones hidden behind auth,
    unreachable routes, dead code, or logic that can't be triggered by HTTP alone.
  - Together, static + dynamic = highest confidence findings.

Usage:
    # Scan a local folder
    python -m agents.code_scan --path /path/to/app/

    # Scan a specific file
    python -m agents.code_scan --path /path/to/login.php

    # Scan and save JSON report
    python -m agents.code_scan --path /path/to/app/ --output report.json

    # Limit to specific language
    python -m agents.code_scan --path /path/to/app/ --lang php

    # Combine with live scan (appends static findings to existing report)
    python -m agents.code_scan --path . --merge-report reports/scan_latest.json
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Colour helpers ─────────────────────────────────────────────────────────────
def _red(s):    return f"\033[91m{s}\033[0m"
def _yellow(s): return f"\033[93m{s}\033[0m"
def _green(s):  return f"\033[92m{s}\033[0m"
def _bold(s):   return f"\033[1m{s}\033[0m"
def _cyan(s):   return f"\033[96m{s}\033[0m"

# ── Supported file extensions → language label ─────────────────────────────────
LANGUAGE_MAP: dict[str, str] = {
    # Web back-end
    ".php":   "PHP",
    ".php3":  "PHP",
    ".php4":  "PHP",
    ".php5":  "PHP",
    ".phtml": "PHP",
    ".py":    "Python",
    ".rb":    "Ruby",
    ".java":  "Java",
    ".jsp":   "Java (JSP)",
    ".cs":    "C#",
    ".aspx":  "ASP.NET",
    ".asp":   "Classic ASP",
    ".go":    "Go",
    ".pl":    "Perl",
    ".cgi":   "CGI",
    # Web front-end
    ".js":    "JavaScript",
    ".mjs":   "JavaScript",
    ".ts":    "TypeScript",
    ".tsx":   "TypeScript (React)",
    ".jsx":   "JavaScript (React)",
    # Templates
    ".twig":  "Twig (PHP)",
    ".blade": "Blade (Laravel)",
    ".ejs":   "EJS (Node)",
    ".erb":   "ERB (Ruby)",
    ".hbs":   "Handlebars",
    ".pug":   "Pug",
    ".jinja": "Jinja2",
    ".j2":    "Jinja2",
    # Config / infra (can leak creds or have injection)
    ".env":   "Environment file",
    ".yml":   "YAML",
    ".yaml":  "YAML",
    ".xml":   "XML",
    ".json":  "JSON",
    ".conf":  "Config",
    ".ini":   "INI",
    ".sh":    "Shell",
    ".bash":  "Shell",
    ".ps1":   "PowerShell",
    ".sql":   "SQL",
}

# Skip these directories — they add noise without relevant app code
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    "vendor", "dist", "build", ".idea", ".vscode", "migrations",
}

# Skip files larger than this (minified JS, compiled artifacts, etc.)
MAX_FILE_BYTES = 150_000


def collect_files(root: str, lang_filter: Optional[str] = None) -> list[Path]:
    """Walk *root* and return all source files matching the language filter."""
    root_path = Path(root)
    files: list[Path] = []

    if root_path.is_file():
        files.append(root_path)
        return files

    for path in root_path.rglob("*"):
        # Skip hidden dirs and known noise dirs
        if any(part in SKIP_DIRS or part.startswith(".") for part in path.parts):
            continue
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in LANGUAGE_MAP:
            continue
        if lang_filter and LANGUAGE_MAP.get(ext, "").lower() != lang_filter.lower():
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            logger.info("Skipping large file: %s (%d bytes)", path, path.stat().st_size)
            continue
        files.append(path)

    return sorted(files)


# ── Vulnerability patterns for fast pre-screen (no API needed) ─────────────────
# If a file has none of these patterns it's very unlikely to have a vulnerability.
QUICK_PATTERNS: dict[str, list[str]] = {
    "PHP":        ["$_GET", "$_POST", "$_REQUEST", "$_COOKIE", "$_SERVER",
                   "mysql_query", "mysqli_query", "PDO", "exec(", "shell_exec(",
                   "system(", "passthru(", "eval(", "include(", "require(",
                   "file_get_contents(", "unserialize(", "echo", "print",
                   "header(", "setcookie(", "password_", "md5(", "sha1("],
    "Python":     ["request.", "execute(", "os.system(", "subprocess", "eval(",
                   "pickle", "yaml.load(", "open(", "render_template", "Markup("],
    "JavaScript": ["innerHTML", "eval(", "document.write", "fetch(", "XMLHttpRequest",
                   "dangerouslySetInnerHTML", "localStorage", "sessionStorage"],
    "Java":       ["executeQuery(", "createStatement(", "Runtime.exec(",
                   "ProcessBuilder(", "ObjectInputStream", "deserialize"],
    "Ruby":       ["params[", "eval(", "system(", "exec(", "Open3", "YAML.load("],
    "Go":         ["sql.Query(", "os/exec", "template.HTML(", "fmt.Sprintf("],
    "Shell":      ["$1", "$@", "eval ", "exec ", "curl ", "wget "],
}


def quick_prescreen(content: str, lang: str) -> bool:
    """Return True if the file contains any interesting pattern worth analysing."""
    patterns = QUICK_PATTERNS.get(lang, [])
    if not patterns:
        return True  # unknown language — send to LLM anyway
    return any(p in content for p in patterns)


# ── LLM analysis ───────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a security code reviewer specialising in web application vulnerabilities.
Analyse the provided source code and identify ALL security vulnerabilities.

For EVERY vulnerability found, output a JSON object with these exact fields:
{
  "severity":    "critical" | "high" | "medium" | "low" | "info",
  "category":    e.g. "SQL Injection", "XSS", "Command Injection", "Path Traversal",
                      "CSRF", "Broken Access Control", "Insecure Deserialization",
                      "Hardcoded Credentials", "Sensitive Data Exposure",
                      "Broken Authentication", "SSRF", "XXE", "Open Redirect",
                      "Security Misconfiguration", "Plaintext Storage",
  "line":        line number (integer) or null if not pinpointable,
  "snippet":     the exact vulnerable code fragment (≤120 chars),
  "description": one-sentence explanation of the vulnerability,
  "remediation": one-sentence fix
}

Return ONLY a JSON array of these objects — no markdown, no prose, no wrapper keys.
If the file has no vulnerabilities return an empty array: []
"""


_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2  # seconds; doubles on each retry


def analyse_file(
    file_path: Path,
    content: str,
    lang: str,
    client,
    model: str,
) -> list[dict]:
    """Send file content to Claude and return a list of vulnerability dicts.

    Retries up to _MAX_RETRIES times with exponential backoff on rate-limit
    and transient API errors.
    """
    user_msg = f"File: {file_path.name}  Language: {lang}\n\n```{lang.lower()}\n{content}\n```"

    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()

            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:])
            if raw.endswith("```"):
                raw = "\n".join(raw.split("\n")[:-1])

            findings = json.loads(raw)
            if not isinstance(findings, list):
                findings = []
            break  # success
        except json.JSONDecodeError:
            logger.warning("JSON parse failed for %s", file_path)
            findings = []
            break  # not a transient error
        except Exception as e:
            last_exc = e
            # Retry on rate-limit or transient server errors
            err_str = str(e).lower()
            is_retryable = any(kw in err_str for kw in ("rate", "429", "500", "502", "503", "overloaded"))
            if is_retryable and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY * (2 ** attempt)
                logger.info("API call for %s failed (%s), retrying in %ds...", file_path.name, type(e).__name__, delay)
                import time; time.sleep(delay)
                continue
            from utils.api_error_handler import handle_api_error
            handle_api_error(e, logger, context="code scan LLM analysis")
            findings = []
            break
    else:
        # All retries exhausted
        logger.warning("All %d retries failed for %s: %s", _MAX_RETRIES, file_path, last_exc)
        findings = []

    # Attach file path to each finding
    for f in findings:
        f["file"] = str(file_path)
        f["language"] = lang

    return findings


# ── Severity colour + rank ─────────────────────────────────────────────────────
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
SEVERITY_COLOR = {
    "critical": _red,
    "high":     _yellow,
    "medium":   _yellow,
    "low":      _cyan,
    "info":     str,
}


def severity_color(s: str):
    return SEVERITY_COLOR.get(s.lower(), str)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Static source code scanner — finds vulnerabilities in any language",
    )
    parser.add_argument("--path", required=True,
                        help="File or directory to scan")
    parser.add_argument("--lang", default=None,
                        help="Only scan files of this language (e.g. php, python, javascript)")
    parser.add_argument("--output", default=None,
                        help="Save JSON report to this file")
    parser.add_argument("--merge-report", default=None,
                        help="Append static findings to an existing dynamic scan report")
    parser.add_argument("--api-key", default=None,
                        help="Anthropic API key (overrides ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="Claude model to use (default: claude-sonnet-4-6)")
    parser.add_argument("--severity", default="low",
                        choices=["critical", "high", "medium", "low", "info"],
                        help="Minimum severity to include in report (default: low)")
    parser.add_argument("--no-prescreen", action="store_true",
                        help="Send every file to LLM even if no risky patterns found")
    args = parser.parse_args()

    # ── API key ────────────────────────────────────────────────────────────────
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print(_red("Error: ANTHROPIC_API_KEY is required for static analysis."))
        print("  Set it with:  set ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except ImportError:
        print(_red("Error: anthropic package not installed."))
        print("  pip install anthropic")
        sys.exit(1)

    # ── Header ─────────────────────────────────────────────────────────────────
    print("=" * 70)
    print(_bold("  AI PENTEST ASSISTANT -- STATIC CODE SCANNER"))
    print("=" * 70)
    print(f"  Path     : {args.path}")
    print(f"  Language : {args.lang or 'all'}")
    print(f"  Model    : {args.model}")
    print(f"  Min sev  : {args.severity}")
    print("=" * 70)

    # ── Collect files ──────────────────────────────────────────────────────────
    files = collect_files(args.path, lang_filter=args.lang)
    if not files:
        print(_yellow("No supported source files found. Check --path and --lang."))
        sys.exit(0)

    print(f"\n  Found {len(files)} file(s) to analyse.\n")

    # ── Analyse each file ──────────────────────────────────────────────────────
    min_sev = SEVERITY_ORDER.get(args.severity, 3)
    all_findings: list[dict] = []
    skipped = 0

    for i, fpath in enumerate(files, 1):
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("Cannot read %s: %s", fpath, e)
            continue

        lang = LANGUAGE_MAP.get(fpath.suffix.lower(), "Unknown")
        rel = str(fpath)

        # Quick pre-screen
        if not args.no_prescreen and not quick_prescreen(content, lang):
            skipped += 1
            continue

        print(f"  [{i}/{len(files)}] Analysing {_cyan(rel)} ({lang}) ...", end=" ", flush=True)

        findings = analyse_file(fpath, content, lang, client, args.model)

        # Filter by severity
        findings = [
            f for f in findings
            if SEVERITY_ORDER.get(f.get("severity", "info").lower(), 4) <= min_sev
        ]

        if findings:
            print(_red(f"{len(findings)} issue(s) found"))
        else:
            print(_green("clean"))

        all_findings.extend(findings)

    # ── Sort by severity ───────────────────────────────────────────────────────
    all_findings.sort(key=lambda f: SEVERITY_ORDER.get(f.get("severity", "info").lower(), 4))

    # ── Print summary table ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(_bold(f"  RESULTS — {len(all_findings)} finding(s)  ({skipped} file(s) prescreened clean)"))
    print("=" * 70)

    if not all_findings:
        print(_green("  No vulnerabilities found."))
    else:
        col = severity_color
        for idx, f in enumerate(all_findings, 1):
            sev   = f.get("severity", "?").upper()
            cat   = f.get("category", "Unknown")
            file_ = Path(f.get("file", "?")).name
            line  = f.get("line", "?")
            desc  = f.get("description", "")
            fix   = f.get("remediation", "")
            color = SEVERITY_COLOR.get(sev.lower(), str)
            print(f"\n  {_bold(str(idx))}. {color(sev):12}  {_bold(cat)}")
            print(f"     File : {file_}  (line {line})")
            print(f"     Issue: {desc}")
            print(f"     Fix  : {fix}")
            snippet = f.get("snippet", "")
            if snippet:
                print(f"     Code : {snippet[:100]}")

    # Counts by severity
    print("\n  Severity breakdown:")
    for sev in ["critical", "high", "medium", "low", "info"]:
        count = sum(1 for f in all_findings if f.get("severity", "").lower() == sev)
        if count:
            color = SEVERITY_COLOR.get(sev, str)
            print(f"    {color(sev.upper()):12} {count}")

    # ── Save / merge report ────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "scan_type": "static_code_analysis",
        "timestamp": timestamp,
        "path": str(args.path),
        "files_scanned": len(files),
        "files_prescreened_clean": skipped,
        "total_findings": len(all_findings),
        "findings": all_findings,
    }

    # Merge into an existing dynamic scan report if requested
    if args.merge_report:
        try:
            with open(args.merge_report, "r") as f:
                existing = json.load(f)
            existing.setdefault("static_analysis", {})
            existing["static_analysis"] = report
            with open(args.merge_report, "w") as f:
                json.dump(existing, f, indent=2)
            print(f"\n  Static findings merged into: {args.merge_report}")
        except Exception as e:
            print(_yellow(f"\n  Warning: could not merge report — {e}"))

    if args.output:
        out_path = args.output
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {out_path}")
    elif all_findings:
        # Auto-save only when there are findings to report
        os.makedirs("reports", exist_ok=True)
        out_path = f"reports/code_scan_{timestamp}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {out_path}")
    else:
        print("\n  No findings — report not saved. Use --output to force saving.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
