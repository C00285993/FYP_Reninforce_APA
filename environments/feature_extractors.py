"""
Feature Extractor
Converts raw HTML responses and analysis results into fixed-size
numerical state vectors that the RL agent (neural network) can process.

This is the "eyes" of the agent — it determines what the agent perceives.
"""

import json as _json
import numpy as np
from bs4 import BeautifulSoup
from utils.response_analyzer import AnalysisResult
import re

# Unified observation size shared by all SQLi environments.
# Both DVWA (HTML) and Juice Shop (JSON) environments produce this
# same 18-dim vector so a single model can transfer between targets.
UNIFIED_SQLI_STATE_SIZE = 18

# Unified observation size shared by all XSS environments.
# Both DVWA (HTML) and Juice Shop (JSON) environments produce this
# same 20-dim vector so a single model can transfer between targets.
UNIFIED_XSS_STATE_SIZE = 20

# Unified observation size for CMDi environments.
# Single training target (DVWA) for now; same 18-dim layout as SQLi
# for architectural consistency.
UNIFIED_CMDI_STATE_SIZE = 18

# Unified observation size for SSTI environments.
# Juice Shop-only (Pug template injection on /profile).
# Same 18-dim layout for architectural consistency.
UNIFIED_SSTI_STATE_SIZE = 18


def extract_unified_sqli_state(
    response_text: str,
    analysis: AnalysisResult,
    agent_memory: dict,
    response_format: str = "html",
) -> np.ndarray:
    """
    Produce a target-agnostic 18-dim observation vector for SQLi environments.

    The first 5 features are parsed differently depending on *response_format*
    ("html" for DVWA, "json" for Juice Shop) but carry the same semantic
    meaning so that a model trained on one target can transfer to another.

    Vector layout:
        [0]  has_structured_data   – 1.0 if HTML table/pre OR JSON data array
        [1]  data_item_count / 50  – normalised row / record count
        [2]  has_error_indicator   – 1.0 if error message in any format
        [3]  has_auth_indicator    – 1.0 if auth token OR login bypass detected
        [4]  response_length / 10000

        [5]  has_sql_error
        [6]  has_data_leak
        [7]  leaked_data_count / 10
        [8]  auth_bypass
        [9]  response_differs
        [10] error_page
        [11] response_time / 10

        [12] attempts_this_field / 20
        [13] total_attempts / 50
        [14] last_payload_category / 10
        [15] found_sql_error
        [16] found_data_leak
        [17] unique_responses_seen / 10
    """
    # --- Features 0-4: response-level (parsed per format) ----------------
    if response_format == "json":
        # Juice Shop REST API — JSON with "data" array
        has_structured_data = 0.0
        data_item_count = 0
        has_error_indicator = 0.0
        has_auth_indicator = 0.0
        try:
            data = _json.loads(response_text)
            if isinstance(data, dict):
                rows = data.get("data", [])
                if isinstance(rows, list) and len(rows) > 0:
                    has_structured_data = 1.0
                    data_item_count = len(rows)
                has_error_indicator = float(
                    "error" in data or "message" in data
                )
                has_auth_indicator = float(
                    bool(data.get("authentication", {}).get("token"))
                )
        except (ValueError, TypeError):
            pass
    elif response_format == "webgoat":
        # WebGoat lesson API — JSON with lessonCompleted + output (HTML)
        has_structured_data = 0.0
        data_item_count = 0
        has_error_indicator = 0.0
        has_auth_indicator = 0.0
        try:
            data = _json.loads(response_text)
            if isinstance(data, dict):
                lesson_completed = bool(data.get("lessonCompleted", False))
                has_auth_indicator = float(lesson_completed)
                # assignment5a: success via lessonCompleted flag
                if lesson_completed:
                    has_structured_data = 1.0
                    data_item_count = 6  # WebGoat user_data has 6 rows
                else:
                    # Fallback: check output HTML for table rows
                    output_html = data.get("output", "") or ""
                    if output_html:
                        from bs4 import BeautifulSoup as _BS
                        soup = _BS(output_html, "lxml")
                        table = soup.find("table")
                        if table:
                            rows = table.find_all("tr")
                            data_rows = max(len(rows) - 1, 0)
                            if data_rows > 0:
                                has_structured_data = 1.0
                                data_item_count = data_rows
                feedback = (data.get("feedback", "") or "").lower()
                has_error_indicator = float(
                    "error" in feedback or "exception" in feedback
                    or re.search(r"org\.h2\.", response_text, re.I) is not None
                )
        except (ValueError, TypeError):
            pass
    else:  # html
        soup = BeautifulSoup(response_text, "lxml")
        table = soup.find("table")
        pre = soup.find("pre")
        has_structured_data = float(bool(table or pre))
        # Count rows in table or <br> delimited entries in <pre>
        data_item_count = 0
        if table:
            data_item_count = max(len(table.find_all("tr")) - 1, 0)
        elif pre:
            data_item_count = max(len(pre.get_text().strip().split("\n")), 0)
        has_error_indicator = float(
            bool(re.search(r"error|warning|fatal|exception", response_text, re.I))
        )
        has_auth_indicator = float(
            bool(re.search(r"admin|logged\s*in|welcome", response_text, re.I))
        )

    # --- Features 5-11: analysis results (identical across targets) ------
    # --- Features 12-17: agent memory ------------------------------------
    features = [
        # 0-4 response-level
        has_structured_data,
        min(data_item_count / 50.0, 1.0),
        has_error_indicator,
        has_auth_indicator,
        min(len(response_text) / 10000.0, 1.0),
        # 5-11 analysis
        float(analysis.has_sql_error),
        float(analysis.has_data_leak),
        min(analysis.leaked_data_count / 10.0, 1.0),
        float(analysis.auth_bypass),
        float(analysis.response_differs),
        float(analysis.error_page),
        min(analysis.response_time / 10.0, 1.0),
        # 12-17 agent memory
        min(agent_memory.get("attempts_this_field", 0) / 20.0, 1.0),
        min(agent_memory.get("total_attempts", 0) / 50.0, 1.0),
        agent_memory.get("last_payload_category", 0) / 10.0,
        float(agent_memory.get("found_sql_error", False)),
        float(agent_memory.get("found_data_leak", False)),
        min(agent_memory.get("unique_responses_seen", 0) / 10.0, 1.0),
    ]

    assert len(features) == UNIFIED_SQLI_STATE_SIZE, (
        f"Expected {UNIFIED_SQLI_STATE_SIZE} features, got {len(features)}"
    )
    return np.array(features, dtype=np.float32)


def extract_unified_xss_state(
    response_text: str,
    analysis: AnalysisResult,
    agent_memory: dict,
    response_format: str = "html",
) -> np.ndarray:
    """
    Produce a target-agnostic 20-dim observation vector for XSS environments.

    The first 6 features (page features) are parsed differently depending on
    *response_format* ("html" for DVWA, "json" for Juice Shop) but carry the
    same semantic meaning so a model trained on one target can transfer.

    Vector layout:
        [0]  input_field_count / 10   – normalised count of input fields
        [1]  has_text_input           – 1.0 if text input present
        [2]  has_textarea             – 1.0 if textarea present
        [3]  form_method_is_get       – 1.0 if GET form present
        [4]  link_count / 50          – normalised link count
        [5]  has_output_area          – 1.0 if output/reflection area exists

        [6]  payload_reflected
        [7]  script_tag_present
        [8]  event_handler_present
        [9]  response_differs
        [10] response_length / 10000
        [11] error_page
        [12] xss_context_html         – 1.0 if context is "html"
        [13] xss_context_attribute    – 1.0 if context is "attribute"

        [14] attempts_this_field / 20
        [15] total_attempts / 50
        [16] last_payload_category / 10
        [17] found_reflection
        [18] found_xss_execution
        [19] unique_responses_seen / 10
    """
    # --- Features 0-5: page-level (parsed per format) ---------------------
    if response_format == "json":
        # Juice Shop returns JSON from REST APIs — no HTML structure
        # Use constants that represent "search endpoint with reflection area"
        input_field_count = 0.1   # search bar exists but not in response
        has_text_input = 1.0      # search input exists on the SPA
        has_textarea = 0.0
        form_method_is_get = 1.0  # search is GET-based
        link_count = 0.0          # no links in JSON response
        has_output_area = 1.0     # search results are the output area
    elif response_format == "webgoat":
        # WebGoat XSS Shopping Cart lesson — GET form with 6 fields, receipt output
        try:
            data = _json.loads(response_text)
            lesson_completed = bool(data.get("lessonCompleted", False))
        except (ValueError, TypeError):
            lesson_completed = False
        # Shopping cart form: 6 inputs (QTY1-4, field1, field2), GET-based, output area present
        input_field_count = min(6 / 10.0, 1.0)
        has_text_input = 1.0
        has_textarea = 0.0
        form_method_is_get = 1.0
        link_count = 0.0
        has_output_area = 1.0
        # If the lesson is complete, ensure memory reflects XSS success
        if lesson_completed:
            agent_memory["found_xss_execution"] = True
    else:  # html
        soup = BeautifulSoup(response_text, "lxml")
        # Count real input fields
        inputs = soup.find_all(["input", "textarea"])
        real_inputs = [i for i in inputs
                       if i.get("type", "text") not in ("hidden", "submit")
                       and i.get("name", "") != "user_token"]
        input_field_count = min(len(real_inputs) / 10.0, 1.0)
        has_text_input = float(len(soup.find_all("input", {"type": "text"})) > 0)
        has_textarea = float(len(soup.find_all("textarea")) > 0)
        # Check form method
        form_method_is_get = 0.0
        for form in soup.find_all("form"):
            if form.get("method", "GET").upper() == "GET":
                form_method_is_get = 1.0
                break
        link_count = min(len(soup.find_all("a", href=True)) / 50.0, 1.0)
        # Check for any output/result area (generic — not just DVWA)
        output_area = soup.find("pre") or soup.find("code")
        if not output_area:
            for div in soup.find_all("div"):
                cls = " ".join(div.get("class", []))
                div_id = div.get("id", "")
                if re.search(r"result|output|response|vulnerable|content|display",
                             cls + " " + div_id, re.I):
                    output_area = div
                    break
        has_output_area = float(bool(output_area))

    # --- Features 6-13: analysis results (identical across targets) -------
    # --- Features 14-19: agent memory -------------------------------------
    features = [
        # 0-5 page-level
        input_field_count,
        has_text_input,
        has_textarea,
        form_method_is_get,
        link_count,
        has_output_area,
        # 6-13 analysis
        float(analysis.payload_reflected),
        float(analysis.script_tag_present),
        float(analysis.event_handler_present),
        float(analysis.response_differs),
        min(analysis.response_length / 10000.0, 1.0),
        float(analysis.error_page),
        float(analysis.xss_context == "html"),
        float(analysis.xss_context == "attribute"),
        # 14-19 agent memory
        min(agent_memory.get("attempts_this_field", 0) / 20.0, 1.0),
        min(agent_memory.get("total_attempts", 0) / 50.0, 1.0),
        agent_memory.get("last_payload_category", 0) / 10.0,
        float(agent_memory.get("found_reflection", False)),
        float(agent_memory.get("found_xss_execution", False)),
        min(agent_memory.get("unique_responses_seen", 0) / 10.0, 1.0),
    ]

    assert len(features) == UNIFIED_XSS_STATE_SIZE, (
        f"Expected {UNIFIED_XSS_STATE_SIZE} features, got {len(features)}"
    )
    return np.array(features, dtype=np.float32)


def extract_unified_cmdi_state(
    response_text: str,
    analysis: AnalysisResult,
    agent_memory: dict,
    response_format: str = "html",
) -> np.ndarray:
    """
    Produce a target-agnostic 18-dim observation vector for CMDi environments.

    *response_format* controls how command output is located in the page:
      - "html"    : DVWA — output in <pre> tags inside .body_padded
      - "webgoat" : WebGoat — output in <div class="lesson-content"> areas
      - other     : plain-text fallback (count lines directly)

    Vector layout:
        [0]  pre_tag_present          – 1.0 if output area exists (pre/div)
        [1]  command_output_lines/10  – normalised line count in output area
        [2]  has_command_execution    – from analysis
        [3]  file_operations_detected – from analysis
        [4]  response_length/10000
        [5]  response_differs
        [6]  error_page
        [7]  response_time/10
        [8]  shell_error_present      – shell_error_type != ""
        [9]  uid_pattern_found        – direct re check on text
        [10] etc_passwd_found         – direct re check on text
        [11] ls_output_found          – drwx or -rwx pattern
        [12] attempts_this_field/20
        [13] total_attempts/50
        [14] last_payload_category/10
        [15] found_command_exec       – agent memory flag
        [16] found_file_operations    – agent memory flag
        [17] unique_responses_seen/10
    """
    # --- Features 0-1: response structure ---------------------------------
    # Generic: search <pre>, <code>, <samp>, and common output divs on any site
    soup = BeautifulSoup(response_text, "lxml")
    output_text = ""

    # Collect text from all output-like elements
    for elem in soup.find_all(["pre", "code", "samp", "tt"]):
        output_text += elem.get_text() + "\n"

    # Also check divs that look like output/result areas
    for div in soup.find_all("div"):
        cls = " ".join(div.get("class", []))
        div_id = div.get("id", "")
        if re.search(r"result|output|response|content|body_padded|command|lesson",
                     cls + " " + div_id, re.I):
            output_text += div.get_text() + "\n"

    pre_tag_present = float(bool(output_text.strip()))
    pre_text = output_text if output_text.strip() else soup.get_text(separator="\n")
    cmd_lines = len([l for l in pre_text.splitlines() if l.strip()])

    # --- Features 9-11: direct pattern checks on response text -----------
    uid_found = float(bool(re.search(r"uid=\d+", pre_text or response_text)))
    etc_passwd_found = float(bool(re.search(r"root:\S*:0:0:", pre_text or response_text)))
    ls_output_found = float(bool(re.search(r"drwx|(?<!\S)-rwx", pre_text or response_text)))

    features = [
        # 0-7 response-level
        pre_tag_present,
        min(cmd_lines / 10.0, 1.0),
        float(analysis.has_command_execution),
        float(analysis.file_operations_detected),
        min(len(response_text) / 10000.0, 1.0),
        float(analysis.response_differs),
        float(analysis.error_page),
        min(analysis.response_time / 10.0, 1.0),
        # 8-11 shell / command output signals
        float(analysis.shell_error_type != ""),
        uid_found,
        etc_passwd_found,
        ls_output_found,
        # 12-17 agent memory
        min(agent_memory.get("attempts_this_field", 0) / 20.0, 1.0),
        min(agent_memory.get("total_attempts", 0) / 50.0, 1.0),
        agent_memory.get("last_payload_category", 0) / 10.0,
        float(agent_memory.get("found_command_exec", False)),
        float(agent_memory.get("found_file_operations", False)),
        min(agent_memory.get("unique_responses_seen", 0) / 10.0, 1.0),
    ]

    assert len(features) == UNIFIED_CMDI_STATE_SIZE, (
        f"Expected {UNIFIED_CMDI_STATE_SIZE} features, got {len(features)}"
    )
    return np.array(features, dtype=np.float32)


def extract_unified_ssti_state(
    response_text: str,
    analysis: AnalysisResult,
    agent_memory: dict,
    response_format: str = "html",
) -> np.ndarray:
    """
    Produce an 18-dim observation vector for SSTI environments.

    The response is always the GET /profile HTML page rendered by Pug.
    The response_format parameter is kept for API consistency.

    Vector layout:
        [0]  expression_evaluated      – arithmetic/string eval confirmed
        [1]  nodejs_info_leaked        – version / env var / platform present
        [2]  has_command_execution     – uid= / hostname / cmd output
        [3]  waf_blocked               – WAF returned 400/403 or "Blocked"
        [4]  response_length/10000
        [5]  response_differs          – differs from baseline benign profile
        [6]  error_page                – 500+ status
        [7]  response_time/10
        [8]  status_200                – profile page loaded successfully
        [9]  payload_reflected_raw     – #{...} appeared unevaluated (Pug didn't process it)
        [10] profile_page_rendered     – page has reasonable content (non-empty body)
        [11] shell_error_present       – shell error string in response
        [12] attempts_this_field/20
        [13] total_attempts/50
        [14] last_payload_category/10
        [15] found_eval                – agent memory flag
        [16] found_rce                 – agent memory flag
        [17] unique_responses_seen/10
    """
    # --- Profile-page-specific features (parse HTML) ----------------------
    payload_reflected_raw = 0.0
    profile_page_rendered = 0.0

    if response_text:
        soup = BeautifulSoup(response_text, "lxml")
        page_text = soup.get_text(strip=True)

        if response_format == "twig":
            # {{...}} appearing raw means Twig did NOT evaluate it (bad sign)
            if "{{" in page_text:
                payload_reflected_raw = 1.0
        elif response_format == "pug":
            # #{...} appearing raw means Pug didn't evaluate it
            if "#{" in page_text:
                payload_reflected_raw = 1.0
        else:
            # Generic: check both template syntaxes
            if "#{" in page_text or "{{" in page_text:
                payload_reflected_raw = 1.0

        # Page rendered if it has reasonable body content (generic check)
        profile_page_rendered = float(len(page_text) > 100)

    status_200 = float(analysis.status_code == 200)
    shell_error = float(bool(analysis.shell_error_type))

    features = [
        # 0-3 SSTI detection signals
        float(analysis.expression_evaluated),
        float(analysis.nodejs_info_leaked),
        float(analysis.has_command_execution),
        float(analysis.waf_blocked),
        # 4-7 general response metrics
        min(analysis.response_length / 10000.0, 1.0),
        float(analysis.response_differs),
        float(analysis.error_page),
        min(analysis.response_time / 10.0, 1.0),
        # 8-11 page-specific features
        status_200,
        payload_reflected_raw,
        profile_page_rendered,
        shell_error,
        # 12-17 agent memory
        min(agent_memory.get("attempts_this_field", 0) / 20.0, 1.0),
        min(agent_memory.get("total_attempts", 0) / 50.0, 1.0),
        agent_memory.get("last_payload_category", 0) / 10.0,
        float(agent_memory.get("found_eval", False)),
        float(agent_memory.get("found_rce", False)),
        min(agent_memory.get("unique_responses_seen", 0) / 10.0, 1.0),
    ]

    assert len(features) == UNIFIED_SSTI_STATE_SIZE, (
        f"Expected {UNIFIED_SSTI_STATE_SIZE} features, got {len(features)}"
    )
    return np.array(features, dtype=np.float32)


class FeatureExtractor:
    """Extracts a fixed-size numerical feature vector from web page state."""

    # Total number of features in the state vector
    # Adjust this if you add/remove features
    SQLI_STATE_SIZE = UNIFIED_SQLI_STATE_SIZE
    JUICESHOP_SQLI_STATE_SIZE = UNIFIED_SQLI_STATE_SIZE
    XSS_STATE_SIZE = UNIFIED_XSS_STATE_SIZE
    JUICESHOP_XSS_STATE_SIZE = UNIFIED_XSS_STATE_SIZE
    CMDI_STATE_SIZE = UNIFIED_CMDI_STATE_SIZE
    SSTI_STATE_SIZE = UNIFIED_SSTI_STATE_SIZE

    def extract_sqli_state(self, html: str, analysis: AnalysisResult,
                            agent_memory: dict) -> np.ndarray:
        """
        Convert current state into a feature vector for the SQLi environment.
        Delegates to the unified extractor with response_format="html".
        """
        return extract_unified_sqli_state(
            html, analysis, agent_memory, response_format="html"
        )

    def extract_juiceshop_sqli_state(
        self,
        response_text: str,
        analysis: AnalysisResult,
        agent_memory: dict,
    ) -> np.ndarray:
        """
        Convert current state into a feature vector for the Juice Shop SQLi env.
        Delegates to the unified extractor with response_format="json".
        """
        return extract_unified_sqli_state(
            response_text, analysis, agent_memory, response_format="json"
        )

    def extract_xss_state(self, html: str, analysis: AnalysisResult,
                           agent_memory: dict) -> np.ndarray:
        """
        Convert current state into a feature vector for the DVWA XSS environment.
        Delegates to the unified extractor with response_format="html".
        """
        return extract_unified_xss_state(
            html, analysis, agent_memory, response_format="html"
        )

    def extract_juiceshop_xss_state(
        self,
        response_text: str,
        analysis: AnalysisResult,
        agent_memory: dict,
    ) -> np.ndarray:
        """
        Convert current state into a feature vector for the Juice Shop XSS env.
        Delegates to the unified extractor with response_format="json".
        """
        return extract_unified_xss_state(
            response_text, analysis, agent_memory, response_format="json"
        )

    # ---- HTML parsing helpers ----

    @staticmethod
    def _count_input_fields(soup: BeautifulSoup) -> float:
        """Count input fields, normalized."""
        inputs = soup.find_all(["input", "textarea"])
        # Exclude hidden/submit/token fields
        real_inputs = [i for i in inputs
                       if i.get("type", "text") not in ("hidden", "submit")
                       and i.get("name", "") != "user_token"]
        return min(len(real_inputs) / 10.0, 1.0)

    @staticmethod
    def _has_text_input(soup: BeautifulSoup) -> bool:
        inputs = soup.find_all("input", {"type": "text"})
        return len(inputs) > 0

    @staticmethod
    def _has_password_input(soup: BeautifulSoup) -> bool:
        inputs = soup.find_all("input", {"type": "password"})
        return len(inputs) > 0

    @staticmethod
    def _has_textarea(soup: BeautifulSoup) -> bool:
        return len(soup.find_all("textarea")) > 0

    @staticmethod
    def _form_method_is_get(soup: BeautifulSoup) -> bool:
        forms = soup.find_all("form")
        for form in forms:
            if form.get("method", "GET").upper() == "GET":
                return True
        return False

    @staticmethod
    def _count_links(soup: BeautifulSoup) -> int:
        return len(soup.find_all("a", href=True))

    @staticmethod
    def _has_output_area(soup: BeautifulSoup) -> bool:
        """Check if page has an area where output/reflection would appear."""
        # DVWA shows output in <pre> tags or specific divs
        return bool(soup.find("pre") or soup.find("div", {"class": "vulnerable_code_area"}))
