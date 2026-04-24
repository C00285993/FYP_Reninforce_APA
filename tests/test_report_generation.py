"""
Tests for the post-testing report generation pipeline.

Tests three layers:
  1. Report generator (utils/report_generator.py) — pure logic, no mocking needed
  2. Environment payload_info enrichment — mocks DVWA HTTP calls
  3. Evaluate loop integration — mocks model + env, verifies end-to-end flow
"""

import sys
import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.report_generator import PentestReportGenerator
from utils.response_analyzer import AnalysisResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DVWA_HTML = """
<html><body>
<div class="body_padded">
  <h2>Vulnerability: SQL Injection</h2>
  <pre>First name: admin<br>Surname: admin</pre>
  <pre>First name: Gordon<br>Surname: Brown</pre>
</div>
</body></html>
"""

SAMPLE_XSS_HTML = """
<html><body>
<div class="body_padded">
  <pre>Hello <script>alert(1)</script></pre>
</div>
</body></html>
"""


def _make_eval_results(vuln_type="sqli", num_episodes=3,
                       num_successes=2) -> dict:
    """Build a synthetic evaluation results dict that mirrors what
    evaluate_agent() produces, including step_details."""
    episodes = []
    for ep_num in range(1, num_episodes + 1):
        success = ep_num <= num_successes
        step_details = [
            {
                "step": 1,
                "action": "inject_single_quote" if vuln_type == "sqli" else "inject_basic_script",
                "action_id": 1,
                "reward": -1.0,
                "severity_score": 15,
                "payload": "'" if vuln_type == "sqli" else "<script>alert(1)</script>",
                "parameter": "id" if vuln_type == "sqli" else "name",
                "url_path": "/vulnerabilities/sqli/" if vuln_type == "sqli" else "/vulnerabilities/xss_r/",
                "full_request_url": "http://localhost:8080/vulnerabilities/sqli/?id=%27&Submit=Submit"
                    if vuln_type == "sqli"
                    else "http://localhost:8080/vulnerabilities/xss_r/?name=%3Cscript%3Ealert%281%29%3C%2Fscript%3E",
                "response_snippet": "<pre>Error in SQL syntax</pre>"
                    if vuln_type == "sqli"
                    else "<pre>Hello <script>alert(1)</script></pre>",
                "reflected": True,
            },
            {
                "step": 2,
                "action": "inject_union_select" if vuln_type == "sqli" else "inject_img_onerror",
                "action_id": 3 if vuln_type == "sqli" else 2,
                "reward": 50.0 if success else -1.0,
                "severity_score": 70 if success else 0,
                "payload": "' UNION SELECT user(),version()--"
                    if vuln_type == "sqli"
                    else '<img src=x onerror="alert(1)">',
                "parameter": "id" if vuln_type == "sqli" else "name",
                "url_path": "/vulnerabilities/sqli/" if vuln_type == "sqli" else "/vulnerabilities/xss_r/",
                "full_request_url": "http://localhost:8080/vulnerabilities/sqli/?id=%27+UNION+SELECT+user%28%29%2Cversion%28%29--&Submit=Submit"
                    if vuln_type == "sqli"
                    else "http://localhost:8080/vulnerabilities/xss_r/?name=%3Cimg+src%3Dx+onerror%3D%22alert%281%29%22%3E",
                "response_snippet": "<pre>First name: admin<br>Surname: admin</pre>"
                    if success
                    else "<pre></pre>",
                "reflected": success,
            },
        ]
        episodes.append({
            "episode": ep_num,
            "reward": 49.0 if success else -2.0,
            "steps": 2,
            "success": success,
            "actions": [1, 3] if vuln_type == "sqli" else [1, 2],
            "step_details": step_details,
        })

    return {
        "algo": "DQN",
        "num_episodes": num_episodes,
        "deterministic": True,
        "episodes": episodes,
        "summary": {
            "success_rate": num_successes / num_episodes,
            "total_successes": num_successes,
            "mean_reward": sum(
                49.0 if i < num_successes else -2.0
                for i in range(num_episodes)
            ) / num_episodes,
            "std_reward": 10.0,
            "median_reward": 49.0,
            "mean_steps": 2.0,
            "mean_steps_to_success": 2.0,
            "min_steps_to_success": 2,
            "action_distribution": {1: num_episodes, 3: num_episodes},
        },
    }


# ===================================================================
# 1. Report Generator unit tests
# ===================================================================

class TestReportGeneratorHeader:
    def test_header_contains_required_fields(self, tmp_path):
        results = _make_eval_results()
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        header = report["header"]

        assert header["target_url"] == "http://localhost:8080"
        assert header["vulnerability_type"] == "sqli"
        assert header["security_level"] == "low"
        assert header["algorithm"] == "DQN"
        assert "timestamp" in header


class TestReportGeneratorExecutiveSummary:
    def test_summary_stats(self, tmp_path):
        results = _make_eval_results(num_episodes=3, num_successes=2)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        summary = report["executive_summary"]

        assert summary["total_episodes"] == 3
        assert summary["success_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert summary["unique_payloads_tried"] >= 1
        assert summary["mean_steps"] == 2.0

    def test_zero_successes(self, tmp_path):
        results = _make_eval_results(num_episodes=2, num_successes=0)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        assert report["executive_summary"]["success_rate"] == 0.0


class TestReportGeneratorFindings:
    def test_findings_count_matches_successes(self, tmp_path):
        results = _make_eval_results(num_episodes=5, num_successes=3)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        assert len(report["findings"]) == 3

    def test_finding_has_payload_details(self, tmp_path):
        results = _make_eval_results(num_episodes=1, num_successes=1)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        finding = report["findings"][0]

        assert finding["payload"] != ""
        assert finding["parameter"] == "id"
        assert "full_request_url" in finding
        assert "response_snippet" in finding
        assert "severity_score" in finding
        assert "reflected" in finding

    def test_no_findings_when_no_successes(self, tmp_path):
        results = _make_eval_results(num_episodes=2, num_successes=0)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        assert report["findings"] == []


class TestReportGeneratorAllPayloads:
    def test_payloads_grouped_by_action(self, tmp_path):
        results = _make_eval_results()
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        all_payloads = report["all_payloads_tested"]

        # Should have entries grouped by action name
        assert isinstance(all_payloads, dict)
        assert len(all_payloads) > 0
        for action_name, entries in all_payloads.items():
            assert isinstance(entries, list)
            for entry in entries:
                assert "payload" in entry
                assert "reflected" in entry
                assert "full_request_url" in entry

    def test_payloads_are_deduplicated(self, tmp_path):
        # All episodes use the same payloads, so dedup should reduce count
        results = _make_eval_results(num_episodes=5, num_successes=3)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        all_payloads = report["all_payloads_tested"]

        total_entries = sum(len(v) for v in all_payloads.values())
        # 5 episodes x 2 steps = 10 steps, but only 2 unique (action, payload) pairs
        assert total_entries == 2


class TestReportGeneratorVulnerableParams:
    def test_parameter_stats(self, tmp_path):
        results = _make_eval_results()
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        params = report["vulnerable_parameters"]

        assert len(params) >= 1
        id_param = next(p for p in params if p["parameter"] == "id")
        assert id_param["total_attempts"] > 0
        assert id_param["confirmed_vulnerable"] is True


class TestReportGeneratorEvidence:
    def test_evidence_matches_findings(self, tmp_path):
        results = _make_eval_results(num_episodes=3, num_successes=2)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        evidence = report["request_response_evidence"]
        findings = report["findings"]

        assert len(evidence) == len(findings)
        for ev in evidence:
            assert "request_url" in ev
            assert "response_snippet" in ev
            assert "reflected" in ev


class TestReportGeneratorIO:
    def test_save_json(self, tmp_path):
        results = _make_eval_results()
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        filepath = gen.save_json(report)

        assert filepath.exists()
        assert filepath.suffix == ".json"
        loaded = json.loads(filepath.read_text())
        assert "header" in loaded
        assert "findings" in loaded

    def test_console_summary_does_not_crash(self, tmp_path, capsys):
        results = _make_eval_results()
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        gen.print_console_summary(report)

        captured = capsys.readouterr()
        assert "PENETRATION TEST REPORT" in captured.out
        assert "EXECUTIVE SUMMARY" in captured.out
        assert "FINDINGS" in captured.out

    def test_console_summary_with_xss(self, tmp_path, capsys):
        results = _make_eval_results(vuln_type="xss")
        gen = PentestReportGenerator(
            results, "xss", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        gen.print_console_summary(report)

        captured = capsys.readouterr()
        assert "PENETRATION TEST REPORT" in captured.out


# ===================================================================
# 2. Environment payload_info enrichment tests
# ===================================================================

class TestSQLiEnvPayloadInfo:
    """Test that SQLiEnv._execute_action returns correct payload_info."""

    @pytest.fixture
    def sqli_env(self):
        """Create a SQLiEnv with mocked DVWA client."""
        with patch("environments.base_env.DVWAClient") as MockClient, \
             patch("environments.base_env.ResponseAnalyzer") as MockAnalyzer, \
             patch("environments.base_env.FeatureExtractor") as MockExtractor:

            # Configure feature extractor
            mock_extractor = MockExtractor.return_value
            mock_extractor.SQLI_STATE_SIZE = 18
            mock_extractor.extract_sqli_state.return_value = np.zeros(18, dtype=np.float32)

            # Configure client
            mock_client = MockClient.return_value
            mock_client.ensure_ready.return_value = True
            mock_client.reset_for_episode.return_value = True
            mock_client.PAGES = {
                "sqli": "/vulnerabilities/sqli/",
                "xss_reflected": "/vulnerabilities/xss_r/",
            }
            mock_client.get_page.return_value = (SAMPLE_DVWA_HTML, 200)

            # submit_sqli returns (html, status, elapsed)
            mock_client.submit_sqli.return_value = (SAMPLE_DVWA_HTML, 200, 0.1)

            # Configure analyzer
            mock_analyzer_inst = MockAnalyzer.return_value
            analysis = AnalysisResult(
                has_data_leak=True,
                leaked_data_count=2,
                has_sql_error=True,
                payload_reflected=True,
                response_length=len(SAMPLE_DVWA_HTML),
            )
            mock_analyzer_inst.analyze_sqli_response.return_value = analysis
            mock_analyzer_inst.set_baseline.return_value = None

            from environments.sqli_env import SQLiEnv
            env = SQLiEnv(
                dvwa_url="http://localhost:8080",
                security_level="low",
                max_steps=10,
            )
            # Manually assign the mocked client's PAGES so _execute_action can access it
            env.client.PAGES = mock_client.PAGES
            env._baseline_html = SAMPLE_DVWA_HTML

            yield env

    def test_execute_action_returns_three_tuple(self, sqli_env):
        result = sqli_env._execute_action(1)  # inject_single_quote
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_payload_info_has_required_keys(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(2)  # inject_or_true
        required_keys = {"payload", "parameter", "url_path",
                         "full_request_url", "response_snippet", "reflected"}
        assert required_keys.issubset(payload_info.keys())

    def test_payload_info_parameter_is_id(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(3)
        assert payload_info["parameter"] == "id"

    def test_payload_info_url_contains_payload(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(1)
        assert "id=" in payload_info["full_request_url"]
        assert "Submit=Submit" in payload_info["full_request_url"]

    def test_payload_info_url_path(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(1)
        assert payload_info["url_path"] == "/vulnerabilities/sqli/"

    def test_report_done_returns_empty_payload_info(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(9)
        assert payload_info["payload"] == ""
        assert payload_info["full_request_url"] == ""

    def test_baseline_action_returns_payload_info(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(0)
        assert payload_info["parameter"] == "id"
        assert payload_info["payload"] != ""  # Random 1-5

    def test_response_snippet_extracted(self, sqli_env):
        _, _, payload_info = sqli_env._execute_action(3)
        # The sample HTML has <pre> tags, snippet should capture them
        assert len(payload_info["response_snippet"]) > 0


class TestXSSEnvPayloadInfo:
    """Test that XSSEnv._execute_action returns correct payload_info."""

    @pytest.fixture
    def xss_env(self):
        """Create an XSSEnv with mocked DVWA client."""
        with patch("environments.base_env.DVWAClient") as MockClient, \
             patch("environments.base_env.ResponseAnalyzer") as MockAnalyzer, \
             patch("environments.base_env.FeatureExtractor") as MockExtractor:

            mock_extractor = MockExtractor.return_value
            mock_extractor.XSS_STATE_SIZE = 20
            mock_extractor.extract_xss_state.return_value = np.zeros(20, dtype=np.float32)

            mock_client = MockClient.return_value
            mock_client.ensure_ready.return_value = True
            mock_client.reset_for_episode.return_value = True
            mock_client.PAGES = {
                "sqli": "/vulnerabilities/sqli/",
                "xss_reflected": "/vulnerabilities/xss_r/",
            }
            mock_client.get_page.return_value = (SAMPLE_XSS_HTML, 200)
            mock_client.submit_xss_reflected.return_value = (SAMPLE_XSS_HTML, 200, 0.05)

            mock_analyzer_inst = MockAnalyzer.return_value
            analysis = AnalysisResult(
                payload_reflected=True,
                script_tag_present=True,
                response_length=len(SAMPLE_XSS_HTML),
            )
            mock_analyzer_inst.analyze_xss_response.return_value = analysis
            mock_analyzer_inst.set_baseline.return_value = None

            from environments.xss_env import XSSEnv
            env = XSSEnv(
                dvwa_url="http://localhost:8080",
                security_level="low",
                max_steps=10,
            )
            env.client.PAGES = mock_client.PAGES
            env._baseline_html = SAMPLE_XSS_HTML

            yield env

    def test_execute_action_returns_three_tuple(self, xss_env):
        result = xss_env._execute_action(1)
        assert len(result) == 3

    def test_payload_info_parameter_is_name(self, xss_env):
        _, _, payload_info = xss_env._execute_action(1)
        assert payload_info["parameter"] == "name"

    def test_payload_info_url_path(self, xss_env):
        _, _, payload_info = xss_env._execute_action(2)
        assert payload_info["url_path"] == "/vulnerabilities/xss_r/"

    def test_payload_info_url_contains_name_param(self, xss_env):
        _, _, payload_info = xss_env._execute_action(1)
        assert "name=" in payload_info["full_request_url"]

    def test_report_done_returns_empty_payload_info(self, xss_env):
        _, _, payload_info = xss_env._execute_action(11)
        assert payload_info["payload"] == ""

    def test_reflected_flag_set(self, xss_env):
        _, _, payload_info = xss_env._execute_action(1)
        assert payload_info["reflected"] is True


# ===================================================================
# 3. Base env step() integration — payload_info in info and step_log
# ===================================================================

class TestBaseEnvStepIntegration:
    """Test that base_env.step() propagates payload_info correctly."""

    @pytest.fixture
    def sqli_env_with_reset(self):
        """Create a fully initialized (reset) SQLiEnv with mocked DVWA."""
        with patch("environments.base_env.DVWAClient") as MockClient, \
             patch("environments.base_env.ResponseAnalyzer") as MockAnalyzer, \
             patch("environments.base_env.FeatureExtractor") as MockExtractor:

            mock_extractor = MockExtractor.return_value
            mock_extractor.SQLI_STATE_SIZE = 18
            mock_extractor.extract_sqli_state.return_value = np.zeros(18, dtype=np.float32)

            mock_client = MockClient.return_value
            mock_client.ensure_ready.return_value = True
            mock_client.reset_for_episode.return_value = True
            mock_client.PAGES = {
                "sqli": "/vulnerabilities/sqli/",
                "xss_reflected": "/vulnerabilities/xss_r/",
            }
            mock_client.get_page.return_value = (SAMPLE_DVWA_HTML, 200)
            mock_client.submit_sqli.return_value = (SAMPLE_DVWA_HTML, 200, 0.1)

            mock_analyzer_inst = MockAnalyzer.return_value
            analysis = AnalysisResult(
                has_data_leak=True,
                leaked_data_count=2,
                has_sql_error=True,
                payload_reflected=True,
                response_length=len(SAMPLE_DVWA_HTML),
            )
            mock_analyzer_inst.analyze_sqli_response.return_value = analysis
            mock_analyzer_inst.set_baseline.return_value = None

            from environments.sqli_env import SQLiEnv
            env = SQLiEnv(
                dvwa_url="http://localhost:8080",
                security_level="low",
                max_steps=10,
            )
            env.client.PAGES = mock_client.PAGES

            env.reset()
            yield env

    def test_info_contains_payload_info(self, sqli_env_with_reset):
        _, _, _, _, info = sqli_env_with_reset.step(1)
        assert "payload_info" in info
        pi = info["payload_info"]
        assert "payload" in pi
        assert "parameter" in pi
        assert "full_request_url" in pi

    def test_step_log_enriched(self, sqli_env_with_reset):
        sqli_env_with_reset.step(2)
        log_entry = sqli_env_with_reset.episode_log[-1]

        assert "payload" in log_entry
        assert "parameter" in log_entry
        assert "full_request_url" in log_entry
        assert "response_snippet" in log_entry
        assert "reflected" in log_entry
        assert log_entry["parameter"] == "id"

    def test_step_log_payload_not_empty_for_injection(self, sqli_env_with_reset):
        sqli_env_with_reset.step(3)  # inject_union_select
        log_entry = sqli_env_with_reset.episode_log[-1]
        assert log_entry["payload"] != ""

    def test_multiple_steps_accumulate_logs(self, sqli_env_with_reset):
        sqli_env_with_reset.step(1)
        sqli_env_with_reset.step(2)
        assert len(sqli_env_with_reset.episode_log) == 2
        for log_entry in sqli_env_with_reset.episode_log:
            assert "payload" in log_entry
            assert "full_request_url" in log_entry


# ===================================================================
# 4. Evaluate loop integration test
# ===================================================================

class TestEvaluateIntegration:
    """Test that evaluate_agent collects step_details with payload data."""

    def test_evaluate_agent_collects_step_details(self):
        """Mock model + env to verify evaluate_agent output structure."""
        from agents.evaluate import evaluate_agent

        # Mock environment
        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(18, dtype=np.float32), {})
        mock_env.action_space = MagicMock()

        # Make env.step return terminated=True on 2nd step to end episode
        call_count = [0]
        def mock_step(action):
            call_count[0] += 1
            terminated = call_count[0] % 2 == 0  # terminate every 2nd step
            return (
                np.zeros(18, dtype=np.float32),
                10.0 if terminated else -1.0,
                terminated,
                False,
                {
                    "step": call_count[0],
                    "action_name": "inject_or_true",
                    "severity_score": 70 if terminated else 15,
                    "episode_reward": 9.0,
                    "payload_info": {
                        "payload": "' OR 1=1 --",
                        "parameter": "id",
                        "url_path": "/vulnerabilities/sqli/",
                        "full_request_url": "http://localhost:8080/vulnerabilities/sqli/?id=%27+OR+1%3D1+--&Submit=Submit",
                        "response_snippet": "<pre>First name: admin</pre>",
                        "reflected": True,
                    },
                },
            )

        mock_env.step.side_effect = mock_step

        # Mock model
        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(2), None)

        results = evaluate_agent(mock_model, mock_env, "DQN", num_episodes=2)

        # Verify structure
        assert len(results["episodes"]) == 2
        for ep in results["episodes"]:
            assert "step_details" in ep
            assert len(ep["step_details"]) > 0
            for sd in ep["step_details"]:
                assert "payload" in sd
                assert "parameter" in sd
                assert "full_request_url" in sd
                assert "response_snippet" in sd
                assert "reflected" in sd

    def test_evaluate_step_details_payload_values(self):
        """Verify actual payload values are captured, not empty strings."""
        from agents.evaluate import evaluate_agent

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(18, dtype=np.float32), {})

        # Single step, immediate termination
        mock_env.step.return_value = (
            np.zeros(18, dtype=np.float32),
            50.0,
            True,  # terminated
            False,
            {
                "step": 1,
                "action_name": "inject_union_select",
                "severity_score": 70,
                "episode_reward": 50.0,
                "payload_info": {
                    "payload": "' UNION SELECT 1,2--",
                    "parameter": "id",
                    "url_path": "/vulnerabilities/sqli/",
                    "full_request_url": "http://localhost:8080/vulnerabilities/sqli/?id=%27+UNION+SELECT+1%2C2--&Submit=Submit",
                    "response_snippet": "<pre>First name: 1<br>Surname: 2</pre>",
                    "reflected": True,
                },
            },
        )

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array(3), None)

        results = evaluate_agent(mock_model, mock_env, "DQN", num_episodes=1)
        sd = results["episodes"][0]["step_details"][0]

        assert sd["payload"] == "' UNION SELECT 1,2--"
        assert sd["parameter"] == "id"
        assert sd["reflected"] is True
        assert "UNION" in sd["full_request_url"]


# ===================================================================
# 5. End-to-end: evaluate → report generator
# ===================================================================

class TestEndToEnd:
    """Test the full pipeline: eval results → report generation → JSON output."""

    def test_full_pipeline_sqli(self, tmp_path):
        results = _make_eval_results(vuln_type="sqli", num_episodes=5, num_successes=3)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        filepath = gen.save_json(report)

        # Load and validate the full report
        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["header"]["vulnerability_type"] == "sqli"
        assert len(loaded["findings"]) == 3
        assert loaded["executive_summary"]["success_rate"] == pytest.approx(0.6, abs=0.01)
        assert len(loaded["request_response_evidence"]) == 3
        assert any(p["confirmed_vulnerable"] for p in loaded["vulnerable_parameters"])

    def test_full_pipeline_xss(self, tmp_path):
        results = _make_eval_results(vuln_type="xss", num_episodes=4, num_successes=2)
        gen = PentestReportGenerator(
            results, "xss", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        filepath = gen.save_json(report)

        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["header"]["vulnerability_type"] == "xss"
        assert len(loaded["findings"]) == 2
        # Check XSS-specific parameter
        params = loaded["vulnerable_parameters"]
        assert any(p["parameter"] == "name" for p in params)

    def test_report_with_zero_successes(self, tmp_path):
        results = _make_eval_results(num_episodes=3, num_successes=0)
        gen = PentestReportGenerator(
            results, "sqli", "http://localhost:8080", "low",
            output_dir=str(tmp_path),
        )
        report = gen.generate()
        filepath = gen.save_json(report)

        with open(filepath) as f:
            loaded = json.load(f)

        assert loaded["findings"] == []
        assert loaded["request_response_evidence"] == []
        assert loaded["executive_summary"]["success_rate"] == 0.0
