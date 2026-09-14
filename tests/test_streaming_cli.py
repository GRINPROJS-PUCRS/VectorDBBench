from click.testing import CliRunner
from pytest import MonkeyPatch

from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.test import cli as test_cli
from vectordb_bench.cli import cli as common_cli


def invoke_test_command(monkeypatch: MonkeyPatch, args: list[str]):
    captured = {}

    def fake_run(tasks, task_label):
        captured["task"] = tasks[0]
        captured["task_label"] = task_label

    monkeypatch.setattr(common_cli.benchmark_runner, "run", fake_run)
    monkeypatch.setattr(common_cli.benchmark_runner, "has_running", lambda: False)
    result = CliRunner().invoke(test_cli.Test, args)
    return result, captured


def test_common_cli_exposes_streaming_options() -> None:
    result = CliRunner().invoke(test_cli.Test, ["--help"])

    assert result.exit_code == 0, result.output
    assert "--insert-rate INTEGER" in result.output
    assert "--search-stages TEXT" in result.output
    assert "--streaming-concurrencies TEXT" in result.output


def test_streaming_case_type_accepted_with_defaults(monkeypatch: MonkeyPatch) -> None:
    result, captured = invoke_test_command(
        monkeypatch,
        ["--case-type", "StreamingPerformanceCase"],
    )

    assert result.exit_code == 0, result.output
    case_config = captured["task"].case_config
    assert case_config.case_id == CaseType.StreamingPerformanceCase
    assert case_config.custom_case["insert_rate"] == 500
    assert case_config.custom_case["search_stages"] == [0.5, 0.8]
    assert case_config.custom_case["concurrencies"] == [5, 10]


def test_streaming_case_type_accepts_overrides(monkeypatch: MonkeyPatch) -> None:
    result, captured = invoke_test_command(
        monkeypatch,
        [
            "--case-type",
            "StreamingPerformanceCase",
            "--insert-rate",
            "1000",
            "--search-stages",
            "0.3,0.6,0.9",
            "--streaming-concurrencies",
            "1,2,4",
        ],
    )

    assert result.exit_code == 0, result.output
    custom_case = captured["task"].case_config.custom_case
    assert custom_case["insert_rate"] == 1000
    assert custom_case["search_stages"] == [0.3, 0.6, 0.9]
    assert custom_case["concurrencies"] == [1, 2, 4]


def test_non_streaming_case_type_ignores_streaming_options(monkeypatch: MonkeyPatch) -> None:
    result, captured = invoke_test_command(
        monkeypatch,
        ["--case-type", "Performance1536D50K", "--insert-rate", "999"],
    )

    assert result.exit_code == 0, result.output
    assert captured["task"].case_config.custom_case == {}
