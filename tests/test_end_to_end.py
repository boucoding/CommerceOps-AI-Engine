"""
Full pipeline over the bundled sample data, scored against evals/gold_labels.json.
"""

import json

import pandas as pd

from src.config import LLMMode
from src.evaluation import run_evaluation
from src.llm.replay import ReplayLLMClient
from src.utils.reporter import build_report, export_report
from tests.conftest import make_settings


def test_replay_fixtures_cover_the_pipeline_exactly():
    """Fails when prompts or retrieval change without re-running scripts/build_reference_fixtures.py."""
    result = run_evaluation(make_settings(BATCH_WORKERS=4), LLMMode.REPLAY)
    assert result.counts["degraded (fell back to rules/templates)"] == 0, result.failures
    assert result.metrics["replies without placeholder leaks"] == 1.0
    if result.replay_source == "hand-authored":
        # Reference answers encode the gold labels, so anything short of perfect is a pipeline bug.
        assert result.failures == []
        assert all(score == 1.0 for score in result.metrics.values())


def test_offline_rules_meet_the_quality_bar():
    result = run_evaluation(make_settings(), LLMMode.OFFLINE)
    assert result.metrics["intent accuracy"] >= 0.95
    assert result.metrics["order line recall"] >= 0.95
    assert result.metrics["replies without placeholder leaks"] == 1.0


def test_fixture_file_is_labelled_honestly():
    replay = ReplayLLMClient("fixtures/llm_replays.json", 1536)
    assert replay.source in {"hand-authored", "recorded"}
    with open("fixtures/llm_replays.json", encoding="utf-8") as fh:
        assert json.load(fh)["model"] == replay.model


def test_report_export_keeps_the_four_required_sheets(build, tmp_path):
    from tests.conftest import email

    engine = build()
    outcomes = engine.orchestrator.process_batch(
        [email("E1", "Please send me 1 SWL2345."), email("E2", "What material is the Leather Tote?")]
    )
    paths = export_report(build_report(outcomes), str(tmp_path))
    sheets = pd.read_excel(paths["workbook"], sheet_name=None)
    assert list(sheets) == [
        "email-classification",
        "order-status",
        "order-response",
        "inquiry-response",
        "processing-audit",
    ]
    assert list(sheets["order-status"].columns) == ["email ID", "product ID", "quantity", "status"]
    assert sheets["order-status"].iloc[0].tolist() == ["E1", "SWL2345", 1, "created"]
    assert list(sheets["inquiry-response"]["email ID"]) == ["E2"]
