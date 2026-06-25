"""End-to-end tests for the preprocessing & language-classification pipeline."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from youtube_sentiment.pipeline import PreprocessingPipeline

SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "sample_comments.csv")


@pytest.fixture(scope="module")
def run_result():
    pipeline = PreprocessingPipeline()
    return pipeline, pipeline.run(csv_path=SAMPLE_CSV)


def test_pipeline_runs_end_to_end(run_result):
    _, res = run_result
    assert res.total_comments > 0
    for col in ["cleaned_comment", "processed_text", "processed_text_no_stemming", "processed_text_with_stemming", "language"]:
        assert col in res.dataframe.columns


def test_pipeline_detects_all_languages(run_result):
    _, res = run_result
    detected = set(res.language_distribution)
    assert {"English", "Hindi", "Bengali", "Code-Mixed"}.issubset(detected)


def test_language_statistics_percentages(run_result):
    _, res = run_result
    total_pct = sum(row["percentage"] for row in res.language_stats)
    assert 99.0 <= total_pct <= 101.0  # allow rounding
    assert sum(res.language_distribution.values()) == res.total_comments


def test_pipeline_logs_recorded(run_result):
    _, res = run_result
    assert any("Cleaned" in line for line in res.logs)
    assert any("language" in line.lower() for line in res.logs)


def test_run_from_dataframe():
    df = pd.DataFrame({"comment": ["great video", "यह वीडियो अच्छा है", "এই গান দারুণ"]})
    res = PreprocessingPipeline().run(dataframe=df)
    assert res.total_comments == 3
    assert {"English", "Hindi", "Bengali"}.issubset(set(res.language_distribution))
    assert "processed_text_no_stemming" in res.dataframe.columns
    assert "processed_text_with_stemming" in res.dataframe.columns


def test_export_and_visualizations(run_result, tmp_path):
    pipeline, res = run_result
    export_path = tmp_path / "cleaned.csv"
    pipeline.export_cleaned(res, str(export_path))
    assert export_path.exists()
    exported = pd.read_csv(export_path)
    assert "language" in exported.columns and "cleaned_comment" in exported.columns
    assert "processed_text_no_stemming" in exported.columns
    assert "processed_text_with_stemming" in exported.columns

    paths = pipeline.generate_visualizations(res, str(tmp_path), theme="dark")
    for key in ["language_bar", "language_pie"]:
        assert os.path.exists(paths[key]) and os.path.getsize(paths[key]) > 0
