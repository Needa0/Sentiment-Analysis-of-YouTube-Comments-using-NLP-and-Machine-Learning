"""Preprocessing & language-classification pipeline.

Orchestrates the in-scope phases of the workflow:

    Dataset Collection / CSV Upload -> Cleaning & Normalisation ->
    NLP Preprocessing (tokenise, stop-words, stemming, lemmatisation) ->
    Language Detection & Classification -> Language Statistics -> Export.

The pipeline records human-readable log messages for each stage and can export
the cleaned, language-annotated dataset to CSV.
"""

from __future__ import annotations

import datetime as _dt
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from . import visualization as viz
from .cleaning import clean_dataframe
from .data_collection import load_comments_csv
from .language_detection import LanguageDetector
from .preprocessing import NLPPreprocessor

EXPORT_COLUMNS = [
    "comment_id",
    "video_id",
    "author",
    "comment",
    "cleaned_comment",
    "processed_text",
    "processed_text_no_stemming",
    "processed_text_with_stemming",
    "language",
]


@dataclass
class PipelineResult:
    """Artefacts produced by a pipeline run."""

    dataframe: pd.DataFrame
    language_distribution: Dict[str, int] = field(default_factory=dict)
    language_stats: List[Dict[str, object]] = field(default_factory=list)
    total_comments: int = 0
    logs: List[str] = field(default_factory=list)
    preprocessing_report: Dict[str, object] = field(default_factory=dict)


class PreprocessingPipeline:
    """Run cleaning, NLP preprocessing and language classification."""

    def __init__(self, use_stemming: bool = False):
        self.use_stemming = use_stemming
        self.language_detector = LanguageDetector()
        self.preprocessor = NLPPreprocessor()
        self.logs: List[str] = []

    # --- logging ------------------------------------------------------------
    def _log(self, message: str) -> None:
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{stamp}] {message}")

    # --- data ingestion -----------------------------------------------------
    def load_data(self, csv_path: str) -> pd.DataFrame:
        df = load_comments_csv(csv_path)
        self._log(f"Loaded {len(df)} comments from {os.path.basename(csv_path)}")
        return df

    # --- per-comment processing --------------------------------------------
    def annotate(self, df: pd.DataFrame, text_column: str = "comment") -> pd.DataFrame:
        """Clean, NLP-preprocess and language-classify each comment."""
        raw_count = len(df)
        df = clean_dataframe(df, text_column=text_column)
        self._log(
            f"Cleaned & normalised text; {raw_count - len(df)} empty/duplicate "
            f"rows removed, {len(df)} remain"
        )

        df = self.language_detector.classify_dataframe(df, text_column=text_column)
        self._log("Detected and classified language for each comment")

        df["processed_text_no_stemming"] = df["cleaned_comment"].map(
            lambda t: self.preprocessor.preprocess_to_string(t, use_stemming=False)
        )
        df["processed_text_with_stemming"] = df["cleaned_comment"].map(
            lambda t: self.preprocessor.preprocess_to_string(t, use_stemming=True)
        )
        df["processed_text"] = (
            df["processed_text_with_stemming"]
            if self.use_stemming
            else df["processed_text_no_stemming"]
        )

        changed_rows = int(
            (df["processed_text_no_stemming"] != df["processed_text_with_stemming"]).sum()
        )
        self._log(
            "Applied NLP preprocessing (tokenisation, stop-word removal, "
            "lemmatisation and stemming comparison)"
        )
        self._log(
            f"Stemming changed {changed_rows} of {len(df)} comments "
            f"({(100.0 * changed_rows / len(df)) if len(df) else 0:.1f}%)"
        )
        return df

    # --- statistics ---------------------------------------------------------
    def language_statistics(self, df: pd.DataFrame) -> List[Dict[str, object]]:
        """Per-language counts and percentages, sorted by count."""
        counts = Counter(df["language"].astype(str))
        total = sum(counts.values()) or 1
        stats = [
            {
                "language": lang,
                "count": count,
                "percentage": round(100.0 * count / total, 2),
            }
            for lang, count in counts.most_common()
        ]
        return stats

    # --- full run -----------------------------------------------------------
    def run(
        self,
        csv_path: Optional[str] = None,
        dataframe: Optional[pd.DataFrame] = None,
        text_column: str = "comment",
    ) -> PipelineResult:
        """Run the pipeline from a CSV path or an in-memory DataFrame."""
        self.logs = []
        if dataframe is not None:
            df = dataframe.copy()
            self._log(f"Received {len(df)} comments from uploaded/collected data")
        elif csv_path is not None:
            df = self.load_data(csv_path)
        else:
            raise ValueError("Provide either csv_path or dataframe.")

        df = self.annotate(df, text_column=text_column)
        stats = self.language_statistics(df)
        distribution = {row["language"]: row["count"] for row in stats}
        self._log(
            "Computed language statistics: "
            + ", ".join(f"{r['language']}={r['count']}" for r in stats)
        )

        preprocessing_report = {
            "stemming_enabled": self.use_stemming,
            "changed_rows": int(
                (df["processed_text_no_stemming"] != df["processed_text_with_stemming"]).sum()
            ),
            "total_rows": int(len(df)),
        }

        return PipelineResult(
            dataframe=df,
            language_distribution=distribution,
            language_stats=stats,
            total_comments=int(len(df)),
            logs=list(self.logs),
            preprocessing_report=preprocessing_report,
        )

    # --- export -------------------------------------------------------------
    def export_cleaned(self, result: PipelineResult, path: str) -> str:
        """Export the cleaned, language-annotated dataset to CSV."""
        df = result.dataframe
        columns = [c for c in EXPORT_COLUMNS if c in df.columns]
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        df[columns].to_csv(path, index=False, encoding="utf-8")
        self._log(f"Exported cleaned dataset ({len(df)} rows) to {path}")
        return path

    # --- visualisation ------------------------------------------------------
    def generate_visualizations(
        self, result: PipelineResult, output_dir: str, theme: str = "light"
    ) -> Dict[str, str]:
        """Generate and save basic language charts; return name -> path map."""
        os.makedirs(output_dir, exist_ok=True)
        paths: Dict[str, str] = {}

        fig = viz.bar_chart(result.language_distribution, theme=theme)
        paths["language_bar"] = viz.save_figure(
            fig, os.path.join(output_dir, "language_bar.png")
        )

        fig = viz.pie_chart(result.language_distribution, theme=theme)
        paths["language_pie"] = viz.save_figure(
            fig, os.path.join(output_dir, "language_pie.png")
        )
        return paths
