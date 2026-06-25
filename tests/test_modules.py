"""Unit tests for the in-scope modules.

Covers data collection, cleaning/normalisation, NLP preprocessing and language
detection/classification.
"""

from __future__ import annotations

import pandas as pd
import pytest

from youtube_sentiment.cleaning import clean_dataframe, clean_text
from youtube_sentiment.config import SUPPORTED_LANGUAGES
from youtube_sentiment.data_collection import (
    YouTubeCommentCollector,
    load_comments_csv,
    save_comments_csv,
)
from youtube_sentiment.language_detection import LanguageDetector
from youtube_sentiment.preprocessing import NLPPreprocessor


# --- Dataset Collection (YouTube Data API v3 + CSV via Pandas) --------------
def test_collector_requires_api_key():
    collector = YouTubeCommentCollector(api_key=None)
    with pytest.raises(ValueError):
        collector.fetch_video_comments("abc")


def test_csv_round_trip(tmp_path):
    df = pd.DataFrame(
        {
            "comment_id": ["1"],
            "video_id": ["v"],
            "author": ["a"],
            "comment": ["hello"],
            "like_count": [3],
        }
    )
    path = tmp_path / "c.csv"
    save_comments_csv(df, str(path))
    loaded = load_comments_csv(str(path))
    assert loaded.iloc[0]["comment"] == "hello"


# --- Data Cleaning and Normalization (Pandas + regex) -----------------------
def test_clean_text_removes_url_punct_emoji_and_lowercases():
    raw = "Check THIS!! https://example.com Amazing,, video 😍 #1 :)"
    cleaned = clean_text(raw)
    assert "http" not in cleaned
    assert "!" not in cleaned and "," not in cleaned and "#" not in cleaned
    assert "😍" not in cleaned  # emojis stripped during normalisation
    assert cleaned == cleaned.lower()


def test_clean_text_keeps_multilingual_letters():
    assert "वीडियो" in clean_text("यह वीडियो 😀!!")
    assert "ভিডিও" in clean_text("এই ভিডিও 🔥")


def test_clean_dataframe_drops_duplicates_and_empties():
    df = pd.DataFrame({"comment": ["Great video!", "Great video!", "😀😀", "Bad one..."]})
    out = clean_dataframe(df)
    # duplicate removed; emoji-only row becomes empty after cleaning and is dropped
    assert len(out) == 2
    assert "cleaned_comment" in out.columns


# --- NLP Text Preprocessing (NLTK) ------------------------------------------
def test_preprocessing_tokenizes_and_removes_stopwords():
    pre = NLPPreprocessor()
    tokens = pre.preprocess("this is an amazing and wonderful video")
    assert "is" not in tokens and "an" not in tokens
    assert "video" in tokens


def test_stemming_available():
    pre = NLPPreprocessor()
    assert pre.stem(["running", "loved"]) == ["run", "love"]


def test_preprocess_to_string():
    pre = NLPPreprocessor()
    out = pre.preprocess_to_string("the videos are amazing")
    assert isinstance(out, str) and "amazing" in out


# --- Language Detection and Classification ----------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("this is a great video", "English"),
        ("यह वीडियो बहुत अच्छा है", "Hindi"),
        ("এই ভিডিও খুব ভালো", "Bengali"),
        ("this video बहुत अच्छा है", "Code-Mixed"),
        ("5 49", "Numeric"),
        ("khub sundor dada", "Bengali"),
        ("mai khud bengali hu", "Hindi"),
        ("we bengalis don't really say g honta we use jottoshob", "Bengali"),
    ],
)
def test_language_detection(text, expected):
    assert LanguageDetector().detect_language(text) == expected


def test_supported_languages_config():
    assert set(SUPPORTED_LANGUAGES) == {"English", "Hindi", "Bengali", "Code-Mixed", "Numeric"}


def test_classify_dataframe_adds_language_column():
    df = pd.DataFrame({"comment": ["great video", "यह अच्छा है"]})
    out = LanguageDetector().classify_dataframe(df)
    assert "language" in out.columns
    assert out.iloc[0]["language"] == "English"
