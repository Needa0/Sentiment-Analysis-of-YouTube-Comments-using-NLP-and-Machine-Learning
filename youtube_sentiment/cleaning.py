"""Data cleaning and text normalization using Pandas and Regular Expressions.

Cleaning steps:

- remove HTML markup and URLs,
- remove user mentions,
- remove punctuation, special characters, symbols and emojis (keeping letters
  from all supported scripts and digits),
- convert to lowercase,
- collapse whitespace,
- drop duplicate and empty records.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Regular expressions for cleaning (re library).
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")


def remove_urls(text: str) -> str:
    """Remove URLs from text using a regular expression."""
    return URL_RE.sub(" ", text)


def remove_html(text: str) -> str:
    """Remove HTML tags that may appear in YouTube comment markup."""
    return HTML_TAG_RE.sub(" ", text)


def remove_special_characters(text: str) -> str:
    """Remove mentions, punctuation, special characters, symbols and emojis.

    Keeps letters (any script, including combining marks/matras) and digits so
    multilingual words such as Devanagari/Bengali text are preserved intact.
    """
    text = MENTION_RE.sub(" ", text)
    kept = []
    for ch in text:
        if ch.isspace():
            kept.append(ch)
            continue
        # Unicode categories: L* letters, M* marks (matras), N* numbers.
        if unicodedata.category(ch)[0] in ("L", "M", "N"):
            kept.append(ch)
        else:
            kept.append(" ")
    return "".join(kept)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase format."""
    return text.lower()


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace into single spaces and strip ends."""
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_text(text: str) -> str:
    """Apply the full cleaning / normalisation chain to a single comment."""
    if not isinstance(text, str):
        return ""
    text = remove_html(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = normalize_whitespace(text)
    return text


def clean_dataframe(
    df: pd.DataFrame,
    text_column: str = "comment",
    cleaned_column: str = "cleaned_comment",
) -> pd.DataFrame:
    """Clean a comments DataFrame: drop duplicates/empties and normalise text.

    Returns a new DataFrame with an added ``cleaned_column``.
    """
    df = df.copy()
    df = df.dropna(subset=[text_column])
    # Remove duplicate records based on the raw comment text.
    df = df.drop_duplicates(subset=[text_column]).reset_index(drop=True)
    df[cleaned_column] = df[text_column].astype(str).map(clean_text)
    # Drop rows that became empty after cleaning.
    df = df[df[cleaned_column].str.strip().astype(bool)].reset_index(drop=True)
    return df
