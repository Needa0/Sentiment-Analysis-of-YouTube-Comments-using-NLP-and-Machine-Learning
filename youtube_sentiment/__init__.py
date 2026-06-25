"""Multilingual YouTube comment collection, preprocessing and language classification.

This package provides the first phases of the workflow:

- Collect multilingual YouTube comments via the YouTube Data API v3 (or load an
  uploaded CSV) and store them in structured CSV format.
- Clean and normalise text with Pandas and regular expressions.
- Preprocess text using NLTK (tokenisation, stop-word removal, stemming,
  lemmatisation).
- Detect and classify language (English, Hindi, Bengali, Code-Mixed) and
  compute language statistics.
"""

from .config import SUPPORTED_LANGUAGES

__all__ = ["SUPPORTED_LANGUAGES"]

__version__ = "2.0.0"
