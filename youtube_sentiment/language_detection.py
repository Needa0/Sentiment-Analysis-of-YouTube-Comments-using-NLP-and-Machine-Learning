"""Language detection and classification.

Implements the "Language Detection and Classification" phase of the Plan of Work:

    "Detect and classify comments into English, Hindi, Bengali, and code-mixed
     categories using the langdetect library."

Detection combines Unicode-script analysis (robust for short social-media text)
with the ``langdetect`` library. Comments that contain more than one script
(for example Latin + Devanagari) are classified as Code-Mixed, directly
supporting the objective of improving interpretation for code-mixed data.
"""

from __future__ import annotations

import re
from typing import Dict, Tuple

try:
    from langdetect import DetectorFactory, LangDetectException, detect
except Exception:  # pragma: no cover - optional dependency
    DetectorFactory = None

    class LangDetectException(Exception):
        pass

    detect = None

from .config import (
    BENGALI_RANGE,
    DEVANAGARI_RANGE,
    LANG_BENGALI,
    LANG_CODE_MIXED,
    LANG_ENGLISH,
    LANG_HINDI,
    LANG_NUMERIC,
    LANG_UNKNOWN,
    LANGDETECT_CODE_MAP,
    LATIN_RANGE,
)

# Make langdetect deterministic when the optional dependency is available.
if DetectorFactory is not None:
    DetectorFactory.seed = 0


def _in_range(ch: str, rng) -> bool:
    return rng[0] <= ord(ch) <= rng[1]


def script_counts(text: str) -> Dict[str, int]:
    """Count characters belonging to Latin, Devanagari and Bengali scripts."""
    counts = {"latin": 0, "devanagari": 0, "bengali": 0, "digits": 0}
    for ch in text:
        if _in_range(ch, LATIN_RANGE):
            counts["latin"] += 1
        elif _in_range(ch, DEVANAGARI_RANGE):
            counts["devanagari"] += 1
        elif _in_range(ch, BENGALI_RANGE):
            counts["bengali"] += 1
        elif ch.isdigit():
            counts["digits"] += 1
    return counts


_ROMANIZED_HINTS = {
    "hindi": {
        "ai", "mai", "main", "hum", "hu", "ho", "hai", "hain", "tha",
        "the", "thi", "nahi", "nahi", "bahut", "bohot", "mujhe", "mujhko",
        "tum", "aap", "kyun", "kyon", "kya", "kese", "kaise", "kyase",
        "bura", "shandar", "sundar", "kripya", "dekh", "badhia", "badhiya",
        "ye", "yeh", "wo", "woh", "bhi", "aur", "kar", "karo", "karna",
        "mera", "meri", "mere", "isko", "usko", "khud",
    },
    "bengali": {
        "ami", "amra", "tumi", "tui", "tomar", "tomra", "khub", "bhalo",
        "valo", "sundor", "sundar", "dada", "didi", "kotha", "kathha",
        "bolte", "chai", "onek", "gaan", "gaani", "besh", "shundor",
        "jottoshob", "jotoshob", "kemon", "kothay", "ki", "na", "eta",
        "ei", "era", "jonno", "khub bhalo",
    },
}

_WORD_RE = re.compile(r"[A-Za-z']+")


def _romanized_scores(text: str) -> Tuple[int, int]:
    """Score Latin-script text for Hindi/Bengali romanised hints."""
    tokens = [t.lower() for t in _WORD_RE.findall(text)]
    hindi = sum(1 for t in tokens if t in _ROMANIZED_HINTS["hindi"])
    bengali = sum(1 for t in tokens if t in _ROMANIZED_HINTS["bengali"])

    lowered = f" {' '.join(tokens)} "
    if " khub bhalo " in lowered:
        bengali += 2
    if " mai " in lowered or " mai khud " in lowered:
        hindi += 2
    return hindi, bengali


class LanguageDetector:
    """Classify comments into English, Hindi, Bengali, Code-Mixed or Numeric."""

    def detect_language(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return LANG_UNKNOWN

        counts = script_counts(text)
        scripts_present = [name for name in ("latin", "devanagari", "bengali") if counts[name] > 0]

        # Pure numeric comments are common in short YouTube datasets.
        if counts["digits"] > 0 and not scripts_present:
            return LANG_NUMERIC

        # More than one non-empty script present -> code-mixed.
        if len(scripts_present) > 1:
            return LANG_CODE_MIXED

        if scripts_present == ["devanagari"]:
            return LANG_HINDI
        if scripts_present == ["bengali"]:
            return LANG_BENGALI

        # Only Latin script (or no detectable script): use light heuristics
        # first, then fall back to langdetect.
        hindi_score, bengali_score = _romanized_scores(text)
        if hindi_score and bengali_score:
            return LANG_CODE_MIXED
        if bengali_score >= 2 and bengali_score >= hindi_score:
            return LANG_BENGALI
        if hindi_score >= 2 and hindi_score >= bengali_score:
            return LANG_HINDI
        if bengali_score == 1 and hindi_score == 0:
            return LANG_BENGALI
        if hindi_score == 1 and bengali_score == 0:
            return LANG_HINDI

        if detect is None:
            detected = LANG_ENGLISH
        else:
            try:
                code = detect(text)
            except LangDetectException:
                return LANG_ENGLISH if scripts_present == ["latin"] else LANG_UNKNOWN
            detected = LANGDETECT_CODE_MAP.get(code, LANG_ENGLISH)

        # Override English when transliterated Hindi/Bengali terms are present.
        if detected == LANG_ENGLISH:
            if bengali_score > hindi_score and bengali_score > 0:
                return LANG_BENGALI
            if hindi_score > bengali_score and hindi_score > 0:
                return LANG_HINDI
            if hindi_score or bengali_score:
                return LANG_CODE_MIXED if hindi_score and bengali_score else (LANG_HINDI if hindi_score else LANG_BENGALI)
        return detected

    def classify_dataframe(self, df, text_column: str = "comment", out_column: str = "language"):
        """Add a language column to a comments DataFrame."""
        df = df.copy()
        df[out_column] = df[text_column].astype(str).map(self.detect_language)
        return df
