"""Central configuration for languages and detection.

The application scope covers multilingual YouTube comment collection,
preprocessing and language classification. Comments are classified into
English, Hindi, Bengali and Code-Mixed categories.
"""

from __future__ import annotations

# --- Supported languages ----------------------------------------------------
LANG_ENGLISH = "English"
LANG_HINDI = "Hindi"
LANG_BENGALI = "Bengali"
LANG_CODE_MIXED = "Code-Mixed"
LANG_NUMERIC = "Numeric"
LANG_UNKNOWN = "Unknown"

SUPPORTED_LANGUAGES = [
    LANG_ENGLISH,
    LANG_HINDI,
    LANG_BENGALI,
    LANG_CODE_MIXED,
    LANG_NUMERIC,
]

# langdetect ISO codes -> human readable language names used by this project.
LANGDETECT_CODE_MAP = {
    "en": LANG_ENGLISH,
    "hi": LANG_HINDI,
    "bn": LANG_BENGALI,
}

# Unicode script ranges used to support multilingual / code-mixed detection.
DEVANAGARI_RANGE = (0x0900, 0x097F)  # Hindi
BENGALI_RANGE = (0x0980, 0x09FF)  # Bengali
LATIN_RANGE = (0x0041, 0x024F)  # English / romanised text

# Colours used for language categories in charts (consistent across the UI).
LANGUAGE_COLORS = {
    LANG_ENGLISH: "#4C72B0",
    LANG_HINDI: "#DD8452",
    LANG_BENGALI: "#55A868",
    LANG_CODE_MIXED: "#C44E52",
    LANG_NUMERIC: "#8172B2",
    LANG_UNKNOWN: "#8C8C8C",
}
