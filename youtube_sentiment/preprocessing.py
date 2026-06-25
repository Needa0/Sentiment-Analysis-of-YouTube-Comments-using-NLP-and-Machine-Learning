"""NLP text preprocessing using NLTK.

Implements the "NLP Text Preprocessing" phase of the Plan of Work:

    "Apply tokenization, stop-word removal, stemming, and lemmatization using
     NLTK library to preprocess multilingual comments for analysis."
"""

from __future__ import annotations

from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

# Required NLTK data packages.
_REQUIRED_NLTK_DATA = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
]


def ensure_nltk_data() -> None:
    """Check for required NLTK corpora if they are already present."""
    for resource_path, _package in _REQUIRED_NLTK_DATA:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            # Keep working offline; the class falls back to local defaults.
            continue


class NLPPreprocessor:
    """Tokenise, remove stop-words, stem and lemmatise comments using NLTK."""

    def __init__(self, language: str = "english"):
        ensure_nltk_data()
        # A regex tokenizer keeps unicode word characters and avoids the
        # 'punkt' dependency, working reliably for multilingual scripts.
        self._tokenizer = RegexpTokenizer(r"\w+")
        self._stemmer = PorterStemmer()
        self._lemmatizer = WordNetLemmatizer()
        try:
            self._stopwords = set(stopwords.words(language))
        except (OSError, LookupError):
            self._stopwords = set(_FALLBACK_STOPWORDS)
        # Augment English stop-words with common Hindi/Bengali ones so
        # multilingual comments are handled (code-mixed data).
        self._stopwords |= _MULTILINGUAL_STOPWORDS

    def tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        return self._tokenizer.tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop-words from a token list."""
        return [t for t in tokens if t not in self._stopwords]

    def stem(self, tokens: List[str]) -> List[str]:
        """Apply Porter stemming to tokens."""
        return [self._stemmer.stem(t) for t in tokens]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Apply WordNet lemmatisation to tokens."""
        lemmas = []
        for token in tokens:
            try:
                lemmas.append(self._lemmatizer.lemmatize(token))
            except LookupError:
                lemmas.append(token)
        return lemmas

    def preprocess(self, text: str, use_stemming: bool = False) -> List[str]:
        """Run the full preprocessing chain and return processed tokens.

        When ``use_stemming`` is ``False``, the method lemmatises tokens.
        When ``True``, it applies stemming instead so both versions can be
        compared side by side in the dashboard.
        """
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_stemming:
            tokens = self.stem(tokens)
        else:
            tokens = self.lemmatize(tokens)
        return tokens

    def preprocess_to_string(self, text: str, use_stemming: bool = False) -> str:
        """Preprocess and join tokens back into a normalised string."""
        return " ".join(self.preprocess(text, use_stemming=use_stemming))


# Basic English fallback stop-words so the project remains usable offline.
_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "he", "her", "his", "i", "in", "is", "it", "its", "me", "my",
    "of", "on", "or", "our", "so", "that", "the", "their", "them", "they",
    "this", "to", "was", "we", "were", "with", "you", "your",
}

# A small multilingual stop-word set for Hindi (romanised + Devanagari) and
# Bengali, supporting the project's code-mixed data requirement.
_MULTILINGUAL_STOPWORDS = {
    # Hindi (romanised)
    "hai", "hai", "ka", "ki", "ke", "ko", "se", "me", "mein", "aur", "ye",
    "yeh", "wo", "woh", "hi", "to", "bhi", "kya", "nahi", "na",
    # Hindi (Devanagari)
    "है", "का", "की", "के", "को", "से", "में", "और", "यह", "वह", "भी",
    "क्या", "नहीं", "ना",
    # Bengali
    "এই", "এটা", "করে", "এবং", "আমি", "তুমি", "খুব", "না", "ও", "যে",
}
