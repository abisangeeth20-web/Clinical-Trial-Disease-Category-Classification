"""Medical text preprocessing and NLP operations.

Steps applied to every clinical-trial summary (training data AND app input, through the same code):
    1. normalise   - unicode -> ASCII, lowercase, strip URLs / e-mails, unify COVID-19 spellings
    2. clean       - remove punctuation, special characters and stand-alone numbers
    3. tokenize    - regex word tokenizer (keeps medical tokens such as "covid-19", "hba1c", "her2")
    4. stop words  - English stop words + generic clinical-trial boiler-plate words
    5. lemmatize   - NLTK WordNet lemmatizer (noun pass then verb pass)

`TextPreprocessor` wraps the steps as a scikit-learn transformer so it is saved inside the model
pipeline and the Streamlit app applies exactly the same processing to user text.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import config as C

# Generic words that appear in almost every trial summary regardless of disease. They carry no
# category signal, so they are treated as stop words (domain-specific stop-word list).
CLINICAL_STOPWORDS = {
    "study", "studies", "trial", "trials", "patient", "patients", "participant", "participants",
    "subject", "subjects", "aim", "aims", "purpose", "objective", "objectives", "investigate",
    "investigator", "investigators", "evaluate", "evaluation", "assess", "assessment", "determine",
    "compare", "compared", "comparison", "group", "groups", "randomized", "randomised", "random",
    "clinical", "research", "researcher", "researchers", "primary", "secondary", "outcome",
    "outcomes", "result", "results", "effect", "effects", "use", "used", "using", "include",
    "including", "included", "also", "may", "will", "would", "could", "can", "however", "well",
    "one", "two", "three", "first", "second", "year", "years", "month", "months", "week", "weeks",
    "day", "days", "time", "per", "within", "via", "versus", "vs", "et", "al", "eg", "ie", "etc",
    "propose", "proposed", "conduct", "conducted", "hypothesis", "hypothesize", "based", "new",
}

STOPWORDS = frozenset(set(ENGLISH_STOP_WORDS) | CLINICAL_STOPWORDS)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\S+@\S+")
_COVID_RE = re.compile(r"\b(?:covid[\s\-_]?19|covid|sars[\s\-]?cov[\s\-]?2|2019[\s\-]?ncov)\b")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\-\s]")
_TOKEN_RE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")


# ------------------------------------------------------------------ NLTK resources
def ensure_nltk() -> None:
    """Make sure the WordNet corpus is available (downloads once into ./nltk_data if missing)."""
    import nltk

    if str(C.NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(C.NLTK_DIR))
    for res, path in (("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")):
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.data.find(path + ".zip")
            except LookupError:
                C.NLTK_DIR.mkdir(parents=True, exist_ok=True)
                nltk.download(res, download_dir=str(C.NLTK_DIR), quiet=True)


_LEMMATIZER = None


def _lemmatizer():
    global _LEMMATIZER
    if _LEMMATIZER is None:
        ensure_nltk()
        from nltk.stem import WordNetLemmatizer

        _LEMMATIZER = WordNetLemmatizer()
    return _LEMMATIZER


@lru_cache(maxsize=200_000)
def lemmatize(token: str) -> str:
    if "-" in token or any(ch.isdigit() for ch in token):
        return token                      # keep codes/compounds such as covid-19, hba1c, sars-cov-2
    lem = _lemmatizer()
    return lem.lemmatize(lem.lemmatize(token, "n"), "v")


# ------------------------------------------------------------------ individual steps
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _COVID_RE.sub(" covid-19 ", text)
    return text


def remove_special_chars(text: str) -> str:
    text = _NON_ALNUM_RE.sub(" ", text)
    text = re.sub(r"\s-+|-+\s", " ", text)         # dashes that are not inside a word
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    out = [lemmatize(t) for t in tokens]
    return [t for t in out if t not in STOPWORDS and len(t) > 1]


def preprocess_steps(text: str) -> dict:
    """Return every intermediate step (used by the notebook and the app's 'NLP steps' view)."""
    norm = normalize(text)
    cleaned = remove_special_chars(norm)
    tokens = tokenize(cleaned)
    no_stop = remove_stopwords(tokens)
    lemmas = lemmatize_tokens(no_stop)
    return {"normalized": norm, "cleaned": cleaned, "tokens": tokens,
            "without_stopwords": no_stop, "lemmas": lemmas, "clean_text": " ".join(lemmas)}


def preprocess_text(text: str) -> str:
    return " ".join(lemmatize_tokens(remove_stopwords(tokenize(remove_special_chars(normalize(text))))))


def preprocess_series(texts: pd.Series) -> pd.Series:
    return texts.fillna("").astype(str).map(preprocess_text)


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """scikit-learn wrapper: raw summaries -> cleaned, lemmatized strings ready for TF-IDF."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        return np.array([preprocess_text(x) for x in pd.Series(X).fillna("").astype(str)], dtype=object)
