"""Feature engineering: NLP text column, text statistics, TF-IDF vectorizer and term analytics."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import chi2

from . import config as C
from .text_processing import preprocess_series

# Default TF-IDF settings (tuned values are written to models/model_metadata.json)
TFIDF_DEFAULTS = dict(ngram_range=(1, 2), min_df=3, max_df=0.9, sublinear_tf=True,
                      max_features=60_000, token_pattern=r"(?u)\b[\w\-]+\b", dtype=np.float32)


def build_tfidf(**overrides) -> TfidfVectorizer:
    params = {**TFIDF_DEFAULTS, **overrides}
    return TfidfVectorizer(**params)


def _keyword_regex(words):
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b")


KEYWORD_RE = {cat: _keyword_regex(words) for cat, words in C.DISEASE_KEYWORDS.items()}
ALL_KEYWORDS_RE = _keyword_regex({w for ws in C.DISEASE_KEYWORDS.values() for w in ws})


def mask_disease_keywords(texts: pd.Series) -> pd.Series:
    """Remove every disease-name keyword (robustness test: can the model classify from context alone?)."""
    return texts.str.replace(ALL_KEYWORDS_RE, " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()


def add_nlp_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the preprocessed text column and text-derived features."""
    out = df.copy()
    out[C.CLEAN_TEXT_COL] = preprocess_series(out[C.TEXT_COL])
    tokens = out[C.CLEAN_TEXT_COL].str.split()
    out["clean_token_count"] = tokens.str.len().astype(int)
    out["unique_token_count"] = tokens.map(lambda t: len(set(t))).astype(int)
    out["lexical_diversity"] = (out["unique_token_count"] / out["clean_token_count"].clip(lower=1)).round(3)
    out["avg_word_length"] = tokens.map(lambda t: np.mean([len(w) for w in t]) if t else 0).round(2)
    out["mentions_own_disease"] = [bool(KEYWORD_RE[c].search(t)) for c, t in zip(out[C.TARGET], out[C.CLEAN_TEXT_COL])]
    out["n_disease_categories_mentioned"] = out[C.CLEAN_TEXT_COL].map(
        lambda t: sum(bool(r.search(t)) for r in KEYWORD_RE.values())).astype(int)
    return out


def drop_empty_texts(df: pd.DataFrame, min_tokens: int = 3) -> tuple[pd.DataFrame, int]:
    """Summaries with fewer than `min_tokens` meaningful tokens after cleaning carry no signal."""
    keep = df["clean_token_count"] >= min_tokens
    return df[keep].reset_index(drop=True), int((~keep).sum())


# ------------------------------------------------------------------ term analytics (EDA + app)
def top_terms(texts: pd.Series, n: int = 20, ngram: int = 1) -> pd.DataFrame:
    if ngram == 1:
        cnt = Counter(w for t in texts for w in t.split())
    else:
        cv = CountVectorizer(ngram_range=(ngram, ngram), token_pattern=r"(?u)\b[\w\-]+\b", min_df=2)
        X = cv.fit_transform(texts)
        cnt = dict(zip(cv.get_feature_names_out(), np.asarray(X.sum(axis=0)).ravel()))
        cnt = Counter(cnt)
    rows = cnt.most_common(n)
    return pd.DataFrame(rows, columns=["term", "count"])


def doc_frequency(texts: pd.Series, n: int = 20) -> pd.DataFrame:
    """Share of summaries containing each term (document frequency)."""
    cnt = Counter(w for t in texts for w in set(t.split()))
    total = max(len(texts), 1)
    return pd.DataFrame([(w, c, round(c / total * 100, 2)) for w, c in cnt.most_common(n)],
                        columns=["term", "documents", "pct_documents"])


def distinctive_terms(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Most category-distinctive unigrams+bigrams by chi-square on TF-IDF (one-vs-rest)."""
    vec = build_tfidf(min_df=5, max_features=30_000)
    X = vec.fit_transform(df[C.CLEAN_TEXT_COL])
    names = vec.get_feature_names_out()
    rows = []
    for cat in C.CATEGORIES:
        y = (df[C.TARGET] == cat).astype(int).values
        scores, _ = chi2(X, y)
        mean_in = np.asarray(X[y == 1].mean(axis=0)).ravel()
        mean_out = np.asarray(X[y == 0].mean(axis=0)).ravel()
        scores = np.where(mean_in > mean_out, scores, 0)          # keep only terms over-represented in the class
        for rank, idx in enumerate(np.argsort(scores)[::-1][:n], 1):
            rows.append({C.TARGET: cat, "rank": rank, "term": names[idx], "chi2": round(float(scores[idx]), 1)})
    return pd.DataFrame(rows)
