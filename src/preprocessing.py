"""Data collection and cleaning: CSV -> pandas DataFrame -> cleaned, validated dataset."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

STR_COLS = ["source_condition_query", "nct_id", "title", "official_title", "brief_summary", "conditions",
            "interventions", "overall_status", "study_type", "phase", "sex", "minimum_age", "maximum_age",
            "eligibility_criteria", "clinicaltrials_url"]

PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1", "PHASE1": "Phase 1", "PHASE1 | PHASE2": "Phase 1/2",
    "PHASE2": "Phase 2", "PHASE2 | PHASE3": "Phase 2/3", "PHASE3": "Phase 3", "PHASE4": "Phase 4",
    "NOT_APPLICABLE": "Not Applicable",
}

_UNIT_TO_YEARS = {"year": 1.0, "month": 1 / 12, "week": 7 / 365.25, "day": 1 / 365.25,
                  "hour": 1 / 8766, "minute": 1 / 525960}


# ------------------------------------------------------------------ data collection
def load_raw() -> pd.DataFrame:
    """Load the raw CSV into a DataFrame.

    Uses the original 158 MB file when it is present; otherwise the three gzip parts in data/raw/
    (the GitHub copy, because GitHub web upload limits files to 25 MB). Both give identical rows.
    """
    if C.RAW_CSV.exists():
        return pd.read_csv(C.RAW_CSV, dtype={"healthy_volunteers": "string"})
    parts = [pd.read_csv(p, dtype={"healthy_volunteers": "string"}) for p in C.RAW_PARTS]
    return pd.concat(parts, ignore_index=True)


def write_parts(df: pd.DataFrame, paths) -> None:
    """Split a DataFrame into gzip CSV parts (< 25 MB each) for the GitHub repository."""
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, chunk in zip(paths, np.array_split(np.arange(len(df)), len(paths))):
        df.iloc[chunk].to_csv(path, index=False, compression={"method": "gzip", "compresslevel": 9})


def load_cleaned() -> pd.DataFrame:
    """Load the cleaned dataset (full CSV if present, otherwise the gzip parts)."""
    if C.CLEANED_CSV.exists():
        return pd.read_csv(C.CLEANED_CSV)
    return pd.concat([pd.read_csv(p) for p in C.CLEANED_PARTS], ignore_index=True)


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile: dtype, missing count/% and unique values."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(),
    })


# ------------------------------------------------------------------ helpers
def age_to_years(value) -> float:
    """'18 Years' -> 18.0, '6 Months' -> 0.5, '28 Days' -> 0.077; missing -> NaN."""
    if not isinstance(value, str):
        return np.nan
    m = re.match(r"\s*([\d.]+)\s*([A-Za-z]+)", value)
    if not m:
        return np.nan
    unit = m.group(2).lower().rstrip("s")
    return round(float(m.group(1)) * _UNIT_TO_YEARS.get(unit, np.nan), 3)


def _count_items(s: pd.Series) -> pd.Series:
    return s.fillna("").map(lambda v: 0 if v in ("", "Not specified") else len([x for x in v.split("|") if x.strip()]))


# ------------------------------------------------------------------ cleaning
def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Handle missing values, duplicates, types and label encoding. Returns (clean_df, cleaning_log)."""
    df = raw.copy()
    log = {"raw_rows": int(len(df)), "raw_columns": int(df.shape[1]),
           "missing_cells_raw": int(df.isna().sum().sum()),
           "missing_by_column_raw": {k: int(v) for k, v in df.isna().sum().items() if v}}

    # 1. text normalisation: strip whitespace, empty strings -> NaN
    for c in STR_COLS:
        df[c] = df[c].astype("string").str.strip().replace("", pd.NA)

    # 2. records without the NLP input or the label cannot be used
    before = len(df)
    df = df.dropna(subset=[C.TEXT_COL, C.LABEL_SOURCE_COL])
    log["dropped_missing_summary_or_label"] = before - len(df)

    # 3. duplicates: exact rows, repeated trial IDs, repeated summaries
    before = len(df)
    df = df.drop_duplicates()
    log["dropped_exact_duplicate_rows"] = before - len(df)
    before = len(df)
    df = df.drop_duplicates(subset="nct_id")
    log["dropped_duplicate_nct_id"] = before - len(df)

    key = df[C.TEXT_COL].str.lower().str.replace(r"\s+", " ", regex=True)
    n_labels = df.groupby(key)[C.LABEL_SOURCE_COL].transform("nunique")
    conflicting = n_labels > 1
    log["dropped_summaries_with_conflicting_labels"] = int(conflicting.sum())
    df, key = df[~conflicting], key[~conflicting]
    dup_summary = key.duplicated()
    log["dropped_duplicate_summaries_same_label"] = int(dup_summary.sum())
    df = df[~dup_summary]

    # 4. target label: readable disease category + integer encoding
    df[C.TARGET] = df[C.LABEL_SOURCE_COL].str.lower().map(C.LABEL_MAP)
    unknown = df[C.TARGET].isna()
    log["dropped_unknown_label"] = int(unknown.sum())
    df = df[~unknown]
    df["label_id"] = df[C.TARGET].map({c: i for i, c in enumerate(C.CATEGORIES)}).astype(int)

    # 5. missing values in descriptive columns
    df["official_title"] = df["official_title"].fillna(df["title"])
    df["conditions"] = df["conditions"].fillna(df[C.TARGET])          # the 1 missing condition -> its category
    df["interventions"] = df["interventions"].fillna("Not specified")
    df["eligibility_criteria"] = df["eligibility_criteria"].fillna("Not specified")
    df["phase"] = df["phase"].fillna("NOT_APPLICABLE")               # observational / device / behavioural
    df["phase_label"] = df["phase"].map(PHASE_LABELS).fillna(df["phase"])
    df["sex"] = df["sex"].fillna("ALL")                               # mode (78.6%)
    hv = df["healthy_volunteers"].astype("string").str.lower()
    df["healthy_volunteers"] = hv.map({"true": "Yes", "false": "No"}).fillna("Unknown")
    df["overall_status"] = df["overall_status"].str.upper()
    df["study_type"] = df["study_type"].str.upper()

    # 6. ages -> numeric years. Missing minimum = no lower limit (0); missing maximum = no upper limit (kept NaN
    #    on purpose and flagged, because imputing a number would invent an age limit).
    df["min_age_years"] = df["minimum_age"].map(age_to_years).fillna(0.0)
    df["max_age_years"] = df["maximum_age"].map(age_to_years)
    df["no_upper_age_limit"] = df["max_age_years"].isna()
    df["includes_children"] = df["min_age_years"] < 18
    df["includes_older_adults"] = df["max_age_years"].isna() | (df["max_age_years"] >= 65)
    df["minimum_age"] = df["minimum_age"].fillna("No limit")
    df["maximum_age"] = df["maximum_age"].fillna("No limit")

    # 7. simple structural features
    df["summary_char_len"] = df[C.TEXT_COL].str.len().astype(int)
    df["summary_word_count"] = df[C.TEXT_COL].str.split().str.len().astype(int)
    df["n_conditions"] = _count_items(df["conditions"]).astype(int)
    df["n_interventions"] = _count_items(df["interventions"]).astype(int)

    df = df.reset_index(drop=True)
    log["rows_after_cleaning"] = int(len(df))
    log["missing_cells_after_cleaning_excl_max_age"] = int(df.drop(columns=["max_age_years"]).isna().sum().sum())
    log["class_counts"] = df[C.TARGET].value_counts().to_dict()
    return df, log
