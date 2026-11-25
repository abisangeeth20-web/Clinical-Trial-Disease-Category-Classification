"""End-to-end NLP pipeline:
collect CSV -> clean -> text preprocessing (tokenize, stop words, lemmatize) -> features -> EDA
-> TF-IDF + model comparison/tuning -> evaluation -> saved model + reports.

Usage:  python run_pipeline.py              (full run, ~15 min)
        python run_pipeline.py --skip-train (data + EDA only)
"""
import json
import sys
import time

from src import config as C
from src import eda, features, preprocessing, train


def log(msg):
    print(msg, flush=True)


def main(skip_train=False):
    t0 = time.time()
    for d in (C.PROCESSED_DIR, C.REPORTS_DIR, C.MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    log("1. Data collection (CSV -> pandas DataFrame)")
    raw = preprocessing.load_raw()
    log(f"   loaded {raw.shape[0]:,} rows x {raw.shape[1]} columns")
    if C.RAW_CSV.exists() and not all(p.exists() for p in C.RAW_PARTS):
        preprocessing.write_parts(raw, C.RAW_PARTS)
        log("   wrote gzip raw-data parts to data/raw/ (GitHub copy)")
    preprocessing.profile(raw).to_csv(C.REPORTS_DIR / "raw_data_profile.csv")

    log("2. Cleaning (missing values, duplicates, types, label encoding)")
    clean, clog = preprocessing.clean(raw)
    for k, v in clog.items():
        if k not in ("missing_by_column_raw", "class_counts"):
            log(f"   {k}: {v}")

    log("3. NLP preprocessing (normalise, clean, tokenize, stop words, lemmatize) + features")
    feat = features.add_nlp_features(clean)
    feat, n_empty = features.drop_empty_texts(feat)
    clog["dropped_summaries_under_3_tokens_after_cleaning"] = n_empty
    clog["final_rows"] = int(len(feat))
    clog["class_counts"] = feat[C.TARGET].value_counts().to_dict()
    log(f"   dropped {n_empty} summaries with < 3 meaningful tokens -> {len(feat):,} rows")

    clean_out = feat.drop(columns=[c for c in feat.columns if c in (
        C.CLEAN_TEXT_COL, "clean_token_count", "unique_token_count", "lexical_diversity", "avg_word_length",
        "mentions_own_disease", "n_disease_categories_mentioned")])
    clean_out.to_csv(C.CLEANED_CSV, index=False)
    preprocessing.write_parts(clean_out, C.CLEANED_PARTS)
    nlp_cols = ["nct_id", C.TARGET, "label_id", C.TEXT_COL, C.CLEAN_TEXT_COL, "summary_word_count",
                "clean_token_count", "unique_token_count", "lexical_diversity", "avg_word_length",
                "mentions_own_disease", "n_disease_categories_mentioned"]
    feat[nlp_cols].to_csv(C.PREPROCESSED_GZ, index=False, compression={"method": "gzip", "compresslevel": 9})
    feat[C.DASHBOARD_COLS].to_csv(C.DASHBOARD_GZ, index=False, compression={"method": "gzip", "compresslevel": 9})

    log("4. Exploratory data analysis")
    res_eda = eda.run_eda(feat)
    log(f"   figures -> {C.FIGURES_DIR}, tables -> {C.TABLES_DIR}")

    summary = {"cleaning_log": clog, "imbalance_ratio": res_eda["imbalance_ratio"]}
    if not skip_train:
        log("5-6. TF-IDF + model training, tuning & evaluation")
        res = train.train_all(feat, log)
        summary["best_model"] = res["best"]
        summary["test_metrics"] = res["metrics"]
        summary["robustness_keyword_masking"] = res["robustness"]
    with open(C.REPORTS_DIR / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(skip_train="--skip-train" in sys.argv)
