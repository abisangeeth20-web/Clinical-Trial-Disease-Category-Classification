"""Model building, tuning, evaluation and persistence for disease-category classification."""
from __future__ import annotations

import json
import platform
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

from . import config as C
from . import plotting as P
from .features import build_tfidf, mask_disease_keywords
from .text_processing import TextPreprocessor

NGRAMS = [(1, 1), (1, 2)]


def candidates():
    """name -> (pipeline, parameter distributions, n_iter)."""
    def pipe(clf, **tfidf):
        return Pipeline([("tfidf", build_tfidf(**tfidf)), ("clf", clf)])

    common = {"tfidf__ngram_range": NGRAMS, "tfidf__min_df": [2, 3, 5]}
    return {
        "Multinomial Naive Bayes": (pipe(MultinomialNB()),
                                    {**common, "clf__alpha": [0.01, 0.03, 0.1, 0.3]}, 6),
        "Complement Naive Bayes": (pipe(ComplementNB()),
                                   {**common, "clf__alpha": [0.03, 0.1, 0.3, 1.0]}, 6),
        "Logistic Regression": (pipe(LogisticRegression(max_iter=3000)),
                                {**common, "clf__C": [2, 5, 10, 20], "clf__class_weight": [None, "balanced"]}, 6),
        "Linear SVM": (pipe(LinearSVC()),
                       {**common, "clf__C": [0.1, 0.3, 0.6, 1.0], "clf__class_weight": [None, "balanced"]}, 6),
        "SGD (modified Huber)": (pipe(SGDClassifier(loss="modified_huber", max_iter=50, random_state=C.RANDOM_STATE)),
                                 {**common, "clf__alpha": [1e-5, 3e-5, 1e-4], "clf__class_weight": [None, "balanced"]}, 6),
        "Random Forest": (pipe(RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=C.RANDOM_STATE),
                               max_features=15_000),
                          {"tfidf__ngram_range": [(1, 1)], "tfidf__min_df": [3, 5],
                           "clf__min_samples_leaf": [1, 2], "clf__class_weight": [None, "balanced_subsample"]}, 3),
    }


def _scores(model, X, y, classes):
    pred = model.predict(X)
    res = {
        "accuracy": accuracy_score(y, pred),
        "precision_macro": precision_score(y, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, pred, average="macro"),
        "precision_weighted": precision_score(y, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y, pred, average="weighted"),
    }
    try:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X)
        else:
            score = model.decision_function(X)
        # score columns follow model.classes_ (alphabetical), not the display order in `classes`
        res["roc_auc_ovr_macro"] = roc_auc_score(label_binarize(y, classes=list(model.classes_)), score,
                                                 average="macro")
    except Exception:  # noqa: BLE001
        res["roc_auc_ovr_macro"] = np.nan
    return {k: round(float(v), 4) for k, v in res.items()}, pred


def _class_coefs(clf):
    """Per-class term weights for linear / NB models (used for explanations)."""
    if isinstance(clf, CalibratedClassifierCV):
        return np.mean([cc.estimator.coef_ for cc in clf.calibrated_classifiers_], axis=0)
    if hasattr(clf, "coef_"):
        return clf.coef_
    if hasattr(clf, "feature_log_prob_"):
        flp = clf.feature_log_prob_
        return flp - flp.mean(axis=0, keepdims=True)
    return None


def train_all(df: pd.DataFrame, log=print) -> dict:
    P.setup()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = df[C.CLEAN_TEXT_COL], df[C.TARGET]
    idx_tr, idx_te = train_test_split(df.index, test_size=C.TEST_SIZE, stratify=y, random_state=C.RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = X.loc[idx_tr], X.loc[idx_te], y.loc[idx_tr], y.loc[idx_te]
    classes = C.CATEGORIES
    log(f"   train {len(X_tr):,} / test {len(X_te):,} summaries, {len(classes)} classes")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=C.RANDOM_STATE)
    rows, fitted = [], {}
    for name, (pipe, params, n_iter) in candidates().items():
        t0 = time.time()
        search = RandomizedSearchCV(pipe, params, n_iter=n_iter, scoring="f1_macro", cv=cv,
                                    random_state=C.RANDOM_STATE, n_jobs=-1 if name != "Random Forest" else 1,
                                    refit=True)
        search.fit(X_tr, y_tr)
        secs = time.time() - t0
        test, _ = _scores(search.best_estimator_, X_te, y_te, classes)
        best_i = search.best_index_
        row = {"model": name, "cv_f1_macro": round(search.best_score_, 4),
               "cv_f1_std": round(float(search.cv_results_["std_test_score"][best_i]), 4),
               **{f"test_{k}": v for k, v in test.items()},
               "tuning_seconds": round(secs, 1),
               "best_params": json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                          for k, v in search.best_params_.items()})}
        rows.append(row); fitted[name] = search.best_estimator_
        log(f"   {name:<26} CV F1 {row['cv_f1_macro']:.4f} | test acc {test['accuracy']:.4f} "
            f"F1-macro {test['f1_macro']:.4f} ({secs:.0f}s)")

    comp = pd.DataFrame(rows).sort_values("cv_f1_macro", ascending=False).reset_index(drop=True)
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comp.to_csv(C.REPORTS_DIR / "model_comparison.csv", index=False)
    best_name = comp.loc[0, "model"]
    best = fitted[best_name]
    log(f"   best model by CV macro-F1: {best_name}")

    # probability-capable final classifier (Linear SVM has no predict_proba -> sigmoid calibration)
    tfidf, clf = best.named_steps["tfidf"], best.named_steps["clf"]
    if not hasattr(clf, "predict_proba"):
        Xtr_vec = tfidf.transform(X_tr)
        clf = CalibratedClassifierCV(estimator=clf, cv=3, method="sigmoid").fit(Xtr_vec, y_tr)
        log("   wrapped in CalibratedClassifierCV to provide class probabilities")
    final = Pipeline([("preprocess", TextPreprocessor()), ("tfidf", tfidf), ("clf", clf)])
    text_model = Pipeline([("tfidf", tfidf), ("clf", clf)])            # same model, on pre-cleaned text

    final_metrics, pred = _scores(text_model, X_te, y_te, classes)
    train_metrics, _ = _scores(text_model, X_tr, y_tr, classes)
    log(f"   final test: acc {final_metrics['accuracy']:.4f} | P {final_metrics['precision_macro']:.4f} | "
        f"R {final_metrics['recall_macro']:.4f} | F1 {final_metrics['f1_macro']:.4f}")

    # end-to-end check: raw summary -> full pipeline gives the same predictions
    sample = df.loc[idx_te[:300]]
    agree = float((final.predict(sample[C.TEXT_COL]) == text_model.predict(sample[C.CLEAN_TEXT_COL])).mean())
    log(f"   raw-text pipeline agreement on 300 test summaries: {agree:.3f}")

    report = classification_report(y_te, pred, labels=classes, output_dict=True, zero_division=0)
    rep_df = pd.DataFrame(report).T.round(4)
    rep_df.to_csv(C.REPORTS_DIR / "classification_report.csv")
    cm = confusion_matrix(y_te, pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(C.REPORTS_DIR / "confusion_matrix.csv")

    # error analysis
    proba = text_model.predict_proba(X_te)
    conf = proba.max(axis=1)
    err = pd.DataFrame({"nct_id": df.loc[idx_te, "nct_id"].values, "true": y_te.values, "predicted": pred,
                        "confidence": conf.round(3), "brief_summary": df.loc[idx_te, C.TEXT_COL].str[:400].values})
    mis = err[err.true != err.predicted].sort_values("confidence", ascending=False)
    mis.head(200).to_csv(C.REPORTS_DIR / "misclassified_examples.csv", index=False)
    pairs = (mis.groupby(["true", "predicted"]).size().sort_values(ascending=False).head(10)
             .reset_index(name="count"))
    pairs.to_csv(C.REPORTS_DIR / "top_confusions.csv", index=False)
    conf_bins = pd.cut(conf, [0, .5, .7, .9, 1.0001], labels=["<0.5", "0.5-0.7", "0.7-0.9", ">=0.9"])
    calib = (pd.DataFrame({"bin": conf_bins, "correct": pred == y_te.values})
             .groupby("bin", observed=False).agg(n=("correct", "size"), accuracy=("correct", "mean")).round(4))
    calib.to_csv(C.REPORTS_DIR / "confidence_vs_accuracy.csv")

    # top terms per class
    coefs = _class_coefs(clf)
    names = tfidf.get_feature_names_out()
    top_terms = {}
    if coefs is not None:
        for i, cat in enumerate(clf.classes_):
            top_terms[cat] = [names[j] for j in np.argsort(coefs[i])[::-1][:15]]

    # robustness: remove disease-name keywords from train AND test, retrain the same configuration
    log("   keyword-masking robustness test ...")
    masked_tr, masked_te = mask_disease_keywords(X_tr), mask_disease_keywords(X_te)
    masked_model = clone(best).fit(masked_tr, y_tr)
    masked_metrics, _ = _scores(masked_model, masked_te, y_te, classes)
    orig_on_masked, _ = _scores(best, masked_te, y_te, classes)
    robustness = {"original_test": {k: final_metrics[k] for k in ("accuracy", "f1_macro")},
                  "original_model_on_masked_test": {k: orig_on_masked[k] for k in ("accuracy", "f1_macro")},
                  "retrained_on_masked_text": {k: masked_metrics[k] for k in ("accuracy", "f1_macro")}}
    log(f"   masked: original model {orig_on_masked['accuracy']:.4f}, retrained {masked_metrics['accuracy']:.4f}")

    _figures(comp, cm, classes, rep_df, top_terms, robustness, calib)

    # example summaries for the app (correctly classified, readable length)
    ex = df.loc[idx_te].assign(pred=pred, conf=conf)
    ex = ex[(ex.pred == ex[C.TARGET]) & ex.summary_word_count.between(35, 110)]
    examples = {cat: ex[ex[C.TARGET] == cat].sort_values("conf", ascending=False)[C.TEXT_COL].head(3).tolist()
                for cat in classes}

    joblib.dump(final, C.MODEL_PATH, compress=("xz", 3))
    best_params = json.loads(comp.loc[0, "best_params"])
    meta = {
        "best_model": best_name,
        "final_estimator": type(clf).__name__,
        "input": "brief_summary (raw text) -> TextPreprocessor -> TF-IDF -> classifier",
        "classes": list(clf.classes_),
        "best_params": best_params,
        "tfidf_vocabulary_size": int(len(names)),
        "train_size": int(len(X_tr)), "test_size": int(len(X_te)),
        "test_metrics": final_metrics, "train_metrics": train_metrics,
        "classification_report": {k: v for k, v in report.items()},
        "confusion_matrix": cm.tolist(),
        "model_comparison": comp.to_dict(orient="records"),
        "top_terms_per_class": top_terms,
        "top_confusions": pairs.to_dict(orient="records"),
        "confidence_vs_accuracy": calib.reset_index().astype({"bin": str}).to_dict(orient="records"),
        "robustness_keyword_masking": robustness,
        "raw_pipeline_agreement": agree,
        "examples": examples,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    with open(C.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    return {"best": best_name, "metrics": final_metrics, "comparison": comp, "robustness": robustness}


def _figures(comp, cm, classes, rep_df, top_terms, robustness, calib):
    # model comparison
    m = comp.set_index("model")[["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]]
    m.columns = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    m.plot(kind="bar", ax=ax, width=0.8, colormap="tab10")
    ax.set_ylim(max(0, m.values.min() - 0.08), 1.0); ax.set_ylabel("score"); ax.set_xlabel("")
    ax.set_title("Model comparison on the held-out test set"); ax.legend(loc="lower right", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    P.save(fig, "model_01_comparison")

    # confusion matrix (counts + row-normalised)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), gridspec_kw={"wspace": 0.08})
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[0], cbar=False)
    axes[0].set_title("Confusion matrix (counts)")
    cmn = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=False, ax=axes[1], cbar=False)
    axes[1].set_title("Confusion matrix (recall-normalised)")
    for ax in axes:
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=35)
    P.save(fig, "model_02_confusion_matrix")

    # per-class metrics
    pc = rep_df.loc[classes, ["precision", "recall", "f1-score"]]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    pc.plot(kind="bar", ax=ax, width=0.8, color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylim(max(0, pc.values.min() - 0.1), 1.0); ax.set_title("Per-class precision / recall / F1 (test set)")
    plt.xticks(rotation=20, ha="right"); ax.set_xlabel("")
    P.save(fig, "model_03_per_class_metrics")

    # top terms per class
    if top_terms:
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        for ax, cat in zip(axes.ravel(), classes):
            terms = top_terms.get(cat, [])[:10][::-1]
            ax.barh(terms, range(1, len(terms) + 1), color=C.PALETTE[cat])
            ax.set_title(cat, fontsize=11); ax.set_xticks([])
        fig.suptitle("Strongest TF-IDF terms the model uses for each category (top 10, by weight rank)",
                     fontweight="bold")
        fig.tight_layout()
        P.save(fig, "model_04_top_terms_per_class")

    # robustness
    r = pd.DataFrame(robustness).T
    r.index = ["Original text", "Original model,\nkeywords removed", "Retrained,\nkeywords removed"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    r.plot(kind="bar", ax=ax, width=0.7, color=["#4C78A8", "#54A24B"])
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_title("Robustness: accuracy / macro-F1 without disease-name keywords")
    plt.xticks(rotation=0)
    P.save(fig, "model_05_keyword_masking")

    # confidence vs accuracy
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(calib.index.astype(str), calib["accuracy"], color="#4C78A8")
    for i, (a, n) in enumerate(zip(calib["accuracy"], calib["n"])):
        ax.text(i, a + 0.01, f"{a:.2f}\n(n={n})", ha="center", fontsize=8)
    ax.set_ylim(0, 1.15); ax.set_xlabel("predicted-class probability"); ax.set_ylabel("accuracy")
    ax.set_title("Prediction confidence vs actual accuracy (test set)")
    P.save(fig, "model_06_confidence_accuracy")
