"""Exploratory data analysis: figures (reports/figures) and summary tables (reports/eda_tables)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C
from . import plotting as P
from .features import distinctive_terms, doc_frequency, top_terms

T = C.TARGET


def _order(df):
    return df[T].value_counts().index.tolist()


def _pct_table(df, col):
    tab = pd.crosstab(df[T], df[col], normalize="index").mul(100).round(1)
    return tab.loc[[c for c in _order(df) if c in tab.index]]


def save_table(df, name, index=True):
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES_DIR / f"{name}.csv", index=index)


def run_eda(df: pd.DataFrame) -> dict:
    P.setup()
    order = _order(df)
    out = {}

    # 01 category distribution -------------------------------------------------------------
    counts = df[T].value_counts()
    dist = pd.DataFrame({"trials": counts, "pct": (counts / len(df) * 100).round(2)})
    save_table(dist, "category_distribution")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(counts.index[::-1], counts.values[::-1], color=P.cat_colors(counts.index[::-1]))
    for i, (v, p) in enumerate(zip(counts.values[::-1], dist.pct.values[::-1])):
        ax.text(v, i, f" {v:,} ({p}%)", va="center", fontsize=9)
    ax.set_title("Disease category distribution (number of clinical trials)")
    ax.set_xlabel("Trials"); ax.set_xlim(0, counts.max() * 1.22)
    P.save(fig, "eda_01_category_distribution")
    out["imbalance_ratio"] = round(counts.max() / counts.min(), 1)

    # 02 summary length -------------------------------------------------------------------
    length = df.groupby(T)[["summary_word_count", "clean_token_count", "lexical_diversity"]].agg(["mean", "median"]).round(2)
    length.columns = ["_".join(c) for c in length.columns]
    save_table(length.loc[order], "summary_length_by_category")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df["summary_word_count"].clip(upper=600), bins=60, ax=axes[0], color="#4C78A8")
    axes[0].set_title("Brief-summary length (words, clipped at 600)")
    axes[0].axvline(df["summary_word_count"].median(), color="k", ls="--", lw=1)
    sns.boxplot(data=df, y=T, x="summary_word_count", order=order, ax=axes[1], palette=C.PALETTE,
                hue=T, legend=False, showfliers=False)
    axes[1].set_title("Summary length by category (outliers hidden)"); axes[1].set_ylabel("")
    P.save(fig, "eda_02_summary_length")

    # 03 / 04 most common terms -------------------------------------------------------------
    uni = top_terms(df[C.CLEAN_TEXT_COL], 25)
    bi = top_terms(df[C.CLEAN_TEXT_COL], 25, ngram=2)
    docf = doc_frequency(df[C.CLEAN_TEXT_COL], 25)
    save_table(uni, "top_unigrams", index=False); save_table(bi, "top_bigrams", index=False)
    save_table(docf, "top_terms_document_frequency", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    sns.barplot(data=uni, y="term", x="count", ax=axes[0], color="#4C78A8")
    axes[0].set_title("Top 25 medical terms (after cleaning & lemmatization)")
    sns.barplot(data=bi, y="term", x="count", ax=axes[1], color="#F58518")
    axes[1].set_title("Top 25 bigrams"); axes[0].set_ylabel(""); axes[1].set_ylabel("")
    P.save(fig, "eda_03_top_terms_bigrams")

    # 04 distinctive terms per category ----------------------------------------------------
    dist_terms = distinctive_terms(df, n=12)
    save_table(dist_terms, "distinctive_terms_by_category", index=False)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, cat in zip(axes.ravel(), C.CATEGORIES):
        d = dist_terms[dist_terms[T] == cat].sort_values("chi2")
        ax.barh(d["term"], d["chi2"], color=C.PALETTE[cat])
        ax.set_title(cat, fontsize=11); ax.tick_params(axis="y", labelsize=9); ax.set_xlabel("chi²")
    fig.suptitle("Most distinctive terms per disease category (chi² on TF-IDF, one-vs-rest)", fontweight="bold")
    fig.tight_layout()
    P.save(fig, "eda_04_distinctive_terms")

    # per-category frequent terms table (raw frequency)
    rows = []
    for cat in C.CATEGORIES:
        t = top_terms(df.loc[df[T] == cat, C.CLEAN_TEXT_COL], 15)
        t.insert(0, T, cat); rows.append(t)
    save_table(pd.concat(rows), "top_terms_by_category", index=False)

    # 05 phase by category ------------------------------------------------------------------
    phase_order = ["Early Phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Not Applicable"]
    ph = _pct_table(df, "phase_label").reindex(columns=[p for p in phase_order if p in df.phase_label.unique()])
    save_table(ph, "phase_by_category_pct")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    sns.heatmap(ph, annot=True, fmt=".1f", cmap="Blues", ax=ax, cbar_kws={"label": "% of category"})
    ax.set_title("Trial phase mix by disease category (% of trials)"); ax.set_ylabel(""); ax.set_xlabel("")
    P.save(fig, "eda_05_phase_by_category")

    # 06 status by category -----------------------------------------------------------------
    status_group = df["overall_status"].map(lambda s: s if s in {"COMPLETED", "RECRUITING", "UNKNOWN", "TERMINATED",
                                                                   "WITHDRAWN", "ACTIVE_NOT_RECRUITING",
                                                                   "NOT_YET_RECRUITING"} else "OTHER")
    st = pd.crosstab(df[T], status_group, normalize="index").mul(100).round(1).loc[order]
    save_table(st, "status_by_category_pct")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    st.plot(kind="barh", stacked=True, ax=ax, colormap="tab20", width=0.8)
    ax.invert_yaxis(); ax.set_xlabel("% of trials"); ax.set_ylabel("")
    ax.set_title("Overall trial status by disease category")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    P.save(fig, "eda_06_status_by_category")

    # 07 study type, sex, healthy volunteers, age -------------------------------------------
    design = pd.DataFrame({
        "interventional_pct": df.assign(v=df.study_type.eq("INTERVENTIONAL")).groupby(T).v.mean() * 100,
        "female_only_pct": df.assign(v=df.sex.eq("FEMALE")).groupby(T).v.mean() * 100,
        "healthy_volunteers_pct": df.assign(v=df.healthy_volunteers.eq("Yes")).groupby(T).v.mean() * 100,
        "includes_children_pct": df.groupby(T).includes_children.mean() * 100,
        "completed_pct": df.assign(v=df.overall_status.eq("COMPLETED")).groupby(T).v.mean() * 100,
        "terminated_or_withdrawn_pct": df.assign(v=df.overall_status.isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"])).groupby(T).v.mean() * 100,
        "median_min_age": df.groupby(T).min_age_years.median(),
        "median_interventions": df.groupby(T).n_interventions.median(),
    }).round(1).loc[order]
    save_table(design, "trial_design_by_category")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    design[["interventional_pct", "healthy_volunteers_pct", "includes_children_pct", "female_only_pct"]].plot(
        kind="barh", ax=axes[0], width=0.8, color=["#4C78A8", "#54A24B", "#EECA3B", "#E45756"])
    axes[0].set_xlabel("% of trials"); axes[0].set_ylabel("")
    axes[0].set_title("Study design & population by category")
    axes[0].legend(["Interventional", "Healthy volunteers accepted", "Includes children (<18)", "Female only"],
                   fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    design[["completed_pct", "terminated_or_withdrawn_pct"]].plot(kind="barh", ax=axes[1], width=0.8,
                                                                 color=["#54A24B", "#E45756"])
    axes[1].set_xlabel("% of trials"); axes[1].set_ylabel("")
    axes[1].set_title("Completion vs termination/withdrawal/suspension")
    axes[1].legend(["Completed", "Terminated / withdrawn / suspended"], fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12), ncol=2)
    axes[0].set_ylim(len(design) - 0.5, -0.5)          # largest category on top (shared y-axis)
    fig.tight_layout()
    P.save(fig, "eda_07_design_population")

    # 08 age eligibility -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.5))
    sns.boxplot(data=df, y=T, x="min_age_years", order=order, palette=C.PALETTE, hue=T, legend=False,
                showfliers=False, ax=ax)
    ax.set_title("Minimum eligible age by category (years)"); ax.set_ylabel("")
    P.save(fig, "eda_08_min_age")

    # 09 own-disease mention & multi-disease summaries -------------------------------------
    mention = pd.DataFrame({
        "mentions_own_disease_pct": df.groupby(T).mentions_own_disease.mean() * 100,
        "mentions_2plus_categories_pct": df.assign(v=df.n_disease_categories_mentioned >= 2).groupby(T).v.mean() * 100,
    }).round(1).loc[order]
    save_table(mention, "disease_keyword_mentions")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    mention.plot(kind="barh", ax=ax, width=0.8, color=["#4C78A8", "#F58518"])
    ax.invert_yaxis(); ax.set_xlabel("% of summaries"); ax.set_ylabel("")
    ax.set_title("Summaries naming their own disease vs. mentioning 2+ disease categories")
    P.save(fig, "eda_09_disease_mentions")

    # 10 text-feature correlation ----------------------------------------------------------
    num = ["summary_word_count", "clean_token_count", "unique_token_count", "lexical_diversity",
           "avg_word_length", "n_conditions", "n_interventions", "min_age_years", "n_disease_categories_mentioned"]
    corr = df[num].corr().round(2)
    save_table(corr, "text_feature_correlation")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of text & trial features")
    P.save(fig, "eda_10_feature_correlation")

    # summary statistics
    save_table(df[num].describe().T.round(2), "summary_statistics")
    out.update({"design": design, "mention": mention, "distribution": dist})
    return out
