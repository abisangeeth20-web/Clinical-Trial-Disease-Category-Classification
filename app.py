"""Streamlit app - Clinical Trial Disease Category Classification (NLP + Machine Learning).

Pages: Predict · Dashboard · Insights · Model Performance
Run:   streamlit run app.py
"""
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.text_processing import ensure_nltk, preprocess_steps
from src.train import _class_coefs

st.set_page_config(page_title="Clinical Trial Disease Classifier", page_icon="🧬", layout="wide")

GRID = "rgba(137,135,129,0.25)"
T = C.TARGET


# --------------------------------------------------------------------------- loaders
@st.cache_resource
def load_model():
    ensure_nltk()
    model = joblib.load(C.MODEL_PATH)
    meta = json.loads(C.METADATA_PATH.read_text(encoding="utf-8"))
    return model, meta


@st.cache_data
def load_data():
    df = pd.read_csv(C.DASHBOARD_GZ)
    df[C.CLEAN_TEXT_COL] = df[C.CLEAN_TEXT_COL].fillna("")
    return df


@st.cache_data
def load_table(name):
    path = C.TABLES_DIR / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_report_csv(name):
    path = C.REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data(show_spinner=False)
def term_counts(texts: tuple, n=20):
    cnt = Counter(w for t in texts for w in t.split())
    return pd.DataFrame(cnt.most_common(n), columns=["term", "count"])


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      font=dict(family="system-ui, Segoe UI, sans-serif"), title_x=0)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def explain(model, text, cls_idx, k=12):
    """Terms in the input that push the prediction toward the predicted class (TF-IDF value x class weight)."""
    clf, tfidf = model.named_steps["clf"], model.named_steps["tfidf"]
    coefs = _class_coefs(clf)
    if coefs is None:
        return pd.DataFrame()
    x = tfidf.transform(model.named_steps["preprocess"].transform([text]))
    idx = x.indices
    contrib = x.data * coefs[cls_idx][idx]
    names = tfidf.get_feature_names_out()
    out = pd.DataFrame({"term": names[idx], "contribution": contrib})
    return out[out.contribution > 0].nlargest(k, "contribution")


# --------------------------------------------------------------------------- Predict
def page_predict(model, meta):
    st.title("🧬 Clinical Trial Disease Category Predictor")
    st.caption("Paste a clinical-trial **Brief Summary**. The saved pipeline runs the same NLP steps used in "
               "training (normalisation → cleaning → tokenization → stop-word removal → lemmatization → TF-IDF) "
               f"and the **{meta['best_model']}** classifier predicts one of 8 disease categories.")

    if "summary_text" not in st.session_state:
        st.session_state.summary_text = meta["examples"][C.CATEGORIES[0]][0]
    c1, c2 = st.columns([3, 1])
    ex_cat = c1.selectbox("Load an example summary from the test set", ["—"] + C.CATEGORIES)
    if c2.button("Load example", width="stretch") and ex_cat != "—":
        exs = meta["examples"].get(ex_cat, [])
        if exs:
            st.session_state.summary_text = exs[np.random.randint(len(exs))]

    text = st.text_area("Brief summary", key="summary_text", height=180,
                        placeholder="e.g. This randomized trial evaluates metformin plus lifestyle coaching on HbA1c ...")
    go_btn = st.button("Predict disease category", type="primary")

    if go_btn or text:
        if not text or not text.strip():
            st.warning("Please enter a summary.")
            return
        steps = preprocess_steps(text)
        if len(steps["lemmas"]) < 3:
            st.warning("After cleaning, the text has fewer than 3 meaningful words. Add more medical detail "
                       "for a reliable prediction.")
            return
        proba = model.predict_proba([text])[0]
        classes = list(model.classes_)
        order = np.argsort(proba)[::-1]
        top, conf = classes[order[0]], proba[order[0]]

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted category", top)
        c2.metric("Confidence", f"{conf:.1%}")
        c3.metric("Runner-up", f"{classes[order[1]]} ({proba[order[1]]:.1%})")
        if conf < 0.5:
            st.warning("Low confidence: the summary may describe several conditions (e.g. anxiety in cancer "
                       "patients) or a disease outside the 8 trained categories.")
        elif conf >= 0.9:
            st.success("High confidence — on the test set, predictions at ≥ 90% confidence are almost always correct.")

        c1, c2 = st.columns(2)
        pdf = pd.DataFrame({"category": classes, "probability": proba}).sort_values("probability")
        fig = px.bar(pdf, x="probability", y="category", orientation="h", color="category",
                     color_discrete_map=C.PALETTE, text=pdf.probability.map(lambda v: f"{v:.1%}"))
        fig.update_layout(showlegend=False, xaxis_tickformat=".0%", xaxis_range=[0, 1.1])
        c1.plotly_chart(style(fig.update_layout(title="Probability for each category")), width="stretch")
        ex = explain(model, text, classes.index(top))
        if len(ex):
            fig = px.bar(ex.sort_values("contribution"), x="contribution", y="term", orientation="h",
                         color_discrete_sequence=[C.PALETTE[top]])
            c2.plotly_chart(style(fig.update_layout(title=f"Words driving the '{top}' prediction")), width="stretch")

        with st.expander("Show the NLP preprocessing steps for this text"):
            st.markdown(f"**1. Normalised:** {steps['normalized'][:600]}")
            st.markdown(f"**2. Punctuation / special characters removed:** {steps['cleaned'][:600]}")
            st.markdown(f"**3. Tokens ({len(steps['tokens'])}):** `{steps['tokens'][:60]}`")
            st.markdown(f"**4. After stop-word removal ({len(steps['without_stopwords'])}):** `{steps['without_stopwords'][:60]}`")
            st.markdown(f"**5. Lemmatized ({len(steps['lemmas'])}) → TF-IDF input:** `{steps['clean_text'][:600]}`")

    st.divider()
    st.subheader("Batch prediction (CSV upload)")
    st.caption("Upload a CSV with a `brief_summary` column (max 5,000 rows). Predictions can be downloaded.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            bdf = pd.read_csv(up)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the file: {e}")
            return
        if C.TEXT_COL not in bdf.columns:
            st.error(f"Column `{C.TEXT_COL}` not found. Columns present: {list(bdf.columns)[:15]}")
            return
        bdf = bdf.head(5000).copy()
        bdf[C.TEXT_COL] = bdf[C.TEXT_COL].fillna("").astype(str)
        with st.spinner("Predicting ..."):
            p = model.predict_proba(bdf[C.TEXT_COL])
        bdf["predicted_category"] = np.array(model.classes_)[p.argmax(axis=1)]
        bdf["confidence"] = p.max(axis=1).round(3)
        st.dataframe(bdf[[C.TEXT_COL, "predicted_category", "confidence"] +
                         ([T] if T in bdf.columns else [])].head(200), width="stretch", hide_index=True)
        if T in bdf.columns:
            st.info(f"Accuracy against the `{T}` column in the file: {(bdf[T] == bdf.predicted_category).mean():.1%}")
        st.plotly_chart(style(px.histogram(bdf, x="predicted_category", color="predicted_category",
                                           color_discrete_map=C.PALETTE, title="Predicted category counts")
                              .update_layout(showlegend=False)), width="stretch")
        st.download_button("Download predictions CSV", bdf.to_csv(index=False).encode(), "predictions.csv", "text/csv")


# --------------------------------------------------------------------------- Dashboard
def page_dashboard(df):
    st.title("📊 Clinical Trial Dashboard")
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        cats = c1.multiselect("Disease category", C.CATEGORIES, default=C.CATEGORIES)
        types = c2.multiselect("Study type", sorted(df.study_type.unique()), default=sorted(df.study_type.unique()))
        phase_opts = ["Early Phase 1", "Phase 1", "Phase 1/2", "Phase 2", "Phase 2/3", "Phase 3", "Phase 4", "Not Applicable"]
        phases = c3.multiselect("Phase", phase_opts, default=phase_opts)
        statuses = c4.multiselect("Status", sorted(df.overall_status.unique()), default=sorted(df.overall_status.unique()))
    d = df[df[T].isin(cats) & df.study_type.isin(types) & df.phase_label.isin(phases) & df.overall_status.isin(statuses)]
    if d.empty:
        st.warning("No trials match the filters.")
        return

    k = st.columns(5)
    k[0].metric("Trials", f"{len(d):,}")
    k[1].metric("Completed", f"{(d.overall_status == 'COMPLETED').mean():.1%}")
    k[2].metric("Interventional", f"{(d.study_type == 'INTERVENTIONAL').mean():.1%}")
    k[3].metric("Median summary length", f"{d.summary_word_count.median():.0f} words")
    k[4].metric("Include children", f"{d.includes_children.mean():.1%}")

    c1, c2 = st.columns(2)
    cnt = d[T].value_counts().reset_index()
    fig = px.bar(cnt, x="count", y=T, orientation="h", color=T, color_discrete_map=C.PALETTE, text="count")
    c1.plotly_chart(style(fig.update_layout(title="Trials per disease category", showlegend=False,
                                            yaxis={"categoryorder": "total ascending"}, yaxis_title="")), width="stretch")
    ph = d.groupby([T, "phase_label"]).size().reset_index(name="trials")
    fig = px.bar(ph, x="trials", y=T, color="phase_label", orientation="h",
                 category_orders={"phase_label": phase_opts}, barmode="stack")
    c2.plotly_chart(style(fig.update_layout(title="Phase mix by category", yaxis_title="")), width="stretch")

    c1, c2 = st.columns(2)
    fig = px.box(d, x="summary_word_count", y=T, color=T, color_discrete_map=C.PALETTE, points=False)
    fig.update_xaxes(range=[0, min(800, d.summary_word_count.quantile(0.99))])
    c1.plotly_chart(style(fig.update_layout(title="Brief-summary length (words)", showlegend=False, yaxis_title="")),
                    width="stretch")
    st_tab = d.groupby([T, "overall_status"]).size().reset_index(name="trials")
    fig = px.bar(st_tab, x="trials", y=T, color="overall_status", orientation="h", barmode="stack")
    c2.plotly_chart(style(fig.update_layout(title="Trial status by category", yaxis_title="")), width="stretch")

    c1, c2 = st.columns(2)
    tc = term_counts(tuple(d[C.CLEAN_TEXT_COL].tolist()), 20)
    fig = px.bar(tc.sort_values("count"), x="count", y="term", orientation="h", color_discrete_sequence=["#4C78A8"])
    c1.plotly_chart(style(fig.update_layout(title="Most frequent medical terms (filtered trials)"), 480), width="stretch")
    sex = d.groupby([T, "sex"]).size().reset_index(name="trials")
    fig = px.bar(sex, x="trials", y=T, color="sex", orientation="h", barmode="stack",
                 color_discrete_map={"ALL": "#4C78A8", "FEMALE": "#E45756", "MALE": "#72B7B2"})
    c2.plotly_chart(style(fig.update_layout(title="Eligible sex by category", yaxis_title=""), 480), width="stretch")

    st.subheader("Search trials")
    q = st.text_input("Keyword in title (e.g. metformin, vaccine, virtual reality)")
    res = d[d.title.str.contains(q, case=False, na=False)] if q else d
    st.dataframe(res[["nct_id", "title", T, "phase_label", "overall_status", "clinicaltrials_url"]].head(300),
                 width="stretch", hide_index=True,
                 column_config={"clinicaltrials_url": st.column_config.LinkColumn("link", display_text="open")})
    st.caption(f"{len(res):,} matching trials (first 300 shown).")


# --------------------------------------------------------------------------- Insights
def page_insights(df, meta):
    st.title("💡 Insights & Recommendations")
    st.caption("All numbers below are computed live from the cleaned dataset and the saved model's test results.")
    n = len(df)
    counts = df[T].value_counts()

    st.header("1. Disease category distribution")
    c1, c2 = st.columns([1, 1.3])
    c1.markdown(f"""
* **{n:,} trials** across **8 disease categories** after cleaning.
* **{counts.index[0]}** is the largest category with **{counts.iloc[0]:,} trials ({counts.iloc[0] / n:.1%})**;
  **{counts.index[-1]}** is the smallest with **{counts.iloc[-1]:,} ({counts.iloc[-1] / n:.1%})**.
* Imbalance ratio largest : smallest = **{counts.iloc[0] / counts.iloc[-1]:.1f} : 1**. The top 3 categories hold
  **{counts.iloc[:3].sum() / n:.1%}** of all trials, so **macro-averaged** metrics are used to judge the model fairly.
""")
    fig = px.pie(counts.reset_index(), values="count", names=T, color=T, color_discrete_map=C.PALETTE, hole=0.45)
    c2.plotly_chart(style(fig.update_layout(title="Share of trials"), 330), width="stretch")

    st.header("2. Frequently occurring medical terms")
    tc = term_counts(tuple(df[C.CLEAN_TEXT_COL].tolist()), 15)
    dterms = load_table("distinctive_terms_by_category")
    c1, c2 = st.columns([1, 1.4])
    c1.markdown("**Most frequent terms overall:** " + ", ".join(f"`{t}`" for t in tc.term))
    c1.markdown("Generic treatment words (`treatment`, `therapy`, `control`, `intervention`) dominate overall, but "
                "they appear in every category and carry little class signal — TF-IDF down-weights them.")
    if len(dterms):
        top5 = dterms[dterms["rank"] <= 6].groupby(T)["term"].apply(lambda s: ", ".join(s)).reindex(C.CATEGORIES)
        c2.dataframe(top5.reset_index().rename(columns={"term": "most distinctive terms (chi²)"}),
                     hide_index=True, width="stretch")
    st.markdown("Beyond disease names, the distinctive vocabulary reveals **standard treatments**: "
                "`metformin`/`insulin` (diabetes), `methotrexate`/`adalimumab` (RA), `hydroxyurea` (sickle cell), "
                "`chemotherapy`/`her2`/`neoadjuvant` (breast cancer), `intraocular pressure` (glaucoma), "
                "`vaccine` (COVID-19) and `virtual reality`/`depression` (anxiety).")

    st.header("3. Clinical-trial text patterns")
    med = df.groupby(T).summary_word_count.median().sort_values()
    own = df.groupby(T).mentions_own_disease.mean().sort_values()
    c1, c2 = st.columns(2)
    c1.markdown(f"""
* Median summary length is **{df.summary_word_count.median():.0f} words**; **{med.index[-1]}** summaries are the
  longest (median {med.iloc[-1]:.0f}) and **{med.index[0]}** the shortest (median {med.iloc[0]:.0f}).
* **{df.mentions_own_disease.mean():.1%}** of summaries explicitly name their own disease. The lowest is
  **{own.index[0]} ({own.iloc[0]:.1%})** — anxiety is often studied as a *symptom* of other conditions
  (surgery, cancer, COVID-19, dental procedures), so it is the category other trials are most often confused with.
* Text cleaning removes about half of the words (stop words + boiler-plate): median
  {df.summary_word_count.median():.0f} → {df.clean_token_count.median():.0f} tokens.
""")
    fig = px.bar((own * 100).round(1).reset_index(), x="mentions_own_disease", y=T, orientation="h", color=T,
                 color_discrete_map=C.PALETTE, labels={"mentions_own_disease": "% naming own disease"})
    c2.plotly_chart(style(fig.update_layout(title="Summaries that name their own disease", showlegend=False,
                                            yaxis_title=""), 320), width="stretch")

    st.header("4. Trial design patterns")
    g = df.groupby(T)
    design = pd.DataFrame({
        "female_only": g.sex.apply(lambda s: (s == "FEMALE").mean()),
        "children": g.includes_children.mean(),
        "interventional": g.study_type.apply(lambda s: (s == "INTERVENTIONAL").mean()),
        "completed": g.overall_status.apply(lambda s: (s == "COMPLETED").mean()),
        "stopped": g.overall_status.apply(lambda s: s.isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"]).mean()),
        "healthy": df.assign(v=df.healthy_volunteers.eq("Yes")).groupby(T).v.mean(),
    })
    st.markdown(f"""
* **{design.female_only.idxmax()}**: {design.female_only.max():.1%} of trials enrol women only — every other category is below {design.female_only.drop(design.female_only.idxmax()).max():.0%}.
* **{design.children.idxmax()}** is the most paediatric category — {design.children.max():.1%} of trials accept children (< 18), vs {df.includes_children.mean():.1%} overall.
* **{design.interventional.idxmin()}** has the lowest share of interventional studies ({design.interventional.min():.1%}) — many are observational
  (registries, cohort follow-up), while **{design.interventional.idxmax()}** is the most interventional ({design.interventional.max():.1%}).
* Highest completion rate: **{design.completed.idxmax()} ({design.completed.max():.1%})**; highest terminated/withdrawn/suspended rate:
  **{design.stopped.idxmax()} ({design.stopped.max():.1%})**.
* **{design.healthy.idxmax()}** trials accept healthy volunteers most often ({design.healthy.max():.1%}), vs {df.healthy_volunteers.eq('Yes').mean():.1%} overall.
""")
    dd = (design * 100).round(1).reset_index().melt(id_vars=T, var_name="metric", value_name="pct")
    fig = px.bar(dd[dd.metric.isin(["female_only", "children", "stopped"])], x="pct", y=T, color="metric",
                 barmode="group", orientation="h")
    st.plotly_chart(style(fig.update_layout(title="Female-only, paediatric and stopped trials (% of category)",
                                            yaxis_title=""), 380), width="stretch")

    st.header("5. Model performance & prediction accuracy")
    m = meta["test_metrics"]
    rep = meta["classification_report"]
    f1 = pd.Series({c: rep[c]["f1-score"] for c in C.CATEGORIES}).sort_values()
    prec = pd.Series({c: rep[c]["precision"] for c in C.CATEGORIES}).sort_values()
    conf = meta["top_confusions"][0] if meta["top_confusions"] else None
    rob = meta["robustness_keyword_masking"]
    calib = pd.DataFrame(meta["confidence_vs_accuracy"])
    hi = calib[calib["bin"] == ">=0.9"]
    st.markdown(f"""
* **{meta['best_model']}** on TF-IDF features reaches **{m['accuracy']:.1%} accuracy** and **{m['f1_macro']:.3f} macro-F1**
  on {meta['test_size']:,} unseen summaries (precision {m['precision_macro']:.3f}, recall {m['recall_macro']:.3f}).
* Best-recognised category: **{f1.index[-1]}** (F1 {f1.iloc[-1]:.3f}); hardest: **{f1.index[0]}** (F1 {f1.iloc[0]:.3f}).
""" + (f"* Most common mistake: **{conf['true']} → {conf['predicted']}** ({conf['count']} test trials). Many such summaries discuss both conditions, or are really about mental "
     f"health and were retrieved under another condition (label noise). "
     f"Lowest precision: **{prec.index[0]}** ({prec.iloc[0]:.3f}), i.e. the class other trials are most often pulled into.\n" if conf else "") +
    (f"* Predictions with ≥ 90% confidence cover **{hi['n'].iloc[0] / meta['test_size']:.1%}** of test trials and are **{hi['accuracy'].iloc[0]:.1%}** accurate — confidence is a reliable triage signal.\n" if len(hi) else "") +
    f"""* **Robustness:** with every disease-name keyword removed, the retrained model still reaches
  **{rob['retrained_on_masked_text']['accuracy']:.1%}** accuracy (vs {rob['original_test']['accuracy']:.1%}) — it learns treatments,
  biomarkers and context, not just disease names.
""")

    st.header("6. Recommendations for healthcare analytics & research")
    st.markdown(f"""
1. **Auto-tag new trial registrations** with the classifier and route only low-confidence (< 50%) predictions to a
   human curator — most of the volume can be tagged automatically at ≥ 90% confidence.
2. **Treat anxiety as a cross-cutting condition.** It co-occurs with surgery, cancer and chronic disease; a
   multi-label scheme (one trial → several categories) would describe these trials better than a single label.
3. **Use the term profiles for portfolio analysis** — e.g. track `vaccine` vs treatment trials in COVID-19, or
   biologic (`adalimumab`) vs conventional (`methotrexate`) therapy in RA, to spot research gaps.
4. **Under-represented areas:** {counts.index[-1]} and {counts.index[-2]} have the fewest trials; with
   {design.children.max():.0%} paediatric trials, {design.children.idxmax()} research especially depends on child-friendly study designs.
5. **Monitor trial risk:** {design.stopped.idxmax()} shows the highest terminated/withdrawn share — sponsors can use
   these patterns to plan recruitment and feasibility early.
6. **Extend the model** with the `conditions` and `eligibility_criteria` text, or with transformer embeddings
   (BioBERT / ClinicalBERT), for multi-label or finer-grained disease classification.
""")


# --------------------------------------------------------------------------- Model performance
def page_model(meta):
    st.title("📈 Model Performance")
    m = meta["test_metrics"]
    st.caption(f"Best model: **{meta['best_model']}** ({meta['final_estimator']}) · TF-IDF vocabulary "
               f"{meta['tfidf_vocabulary_size']:,} terms · train {meta['train_size']:,} / test {meta['test_size']:,} "
               f"summaries (stratified 80/20).")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{m['accuracy']:.4f}")
    k[1].metric("Precision (macro)", f"{m['precision_macro']:.4f}")
    k[2].metric("Recall (macro)", f"{m['recall_macro']:.4f}")
    k[3].metric("F1 (macro)", f"{m['f1_macro']:.4f}")
    k[4].metric("ROC-AUC (OvR)", f"{m['roc_auc_ovr_macro']:.4f}")

    st.subheader("Model comparison (RandomizedSearchCV, 3-fold CV, scored on macro-F1)")
    comp = pd.DataFrame(meta["model_comparison"])
    show = comp[["model", "cv_f1_macro", "cv_f1_std", "test_accuracy", "test_precision_macro", "test_recall_macro",
                 "test_f1_macro", "test_f1_weighted", "test_roc_auc_ovr_macro", "tuning_seconds", "best_params"]]
    st.dataframe(show, hide_index=True, width="stretch")
    long = comp.melt(id_vars="model", value_vars=["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"],
                     var_name="metric", value_name="score")
    fig = px.bar(long, x="model", y="score", color="metric", barmode="group")
    fig.update_yaxes(range=[max(0, long.score.min() - 0.05), 1])
    st.plotly_chart(style(fig.update_layout(title="Test-set metrics by model", xaxis_title="")), width="stretch")

    c1, c2 = st.columns([1.2, 1])
    cm = np.array(meta["confusion_matrix"])
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig = go.Figure(go.Heatmap(z=cmn, x=C.CATEGORIES, y=C.CATEGORIES, colorscale="Blues",
                               text=cm, texttemplate="%{text}", hovertemplate="actual %{y}<br>predicted %{x}<br>recall share %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title="Actual"); fig.update_xaxes(title="Predicted")
    c1.plotly_chart(style(fig.update_layout(title="Confusion matrix (colour = row-normalised)"), 480), width="stretch")
    rep = pd.DataFrame(meta["classification_report"]).T.loc[C.CATEGORIES + ["macro avg", "weighted avg"]]
    c2.markdown("**Per-class report**")
    c2.dataframe(rep.round(3), width="stretch")

    c1, c2 = st.columns(2)
    rob = pd.DataFrame(meta["robustness_keyword_masking"]).T.reset_index(names="setting")
    rob = rob.melt(id_vars="setting", var_name="metric", value_name="score")
    fig = px.bar(rob, x="setting", y="score", color="metric", barmode="group", text=rob.score.map(lambda v: f"{v:.3f}"))
    c1.plotly_chart(style(fig.update_layout(title="Robustness: disease-name keywords removed", xaxis_title="",
                                            yaxis_range=[0, 1.08])), width="stretch")
    cal = pd.DataFrame(meta["confidence_vs_accuracy"])
    fig = px.bar(cal, x="bin", y="accuracy", text=cal.apply(lambda r: f"{r.accuracy:.2f} (n={r.n})", axis=1))
    c2.plotly_chart(style(fig.update_layout(title="Prediction confidence vs accuracy", xaxis_title="confidence",
                                            yaxis_range=[0, 1.1])), width="stretch")

    st.subheader("Strongest terms per category (model weights)")
    tt = meta.get("top_terms_per_class", {})
    if tt:
        st.dataframe(pd.DataFrame({k: v[:12] for k, v in tt.items()}), hide_index=True, width="stretch")

    st.subheader("Most frequent confusions & misclassified examples")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(pd.DataFrame(meta["top_confusions"]), hide_index=True, width="stretch")
    mis = load_report_csv("misclassified_examples.csv")
    if len(mis):
        c2.dataframe(mis.head(50), hide_index=True, width="stretch")


# --------------------------------------------------------------------------- main
def main():
    model, meta = load_model()
    df = load_data()
    st.sidebar.title("🧬 Clinical Trial NLP")
    page = st.sidebar.radio("Go to", ["🔮 Predict", "📊 Dashboard", "💡 Insights", "📈 Model Performance"])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model:** {meta['best_model']}  \n**Test accuracy:** {meta['test_metrics']['accuracy']:.2%}  \n"
                        f"**Macro F1:** {meta['test_metrics']['f1_macro']:.3f}  \n**Trials:** {len(df):,}")
    st.sidebar.caption("Data: ClinicalTrials.gov summaries for 8 conditions. For research/education — not medical advice.")
    if page.endswith("Predict"):
        page_predict(model, meta)
    elif page.endswith("Dashboard"):
        page_dashboard(df)
    elif page.endswith("Insights"):
        page_insights(df, meta)
    else:
        page_model(meta)


main()
