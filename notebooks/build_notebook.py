"""Generates notebooks/Clinical_Trial_NLP_Classification.ipynb (walk-through of the pipeline).

python notebooks/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/Clinical_Trial_NLP_Classification.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Clinical Trial Disease Category Classification Using NLP and Machine Learning\n"
       "End-to-end walk-through: CSV ➜ DataFrame ➜ cleaning ➜ text preprocessing (tokenization, stop words, "
       "lemmatization) ➜ TF-IDF ➜ model comparison ➜ evaluation ➜ insights.\n\n"
       "All logic lives in the `src/` package, so this notebook, `run_pipeline.py` and the Streamlit app share "
       "exactly the same code. Model tuning (≈10 min) is run by `python run_pipeline.py`; here we re-run the data "
       "steps and load the saved model and results."),
    code("import sys, json, warnings\nsys.path.insert(0, '..')\nwarnings.filterwarnings('ignore')\n"
         "import pandas as pd, numpy as np, joblib\nfrom IPython.display import Image, display\n"
         "from src import config as C, preprocessing, features, text_processing as tp\n"
         "pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200); "
         "pd.set_option('display.max_colwidth', 120)"),

    md("## 1. Data collection — CSV ➜ pandas DataFrame"),
    code("raw = preprocessing.load_raw()\nprint(raw.shape)\nraw.head(3)"),
    code("raw.info()"),
    code("preprocessing.profile(raw)"),
    md("The NLP input is **`brief_summary`** (no missing values). The disease-category label comes from "
       "**`source_condition_query`** — the condition each trial was retrieved for:"),
    code("raw.source_condition_query.value_counts()"),

    md("## 2. Data cleaning — missing values, duplicates, types, label encoding\n"
       "* trailing whitespace stripped, empty strings treated as missing\n"
       "* duplicates checked on full rows, `nct_id` and **summary text**. Identical summaries with *different* labels are "
       "ambiguous and are dropped; repeated summaries with the same label keep one copy (this also prevents the same text "
       "landing in both train and test)\n"
       "* missing values: `official_title` ← `title`, `phase` ← *Not Applicable* (observational / device studies), "
       "`interventions`/`eligibility_criteria` ← *Not specified*, `sex` ← mode, `healthy_volunteers` ← *Unknown*\n"
       "* ages converted to numeric years; a missing maximum age means *no upper limit* and is flagged rather than imputed\n"
       "* label encoded: readable `disease_category` + integer `label_id`"),
    code("clean, clog = preprocessing.clean(raw)\n"
         "{k: v for k, v in clog.items() if k not in ('missing_by_column_raw',)}"),
    code("print('missing after cleaning:')\nclean.isna().sum()[lambda s: s > 0]"),

    md("## 3. Text preprocessing & NLP operations\n"
       "Steps: **normalise** (ASCII, lowercase, URLs, unify *COVID-19* spellings) ➜ **remove punctuation & special "
       "characters** ➜ **tokenize** ➜ **remove stop words** (English + clinical boiler-plate such as *study, patient, "
       "trial*) ➜ **lemmatize** (NLTK WordNet)."),
    code("example = clean.brief_summary.iloc[0]\nsteps = tp.preprocess_steps(example)\n"
         "for k in ['normalized', 'cleaned']: print(f'{k}:', steps[k][:400], '\\n')\n"
         "for k in ['tokens', 'without_stopwords', 'lemmas']: print(f'{k} ({len(steps[k])}):', steps[k][:40], '\\n')"),
    code("feat = features.add_nlp_features(clean)\nfeat, n_empty = features.drop_empty_texts(feat)\n"
         "print('dropped (fewer than 3 tokens after cleaning):', n_empty, '| final rows:', len(feat))\n"
         "feat[['brief_summary', 'clean_text', 'summary_word_count', 'clean_token_count', 'lexical_diversity', "
         "'mentions_own_disease']].head()"),

    md("## 4. Exploratory data analysis\nFigures are produced by `src/eda.py`; tables are in `reports/eda_tables/`."),
    code("pd.read_csv(C.TABLES_DIR / 'category_distribution.csv')"),
    code("display(Image(C.FIGURES_DIR / 'eda_01_category_distribution.png'))"),
    code("display(Image(C.FIGURES_DIR / 'eda_02_summary_length.png'))\n"
         "pd.read_csv(C.TABLES_DIR / 'summary_length_by_category.csv')"),
    md("### Most common medical terms and keywords"),
    code("display(Image(C.FIGURES_DIR / 'eda_03_top_terms_bigrams.png'))"),
    code("display(Image(C.FIGURES_DIR / 'eda_04_distinctive_terms.png'))"),
    md("### Trial design patterns by disease category"),
    code("display(Image(C.FIGURES_DIR / 'eda_05_phase_by_category.png'))\n"
         "display(Image(C.FIGURES_DIR / 'eda_06_status_by_category.png'))"),
    code("display(Image(C.FIGURES_DIR / 'eda_07_design_population.png'))\n"
         "pd.read_csv(C.TABLES_DIR / 'trial_design_by_category.csv')"),
    code("display(Image(C.FIGURES_DIR / 'eda_08_min_age.png'))"),
    md("### Relationship between summary text and category"),
    code("display(Image(C.FIGURES_DIR / 'eda_09_disease_mentions.png'))\n"
         "pd.read_csv(C.TABLES_DIR / 'disease_keyword_mentions.csv')"),
    code("display(Image(C.FIGURES_DIR / 'eda_10_feature_correlation.png'))"),

    md("## 5. TF-IDF vectorization & feature engineering\n"
       "Unigrams + bigrams, sub-linear TF, `min_df`/`max_df` filtering. The tuned settings are stored in the model."),
    code("meta = json.loads(C.METADATA_PATH.read_text())\nprint(meta['best_params'])\n"
         "model = joblib.load(C.MODEL_PATH)\nprint(model)\n"
         "X = model.named_steps['tfidf'].transform(feat.clean_text.head(5))\n"
         "print('TF-IDF matrix for 5 summaries:', X.shape, '| non-zero:', X.nnz)"),

    md("## 6. Model comparison & evaluation\n"
       "Six classifiers (Multinomial NB, Complement NB, Logistic Regression, Linear SVM, SGD, Random Forest) tuned with "
       "`RandomizedSearchCV` (3-fold stratified CV, macro-F1). The best by CV score is evaluated on the 20% hold-out set."),
    code("pd.read_csv(C.REPORTS_DIR / 'model_comparison.csv').drop(columns='best_params')"),
    code("display(Image(C.FIGURES_DIR / 'model_01_comparison.png'))"),
    code("print(meta['best_model'], meta['test_metrics'])\n"
         "pd.read_csv(C.REPORTS_DIR / 'classification_report.csv', index_col=0)"),
    code("display(Image(C.FIGURES_DIR / 'model_02_confusion_matrix.png'))\n"
         "display(Image(C.FIGURES_DIR / 'model_03_per_class_metrics.png'))"),
    code("display(Image(C.FIGURES_DIR / 'model_04_top_terms_per_class.png'))"),
    md("### Robustness & leakage check\nDisease-name keywords are removed from train and test text and the model is "
       "retrained: if accuracy stayed high only because summaries name the disease, it would collapse."),
    code("display(Image(C.FIGURES_DIR / 'model_05_keyword_masking.png'))\nmeta['robustness_keyword_masking']"),
    code("display(Image(C.FIGURES_DIR / 'model_06_confidence_accuracy.png'))\n"
         "pd.read_csv(C.REPORTS_DIR / 'top_confusions.csv')"),

    md("## 7. Prediction with the saved end-to-end pipeline\nRaw text in ➜ category out (same code as the Streamlit app)."),
    code("samples = [\n"
         "    'This study evaluates whether adding empagliflozin to metformin improves HbA1c and weight in adults.',\n"
         "    'A phase 3 trial of trastuzumab deruxtecan versus chemotherapy in HER2-low metastatic disease.',\n"
         "    'Virtual reality distraction to reduce pre-operative worry and fear in children before surgery.',\n"
         "    'Hydroxyurea dose escalation to reduce vaso-occlusive pain crises in children with HbSS.',\n"
         "]\nproba = model.predict_proba(samples)\n"
         "pd.DataFrame({'summary': samples, 'predicted': model.classes_[proba.argmax(1)], 'confidence': proba.max(1).round(3)})"),

    md("## 8. Key insights\nSee `reports/EDA_Report.md` and `reports/Model_Evaluation_Report.md`, and the **Insights** page "
       "of the Streamlit app for findings computed live."),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "Clinical_Trial_NLP_Classification.ipynb"
nbf.write(nb, out)
print("wrote", out)
