# Clinical Trial Disease Category Classification Using NLP and Machine Learning

An end-to-end NLP and machine-learning pipeline that reads the **Brief Summary** of a clinical trial and predicts its **disease category**: Breast Cancer, Type 2 Diabetes, COVID-19, Anxiety, COPD, Rheumatoid Arthritis, Glaucoma or Sickle Cell Anemia. It covers CSV ➜ DataFrame conversion, cleaning, medical text preprocessing (tokenization, stop-word removal, lemmatization), TF-IDF feature engineering, a six-model comparison with tuning, evaluation, EDA insights and an interactive Streamlit app.

**Domain:** Healthcare Analytics · Medical NLP · Clinical Trial Intelligence
**Stack:** Python, Pandas, NumPy, NLTK, scikit-learn, Matplotlib, Seaborn, Plotly, Streamlit

🔗 **Live app:** _LIVE_APP_LINK_

---

## Results at a glance

| Item | Result |
|---|---|
| Dataset | 60,337 raw trials → **60,024** after cleaning · 8 disease categories (imbalance 14.3 : 1) |
| Best model | **TF-IDF (1–2 grams) + Linear SVM** (calibrated), best of 6 tuned models |
| Test **Accuracy** | **0.9586** |
| Test **Precision / Recall / F1** (macro) | **0.9619 / 0.9485 / 0.9549** |
| ROC-AUC (one-vs-rest) | 0.9978 |
| Per-class F1 | 0.936 (COPD) – 0.975 (Glaucoma) |
| Robustness | **90.5%** accuracy even with every disease-name keyword removed (retrained) |
| Confidence | 82.7% of predictions have ≥ 0.9 probability and are 99.6% accurate |

Exact figures for every model are in [reports/Model_Evaluation_Report.md](reports/Model_Evaluation_Report.md). Insights and patterns are in [reports/EDA_Report.md](reports/EDA_Report.md).

## Project deliverables

| Deliverable | Where |
|---|---|
| Clinical Trial Dataset (CSV raw data file) | [`data/raw/`](data/raw/): the original 158 MB CSV split into 3 gzip parts (< 25 MB each, identical rows) |
| Cleaned and preprocessed dataset (CSV / DataFrame) | [`data/processed/cleaned_clinical_trials_part1-3.csv.gz`](data/processed/) (cleaned), [`preprocessed_clinical_trials.csv.gz`](data/processed/preprocessed_clinical_trials.csv.gz) (NLP-ready `clean_text` + text features) |
| Python script for text preprocessing and NLP operations | [`src/text_processing.py`](src/text_processing.py) |
| TF-IDF vectorization and feature engineering | [`src/features.py`](src/features.py) |
| ML model for disease category classification | [`src/train.py`](src/train.py) → [`models/disease_classifier.joblib`](models/disease_classifier.joblib) + [`model_metadata.json`](models/model_metadata.json) |
| Model evaluation report with performance metrics | [`reports/Model_Evaluation_Report.md`](reports/Model_Evaluation_Report.md) |
| Data visualization (Plotly / Seaborn / Matplotlib) | [`reports/figures/`](reports/figures/) (Matplotlib/Seaborn) and the Streamlit app (Plotly) |
| Interactive Streamlit application | [`app.py`](app.py) (live link above) |
| EDA report with clinical trial insights and patterns | [`reports/EDA_Report.md`](reports/EDA_Report.md) |
| End-to-end NLP pipeline | [`run_pipeline.py`](run_pipeline.py) and the executed notebook [`notebooks/Clinical_Trial_NLP_Classification.ipynb`](notebooks/Clinical_Trial_NLP_Classification.ipynb) |
| Final project documentation | this README + [`reports/Project_Documentation.md`](reports/Project_Documentation.md) |

## Project structure

```
├── data/
│   ├── raw/clinical_trials_raw_part1-3.csv.gz        raw dataset (as delivered, split for GitHub)
│   └── processed/
│       ├── cleaned_clinical_trials_part1-3.csv.gz    cleaned dataset (all columns)
│       ├── preprocessed_clinical_trials.csv.gz       brief_summary + clean_text + text features + label
│       └── dashboard_data.csv.gz                     compact file used by the app
├── run_pipeline.py                 one command: load → clean → NLP → EDA → TF-IDF + train → evaluate → save
├── app.py                          Streamlit app (Predict · Dashboard · Insights · Model Performance)
├── src/
│   ├── config.py                   paths, labels, disease keywords, palette
│   ├── preprocessing.py            CSV loading, profiling, cleaning (missing values, duplicates, types, labels)
│   ├── text_processing.py          normalise → clean → tokenize → stop words → lemmatize (+ sklearn transformer)
│   ├── features.py                 clean_text + text features, TF-IDF builder, term analytics, keyword masking
│   ├── eda.py                      EDA figures + tables
│   ├── train.py                    6-model comparison, tuning, evaluation, robustness test, persistence
│   └── plotting.py                 shared chart style
├── models/
│   ├── disease_classifier.joblib   full pipeline: TextPreprocessor → TF-IDF → classifier (raw text in, category out)
│   └── model_metadata.json         metrics, comparison, confusion matrix, top terms, examples
├── notebooks/
│   ├── Clinical_Trial_NLP_Classification.ipynb   executed walk-through
│   └── build_notebook.py                         regenerates the notebook
└── reports/
    ├── EDA_Report.md · Model_Evaluation_Report.md · Project_Documentation.md
    ├── model_comparison.csv · classification_report.csv · confusion_matrix.csv · top_confusions.csv
    ├── misclassified_examples.csv · confidence_vs_accuracy.csv · raw_data_profile.csv · pipeline_summary.json
    ├── eda_tables/*.csv
    └── figures/*.png
```

## How to run

```bash
pip install -r requirements-dev.txt   # requirements.txt alone is enough for the app
python run_pipeline.py                # ~15 min on 4 cores: data, figures, model and reports
streamlit run app.py                  # http://localhost:8501
```

`run_pipeline.py` reads the original CSV if it is in the project root, otherwise the gzip parts in `data/raw/`. NLTK's WordNet data downloads automatically into `nltk_data/` on first use.
To rerun the notebook: `python -m nbconvert --to notebook --execute --inplace notebooks/Clinical_Trial_NLP_Classification.ipynb`

## Approach

### 1. Data collection
The CSV is loaded with Pandas into a DataFrame of 60,337 trials × 16 columns. `brief_summary` is the NLP input and `source_condition_query` (8 values) is the disease-category label.

### 2. Data preprocessing
* **Missing values:** `phase` → *Not Applicable*, `official_title` ← `title`, `interventions` / `eligibility_criteria` → *Not specified*, `sex` → mode, `healthy_volunteers` → *Unknown*. Ages are parsed to numeric years, and a missing maximum age is flagged as *no upper limit*.
* **Duplicates:** checked on full rows, `nct_id` and summary text. 30 identical summaries with conflicting labels were dropped, and 262 repeated summaries were reduced to one copy each, so no text appears in both train and test.
* **Text cleaning & normalisation:** unicode → ASCII, lowercase, URL and e-mail removal, and all COVID spellings unified to `covid-19`.
* **Removal of punctuation, special characters and stand-alone numbers.**
* **Tokenization:** a regex tokenizer that keeps medical tokens such as `covid-19`, `hba1c`, `her2` and `sars-cov-2`.
* **Stop words:** the English list plus clinical boiler-plate (*study, patient, trial, evaluate, randomized …*).
* **Lemmatization:** NLTK WordNet, noun pass then verb pass (*studies → study, tumors → tumor, using → use*).
* **Label encoding:** readable `disease_category` and integer `label_id`. 21 summaries with fewer than 3 meaningful tokens were removed, leaving **60,024 trials**.

### 3. Feature engineering
* **TF-IDF** on unigrams + bigrams with sub-linear TF and `min_df` / `max_df` filtering. The n-gram range and `min_df` were tuned.
* Text features for EDA: word and token counts, lexical diversity, average word length, *mentions own disease* and *number of disease categories mentioned*.
* All preprocessing is wrapped in a scikit-learn `TextPreprocessor` and saved **inside** the model pipeline, so the app applies exactly the same steps to user text.

### 4. EDA
Category distribution, summary-length patterns, most frequent terms and bigrams, chi² distinctive terms per disease, phase, status, sex, age and design patterns, and keyword-mention analysis. See [reports/EDA_Report.md](reports/EDA_Report.md).

### 5–6. NLP-based classification & evaluation
* Stratified 80/20 split (`random_state=42`).
* Six models were compared: **Multinomial NB, Complement NB, Logistic Regression, Linear SVM, SGD (modified Huber) and Random Forest**. Each was tuned with `RandomizedSearchCV` (3-fold stratified CV, scored on macro-F1), and the best was chosen by CV score.
* Metrics:
  * **Accuracy, Precision, Recall and F1** (macro and weighted)
  * ROC-AUC (one-vs-rest)
  * per-class report and confusion matrix
  * confidence-vs-accuracy analysis
* **Leakage and robustness check:** disease-name keywords were removed from the text and the model retrained, to show it learns context (drugs, biomarkers, procedures) rather than just disease names.

### 7. Streamlit app
* **Predict:**
  * paste a summary, or load a test-set example
  * shows the predicted category, its confidence and the probabilities for all 8 classes
  * shows the **words driving the prediction**, every NLP step, and a low-confidence warning
  * **batch CSV prediction** with download
* **Dashboard:** filters by category, study type, phase and status, with KPIs, Plotly charts, live term frequencies and trial search.
* **Insights:** findings computed live on category distribution, frequent medical terms, text patterns, trial-design patterns and model performance, plus recommendations.
* **Model Performance:** model comparison, metrics, confusion matrix, per-class report, robustness test, confidence calibration, top terms and misclassified examples.

## Deploy (permanent public link) — Streamlit Community Cloud

1. Upload this folder to a GitHub repository. Every file is under 25 MB. The original 158 MB CSV and the uncompressed cleaned CSV are git-ignored, and their gzip parts are committed instead.
2. Go to https://share.streamlit.io, choose **Create app**, select the repository and branch `main`, and set the main file to `app.py`.
3. Under **Advanced settings**, choose **Python 3.13**. `requirements.txt` pins the exact library versions the model was trained with.
4. Click **Deploy** to get a permanent `https://<name>.streamlit.app` URL.

Free apps sleep after a period with no visitors. When that happens, click **"Yes, get this app back up"** and the app wakes in about 30–60 seconds. The URL never changes.
