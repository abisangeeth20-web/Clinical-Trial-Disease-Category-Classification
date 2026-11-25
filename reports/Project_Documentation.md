# Final Project Documentation
## Clinical Trial Disease Category Classification Using NLP and Machine Learning

| | |
|---|---|
| **Domain** | Healthcare Analytics, Medical NLP & Clinical Trial Intelligence Systems |
| **Goal** | Classify clinical trials into disease categories from the free-text **Brief Summary** |
| **Data** | 60,337 ClinicalTrials.gov records, 16 columns, 8 disease categories |
| **Best model** | TF-IDF (unigrams + bigrams) + Linear SVM (sigmoid-calibrated for probabilities) |
| **Test performance** | Accuracy **0.9586** · Precision (macro) **0.9619** · Recall (macro) **0.9485** · F1 (macro) **0.9549** · ROC-AUC (OvR) **0.9978** |
| **Deployment** | Streamlit Community Cloud. Links are in the README. |

---

## 1. Problem statement
Clinical-trial registries hold large amounts of unstructured text: summaries, eligibility criteria and intervention details. The task is to turn this text into structured features and build an NLP + ML system that assigns each trial's *Brief Summary* to the correct **disease category**. This supports research organisation, portfolio analysis and automatic tagging of new registrations.

## 2. Dataset
| Column group | Columns |
|---|---|
| Identifier | `nct_id`, `clinicaltrials_url` |
| Text | `title`, `official_title`, **`brief_summary`** (NLP input), `conditions`, `interventions`, `eligibility_criteria` |
| Label | **`source_condition_query`**: breast cancer, type 2 diabetes, covid-19, anxiety, chronic obstructive pulmonary disease, rheumatoid arthritis, glaucoma, sickle cell anemia |
| Trial attributes | `overall_status`, `study_type`, `phase`, `sex`, `minimum_age`, `maximum_age`, `healthy_volunteers` |

## 3. Pipeline architecture

```
CSV (raw) ──► pandas DataFrame ──► cleaning ──► NLP preprocessing ──► features ──► EDA
                                     │              │                    │
                         missing values,   normalise → clean →     TF-IDF (1-2 grams)
                         duplicates,       tokenize → stop words →  + text statistics
                         types, labels     lemmatize
                                                                         │
                                       6 models × RandomizedSearchCV (3-fold, macro-F1)
                                                                         │
                         best model ──► evaluation (Acc, P, R, F1, AUC, CM, robustness)
                                                                         │
             models/disease_classifier.joblib = TextPreprocessor → TF-IDF → classifier
                                                                         │
                                          Streamlit app (Predict · Dashboard · Insights · Model)
```

One command (`python run_pipeline.py`) runs every step. The saved model is a single scikit-learn `Pipeline` that takes **raw text**, so the training-time and app-time preprocessing can never drift apart. This was verified: the raw-text pipeline and the pre-cleaned-text model gave identical predictions on 300 test summaries (agreement 1.000).

## 4. Data preprocessing (`src/preprocessing.py`)
1. Load the CSV into a DataFrame and profile it (dtype, missing values and unique values per column → `reports/raw_data_profile.csv`).
2. Strip whitespace and treat empty strings as missing.
3. **Duplicates:**
   * none on full rows or `nct_id`
   * 30 rows of identical summaries with *different* labels dropped (ambiguous)
   * 262 repeated summaries reduced to one copy, which prevents train/test overlap
4. **Missing values:**
   * `phase` → *Not Applicable* (37,001)
   * `official_title` ← `title`
   * `interventions` / `eligibility_criteria` → *Not specified*
   * `sex` → mode
   * `healthy_volunteers` → *Unknown*
   * ages parsed to years; a missing maximum age means *no upper limit* and is flagged, not imputed
5. **Label encoding:** `disease_category` (readable) + `label_id` (0–7).

## 5. Text preprocessing & NLP operations (`src/text_processing.py`)
| Step | Implementation |
|---|---|
| Normalisation | unicode → ASCII, lowercase, URLs and e-mails removed, *COVID 19 / SARS-CoV-2 / 2019-nCoV* → `covid-19` |
| Cleaning | punctuation, special characters and stand-alone numbers removed |
| Tokenization | regex tokenizer that keeps medical codes and compounds (`hba1c`, `her2`, `covid-19`, `vaso-occlusive`) |
| Stop-word removal | scikit-learn English list (318 words) + 71 additional clinical boiler-plate words (389 in total) (*study, patient, trial, randomized, evaluate …*) |
| Lemmatization | NLTK WordNet lemmatizer, noun then verb (*studies → study, tumors → tumor, receiving → receive*) |
| Output | `clean_text`. Median length drops from 69 words to 34 informative tokens. |

21 summaries with fewer than 3 meaningful tokens after cleaning were removed, leaving **60,024 trials**.

## 6. Feature engineering (`src/features.py`)
* **TF-IDF vectorization:** unigrams + bigrams, `min_df = 3`, `max_df = 0.9`, sub-linear TF, up to 60,000 features. The n-gram range and `min_df` were tuned.
* **Text statistics:** word count, token count, unique tokens, lexical diversity, average word length.
* **Domain features:** `mentions_own_disease`, `n_disease_categories_mentioned` (keyword dictionaries per disease), plus `includes_children`, `includes_older_adults`, `n_conditions` and `n_interventions`.
* **Term analytics:** most frequent unigrams and bigrams, document frequency, and chi² distinctive terms per category.

## 7. Exploratory data analysis
Full report: [EDA_Report.md](EDA_Report.md). Highlights:
* The classes are imbalanced, 14.3 : 1 (Breast Cancer 27.1% … Sickle Cell Anemia 1.9%).
* Each disease has a distinct vocabulary: drugs (metformin, methotrexate, hydroxyurea), biomarkers (HbA1c, HER2, IOP) and procedures (mastectomy, trabeculectomy).
* Only 67.9% of Anxiety summaries name anxiety. It is often a secondary outcome in other studies.
* Design patterns: 62% of breast-cancer trials are female-only, 65% of sickle-cell trials accept children, 86% of anxiety trials have no drug phase (behavioural), and COVID-19 has the most observational studies and unknown statuses.

## 8. Modelling (`src/train.py`)
* Stratified 80/20 split: 48,019 train / 12,005 test.
* Six algorithms, each tuned with `RandomizedSearchCV` (3-fold stratified CV, scored on macro-F1, which suits imbalanced classes):

| Model | CV macro-F1 | Test accuracy | Test macro-F1 |
|---|---:|---:|---:|
| **Linear SVM** | **0.9472** | **0.9579** | **0.9544** |
| SGD (modified Huber) | 0.9460 | 0.9574 | 0.9529 |
| Logistic Regression | 0.9442 | 0.9563 | 0.9523 |
| Random Forest | 0.9363 | 0.9457 | 0.9414 |
| Complement Naive Bayes | 0.9329 | 0.9449 | 0.9421 |
| Multinomial Naive Bayes | 0.9065 | 0.9232 | 0.9185 |

* The best model by CV score is **Linear SVM** (C = 0.6, unigrams + bigrams, `min_df` = 3). It is wrapped in `CalibratedClassifierCV` (sigmoid) so the app can show probabilities. Final test result: **accuracy 0.9586, macro-F1 0.9549**.

## 9. Evaluation
Full report: [Model_Evaluation_Report.md](Model_Evaluation_Report.md).
* Per-class F1 ranges from 0.936 (COPD) to 0.975 (Glaucoma). Every class is ≥ 0.93.
* The main confusions involve **Anxiety** (for example COVID-19 → Anxiety, 46 cases). Error analysis shows three causes: trials that genuinely study both conditions, label noise (summaries about mental health retrieved under another condition's search), and healthy-volunteer pharmacokinetic studies with no disease vocabulary.
* **Confidence is reliable:** 82.7% of test predictions have ≥ 0.9 probability and are 99.6% accurate. Below 0.5 probability, accuracy drops to 44%.
* **Robustness / leakage check:** after removing all disease-name keywords from train and test text and retraining, accuracy is still **90.5%**. The model uses drugs, biomarkers and clinical context, not only the disease name. No metadata columns (`conditions`, `title`) are used as inputs, so the label can't leak in through them.

## 10. Streamlit application (`app.py`)
| Page | Features |
|---|---|
| **Predict** | text input + test-set examples · predicted category, confidence, runner-up · probability chart for all 8 classes · words driving the prediction · every NLP step shown · low-confidence and too-short warnings · batch CSV prediction with download |
| **Dashboard** | filters (category, study type, phase, status) · KPIs · category, phase, status, length and sex charts (Plotly) · live term frequencies · trial search with ClinicalTrials.gov links |
| **Insights** | findings computed live: category distribution, frequent medical terms, text patterns, design patterns, model performance and prediction accuracy · recommendations |
| **Model Performance** | metrics · model comparison · confusion matrix · per-class report · robustness test · confidence vs accuracy · top terms · misclassified examples |

## 11. Insights & recommendations
1. Auto-tag new trial registrations and send only low-confidence predictions (< 0.5) to human curators. About 83% of trials can be tagged at ≥ 0.9 confidence with 99.6% accuracy.
2. Model anxiety as a **cross-cutting** condition. A multi-label scheme would represent trials that study several diseases better.
3. Use the term profiles for research-portfolio analysis, for example vaccine vs treatment trials in COVID-19, or biologic vs conventional DMARDs in RA.
4. Sickle Cell Anemia and Glaucoma are under-represented, and sickle-cell research is mostly paediatric. These are areas where more trials would help.
5. Sickle-cell and breast-cancer trials are terminated or withdrawn most often. These patterns can feed into feasibility planning.

## 12. Limitations & future work
* The label is the *search query* each trial was retrieved with. Trials about several diseases therefore carry one label, which adds some label noise.
* Bag-of-words TF-IDF ignores word order and negation. Transformer models (BioBERT, ClinicalBERT) and multi-label classification are natural next steps.
* `eligibility_criteria`, `conditions` and `interventions` could be added as extra text fields for a richer model.

## 13. Reproducibility
`pip install -r requirements-dev.txt` → `python run_pipeline.py` → `streamlit run app.py`. Library versions are pinned in `requirements.txt`, `random_state = 42` is used throughout, and the executed notebook `notebooks/Clinical_Trial_NLP_Classification.ipynb` walks through every step.
