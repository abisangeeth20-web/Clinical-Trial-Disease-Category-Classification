# Model Evaluation Report — Disease Category Classification

**Task:** multi-class text classification of clinical-trial `brief_summary` into 8 disease categories
**Data:** 60,024 cleaned trials, stratified 80/20 split → **48,019 train / 12,005 test** (`random_state = 42`)
**Features:** TF-IDF on lemmatized, stop-word-free text (unigrams + bigrams, sub-linear TF, 60,000 terms)
**Selection metric:** macro-F1 from 3-fold stratified CV. Macro averaging weights every disease equally, which matters with a 14 : 1 class imbalance.

Figures: [`figures/model_*.png`](figures/) · Tables: [`model_comparison.csv`](model_comparison.csv), [`classification_report.csv`](classification_report.csv), [`confusion_matrix.csv`](confusion_matrix.csv), [`top_confusions.csv`](top_confusions.csv), [`misclassified_examples.csv`](misclassified_examples.csv), [`confidence_vs_accuracy.csv`](confidence_vs_accuracy.csv)

---

## 1. Model comparison — `model_01_comparison.png`

Each model is a `TfidfVectorizer → classifier` pipeline. `RandomizedSearchCV` tuned the TF-IDF n-gram range and `min_df` together with each classifier's own hyper-parameters.

| Rank | Model | CV macro-F1 (± std) | Test Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | F1 (weighted) | ROC-AUC (OvR) | Tuning time |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **Linear SVM** | **0.9472 ± 0.0023** | **0.9579** | **0.9625** | 0.9471 | **0.9544** | 0.9579 | 0.9969 | 84 s |
| 2 | SGD (modified Huber) | 0.9460 ± 0.0010 | 0.9574 | 0.9547 | **0.9514** | 0.9529 | 0.9576 | 0.9959 | 69 s |
| 3 | Logistic Regression | 0.9442 ± 0.0023 | 0.9563 | 0.9561 | 0.9488 | 0.9523 | 0.9564 | **0.9977** | 162 s |
| 4 | Random Forest | 0.9363 ± 0.0026 | 0.9457 | 0.9585 | 0.9268 | 0.9414 | 0.9456 | 0.9957 | 662 s |
| 5 | Complement Naive Bayes | 0.9329 ± 0.0014 | 0.9449 | 0.9498 | 0.9360 | 0.9421 | 0.9451 | 0.9957 | 58 s |
| 6 | Multinomial Naive Bayes | 0.9065 ± 0.0019 | 0.9232 | 0.9274 | 0.9121 | 0.9185 | 0.9239 | 0.9933 | 67 s |

Best hyper-parameters:

| Model | Best hyper-parameters |
|---|---|
| Linear SVM | C = 0.6, no class weights, unigrams + bigrams, `min_df` = 3 |
| SGD | α = 1e-4, balanced class weights |
| Logistic Regression | C = 20, balanced class weights |
| Random Forest | 200 trees, unigrams, `min_df` = 5, balanced-subsample class weights |
| Complement NB | α = 0.1 |
| Multinomial NB | α = 0.03 |

**Observations**
* The three **linear models** (SVM, SGD, Logistic Regression) are within 0.2 F1 points of each other and clearly ahead. High-dimensional sparse TF-IDF text is close to linearly separable.
* **Bigrams helped** every model that could use them: *breast cancer*, *sickle cell*, *intraocular pressure*, *type diabetes*.
* **Complement NB** beats Multinomial NB by 2.4 F1 points. It is designed for imbalanced text.
* **Random Forest** is 8× slower and weaker on recall (0.927). Trees struggle with tens of thousands of sparse features.
* CV standard deviations are tiny (≤ 0.003), so the ranking is stable.

## 2. Final model

**Linear SVM** has the best CV macro-F1. It has no `predict_proba`, so it is wrapped in `CalibratedClassifierCV` (sigmoid, 3-fold) to give the class probabilities the app needs. The saved artefact is `models/disease_classifier.joblib`, a single pipeline: `TextPreprocessor → TF-IDF → calibrated Linear SVM`. It takes raw summary text.

| Metric (test set, n = 12,005) | Value |
|---|---:|
| **Accuracy** | **0.9586** |
| **Precision** (macro) | **0.9619** |
| **Recall** (macro) | **0.9485** |
| **F1-score** (macro) | **0.9549** |
| Precision / Recall / F1 (weighted) | 0.9589 / 0.9586 / 0.9586 |
| ROC-AUC (one-vs-rest, macro) | 0.9978 |
| Train accuracy / macro-F1 | 0.9941 / 0.9941 |

The 3.5-point train–test gap is normal for a linear text model with tens of thousands of features. The low CV variance and the strong hold-out score show it generalises well.
**End-to-end check:** the raw-text pipeline and the model run on pre-cleaned text agree on **100%** of 300 test summaries, so the app's preprocessing exactly matches training.

## 3. Per-class performance — `model_03_per_class_metrics.png`

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Breast Cancer | 0.969 | 0.976 | 0.972 | 3,251 |
| Type 2 Diabetes | 0.960 | 0.973 | 0.966 | 2,277 |
| COVID-19 | 0.974 | 0.953 | 0.963 | 2,020 |
| Anxiety | **0.924** | 0.954 | 0.939 | 1,851 |
| COPD | 0.940 | 0.932 | **0.936** | 1,227 |
| Rheumatoid Arthritis | 0.971 | 0.920 | 0.945 | 722 |
| Glaucoma | **0.986** | 0.965 | **0.975** | 430 |
| Sickle Cell Anemia | 0.972 | **0.916** | 0.943 | 227 |

* Every category reaches F1 ≥ 0.93, including the smallest (Sickle Cell Anemia, 227 test trials).
* **Glaucoma** is the easiest category. Its vocabulary (IOP, trabeculectomy, timolol) is highly specific.
* **Anxiety has the lowest precision.** Trials from other categories that measure anxiety as an outcome get pulled into it.
* The small classes (RA, Sickle Cell) have lower *recall* than precision. Some of their trials are absorbed by larger classes, such as RA → Type 2 Diabetes (22) and RA → Breast Cancer (12).

## 4. Confusion matrix & error analysis — `model_02_confusion_matrix.png`

Most frequent confusions (actual → predicted):

| Actual → Predicted | Test trials |
|---|---:|
| COVID-19 → Anxiety | 46 |
| COPD → Anxiety | 30 |
| Breast Cancer → Anxiety | 29 |
| Anxiety → Breast Cancer | 28 |
| Anxiety → COPD | 25 |
| Breast Cancer → Type 2 Diabetes | 23 |
| Rheumatoid Arthritis → Type 2 Diabetes | 22 |
| Type 2 Diabetes → Anxiety | 22 |

**5 of the top 8 confusions involve Anxiety.** Reading [`misclassified_examples.csv`](misclassified_examples.csv) shows three kinds of error:

1. **Genuinely multi-disease trials.** These include mindfulness for stress and anxiety in people with COPD, pre-operative anxiety before mastectomy, and physiotherapy that reduces pain *and anxiety* after breast surgery. Both labels are defensible, but the dataset allows only one.
2. **Label noise from the registry search.** Some trials labelled *COVID-19* or *COPD* have summaries entirely about other mental-health topics: adolescent CBT, parental presence at anaesthesia induction, schizophrenia, substance use. The search query matched a secondary listed condition, so the model's "Anxiety" prediction actually describes the text better than the label does.
3. **Generic study designs with no disease context.** Most *RA → Type 2 Diabetes* errors are pharmacokinetic or bioequivalence studies in healthy volunteers (meloxicam, fostamatinib in renal impairment, radiolabelled BMS-986142). The text contains no disease vocabulary, so the prediction is close to a guess.

A large share of the remaining errors is therefore **label ambiguity rather than model failure**, so the true accuracy is likely a little higher than the measured 95.9%.

## 5. Confidence vs accuracy — `model_06_confidence_accuracy.png`

| Predicted-class probability | Test trials | Share | Accuracy |
|---|---:|---:|---:|
| ≥ 0.9 | 9,923 | 82.7% | **99.6%** |
| 0.7 – 0.9 | 1,242 | 10.3% | 92.6% |
| 0.5 – 0.7 | 525 | 4.4% | 63.8% |
| < 0.5 | 315 | 2.6% | 44.1% |

The calibrated probabilities are meaningful. **83% of trials can be auto-classified with 99.6% accuracy**, and the 7% with probability below 0.7 can be routed to a human reviewer. The app warns the user when confidence is below 0.5.

## 6. Leakage & robustness check — `model_05_keyword_masking.png`

**Leakage controls:**
* Only `brief_summary` is used as input. `conditions` and `title`, which often restate the label, are *not* used.
* Duplicate summaries were removed before splitting, so no text is shared between train and test.
* TF-IDF is fitted inside the pipeline on the training folds only.

**Does the model just look for the disease name?** All disease-name keywords were removed from the text, for example *breast, cancer, diabetes, covid-19, anxiety, copd, pulmonary, arthritis, glaucoma, sickle, anemia, insulin, HbA1c, IOP* (see `config.DISEASE_KEYWORDS`):

| Setting | Accuracy | Macro-F1 |
|---|---:|---:|
| Original text | 0.9586 | 0.9549 |
| Original model, keywords removed from test text | 0.8416 | 0.8322 |
| **Retrained without keywords** | **0.9049** | **0.8964** |

Without any disease names, the retrained model still classifies **90.5%** of trials correctly. It learns the clinical context: drugs and devices (*metformin, methotrexate, adalimumab, hydroxyurea, timolol, inhaler*), procedures (*mastectomy, trabeculectomy, transfusion*), outcomes (*exacerbation, vaso-occlusive crisis, lymphedema*) and populations. Disease names are useful, legitimate signal, but they are not the only thing the model relies on.

## 7. What drives predictions — `model_04_top_terms_per_class.png`

Strongest model weights per class (the app also shows per-prediction contributions):

| Category | Top weighted terms |
|---|---|
| Breast Cancer | breast, cancer, tumor, chemotherapy, mastectomy, mammography, lymphedema, radiation |
| Type 2 Diabetes | diabetes, diabetic, t2dm, insulin, glucose, metformin, hba1c |
| COVID-19 | covid-19, pandemic, coronavirus, vaccine, infection, immunogenicity, ritonavir, mask |
| Anxiety | anxiety, kinesiophobia, ocd, disorder, mental, mood, anxiolytic, gad |
| COPD | copd, chronic, asthma, pulmonary, emphysema, lung, inhaler, aecopd |
| Rheumatoid Arthritis | ra, arthritis, rheumatoid, sjogren, autoimmune, adalimumab, methotrexate |
| Glaucoma | glaucoma, iop, eye, trabeculectomy, intraocular, timolol, xalatan |
| Sickle Cell Anemia | scd, sickle, hemoglobinopathy, hydroxyurea, transfusion, crisis |

The weights are clinically sensible and easy to interpret, which is one advantage of a linear TF-IDF model over a black box.

## 8. Conclusion
* The spec's evaluation requirements are met: **Accuracy 0.959, Precision 0.962, Recall 0.949, F1 0.955** (macro), with every class ≥ 0.93 F1.
* The model is fast (milliseconds per summary), small (10.6 MB compressed), interpretable and robust to the removal of disease names.
* **Next steps:**
  * multi-label targets for multi-disease trials
  * adding `conditions` and `eligibility_criteria` text
  * transformer embeddings (BioBERT / ClinicalBERT) to capture negation and word order
