"""Central configuration: paths, column groups, label names and plot palette."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- raw data
RAW_CSV = ROOT / "clinical_trials_raw_patient2trial_conditions.csv"      # original file (158 MB, git-ignored)
RAW_DIR = ROOT / "data" / "raw"
RAW_PARTS = [RAW_DIR / f"clinical_trials_raw_part{i}.csv.gz" for i in (1, 2, 3)]  # same rows, < 25 MB each

# ---------------------------------------------------------------- outputs
PROCESSED_DIR = ROOT / "data" / "processed"
CLEANED_CSV = PROCESSED_DIR / "cleaned_clinical_trials.csv"               # git-ignored (large)
CLEANED_PARTS = [PROCESSED_DIR / f"cleaned_clinical_trials_part{i}.csv.gz" for i in (1, 2, 3)]  # < 25 MB each
PREPROCESSED_GZ = PROCESSED_DIR / "preprocessed_clinical_trials.csv.gz"   # NLP-ready text + features
DASHBOARD_GZ = PROCESSED_DIR / "dashboard_data.csv.gz"                    # light file used by the app

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "disease_classifier.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "eda_tables"
NLTK_DIR = ROOT / "nltk_data"

# ---------------------------------------------------------------- columns
TEXT_COL = "brief_summary"
LABEL_SOURCE_COL = "source_condition_query"
TARGET = "disease_category"
CLEAN_TEXT_COL = "clean_text"

LABEL_MAP = {
    "breast cancer": "Breast Cancer",
    "type 2 diabetes": "Type 2 Diabetes",
    "covid-19": "COVID-19",
    "anxiety": "Anxiety",
    "chronic obstructive pulmonary disease": "COPD",
    "rheumatoid arthritis": "Rheumatoid Arthritis",
    "glaucoma": "Glaucoma",
    "sickle cell anemia": "Sickle Cell Anemia",
}
CATEGORIES = list(LABEL_MAP.values())

# Disease-name words per category. Used (a) in EDA to measure how often a summary names its own
# disease and (b) for the keyword-masking robustness test of the model.
DISEASE_KEYWORDS = {
    "Breast Cancer": ["breast", "cancer", "carcinoma", "tumor", "tumour", "mammary", "her2", "oncology", "neoplasm"],
    "Type 2 Diabetes": ["diabetes", "diabetic", "t2d", "t2dm", "glycemic", "glycaemic", "insulin", "hba1c", "glucose"],
    "COVID-19": ["covid", "covid-19", "covid19", "sars-cov-2", "coronavirus", "sars", "cov"],
    "Anxiety": ["anxiety", "anxious", "anxiolytic"],
    "COPD": ["copd", "pulmonary", "obstructive", "chronic obstructive", "lung", "emphysema", "bronchitis"],
    "Rheumatoid Arthritis": ["rheumatoid", "arthritis", "ra"],
    "Glaucoma": ["glaucoma", "intraocular", "iop", "ocular", "eye"],
    "Sickle Cell Anemia": ["sickle", "scd", "anemia", "anaemia", "hemoglobin", "haemoglobin"],
}

DASHBOARD_COLS = [
    "nct_id", "title", TARGET, "overall_status", "study_type", "phase_label", "sex",
    "healthy_volunteers", "min_age_years", "max_age_years", "includes_children", "includes_older_adults",
    "summary_word_count", "clean_token_count", "n_conditions", "n_interventions",
    "mentions_own_disease", CLEAN_TEXT_COL, "clinicaltrials_url",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20

PALETTE = {
    "Breast Cancer": "#E45756", "Type 2 Diabetes": "#4C78A8", "COVID-19": "#F58518",
    "Anxiety": "#B279A2", "COPD": "#54A24B", "Rheumatoid Arthritis": "#9D755D",
    "Glaucoma": "#72B7B2", "Sickle Cell Anemia": "#EECA3B",
}
