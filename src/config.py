"""Central project configuration.

All paths, column names and global constants live here so that the
notebooks and the Streamlit app share exactly the same definitions.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Reproducibility and evaluation settings
# --------------------------------------------------------------------------
RANDOM_STATE = 42  # Used for every step that involves randomness
TEST_SIZE = 0.2  # Share of the data held out for the final evaluation
CV_FOLDS = 5  # Number of stratified cross-validation folds

# --------------------------------------------------------------------------
# Paths (resolved relative to the project root, independent of the cwd)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "data.csv"
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "clean.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "personality_pipeline.joblib"

# --------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------
# The 19 questionnaire items (answers from 1 = does not apply to 5 = applies)
QUESTION_COLUMNS = [
    "N1", "N2", "N3", "N4", "N5", "N6", "N7", "N8", "N9", "N10",
    "E1", "E3", "E4", "E5", "E7", "E9", "E10",
    "A4", "C4",
]

NUMERIC_COLUMNS = QUESTION_COLUMNS + ["age"]
CATEGORICAL_COLUMNS = ["gender", "hand"]

# Feature order used for training and, later, for the app input. The model
# input must always contain exactly these columns in exactly this order.
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "target"

# --------------------------------------------------------------------------
# App input definitions (must match the categories seen during training)
# --------------------------------------------------------------------------
MIN_AGE = 13  # Youngest age kept by the data cleaning
MAX_AGE = 100  # Oldest age kept by the data cleaning
GENDER_OPTIONS = ["Female", "Male", "Other"]
HAND_OPTIONS = ["Right", "Left", "Both"]
