"""Preprocessing pipeline for the personality type prediction project."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Create the (unfitted) preprocessing step of the model pipeline.

    The question scores and the age are standardized (zero mean, unit
    variance), which matters for linear models. Gender and writing hand are
    one-hot encoded. Categories that were not seen during training are
    encoded as all zeros instead of raising an error.

    Only scikit-learn classes are used (no custom transformers), so a
    pipeline containing this step can be loaded with ``joblib`` without
    importing any project code.

    Returns:
        A ``ColumnTransformer`` that expects a DataFrame with the columns
        defined in ``src.config.FEATURE_COLUMNS``.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLUMNS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="drop",
    )
