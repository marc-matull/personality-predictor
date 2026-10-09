"""Evaluation helpers for comparing models with ``f1_macro``.

The macro-averaged F1 score weights all personality types equally, which is
important because the classes are imbalanced.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config


def get_cv_splitter() -> StratifiedKFold:
    """Return the cross-validation splitter used for all comparisons.

    Using one shared, seeded splitter guarantees that all models are
    evaluated on exactly the same folds.

    Returns:
        A shuffled ``StratifiedKFold`` with ``config.CV_FOLDS`` folds and
        ``config.RANDOM_STATE`` as random state.
    """
    return StratifiedKFold(
        n_splits=config.CV_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )


def evaluate_model(model: Any, X: Any, y: Any) -> Dict[str, float]:
    """Evaluate an already fitted model on held-out data.

    Args:
        model: A fitted estimator or pipeline with a ``predict`` method.
        X: Raw input features (a DataFrame with the expected columns).
        y: True class labels.

    Returns:
        A dictionary with the ``f1_macro`` score and the ``accuracy``.
    """
    predictions = model.predict(X)
    return {
        "f1_macro": f1_score(y, predictions, average="macro"),
        "accuracy": accuracy_score(y, predictions),
    }


def cross_validate_model(model: Any, X: Any, y: Any) -> Dict[str, float]:
    """Cross-validate an unfitted model with stratified folds.

    Args:
        model: An unfitted estimator or pipeline.
        X: Raw input features (usually the training data).
        y: True class labels.

    Returns:
        A dictionary with the mean and standard deviation of the validation
        ``f1_macro``, the mean validation accuracy, the mean training
        ``f1_macro`` (to spot overfitting) and the mean fit time in seconds.
    """
    scores = cross_validate(
        model,
        X,
        y,
        cv=get_cv_splitter(),
        scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"},
        return_train_score=True,
        n_jobs=-1,
    )
    return {
        "cv_f1_macro": scores["test_f1_macro"].mean(),
        "cv_f1_macro_std": scores["test_f1_macro"].std(),
        "cv_accuracy": scores["test_accuracy"].mean(),
        "train_f1_macro": scores["train_f1_macro"].mean(),
        "fit_time_s": scores["fit_time"].mean(),
    }


def plot_confusion_matrix(
    model: Any,
    X: Any,
    y: Any,
    normalize: Optional[str] = None,
    title: str = "Confusion matrix",
    ax: Optional[plt.Axes] = None,
) -> ConfusionMatrixDisplay:
    """Plot the confusion matrix of a fitted model.

    Args:
        model: A fitted estimator or pipeline with a ``predict`` method.
        X: Raw input features.
        y: True class labels.
        normalize: ``None`` for absolute counts or ``"true"`` to normalize
            each row (true class) to sum to one.
        title: Title of the plot.
        ax: Optional matplotlib axes to draw on.

    Returns:
        The ``ConfusionMatrixDisplay`` object of the plot.
    """
    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X,
        y,
        normalize=normalize,
        cmap="Blues",
        values_format=".2f" if normalize else "d",
        xticks_rotation=20,
        ax=ax,
    )
    display.ax_.set_title(title)
    return display
