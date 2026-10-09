"""Streamlit app: predict a personality type from a short questionnaire.

Start it from the project root with::

    streamlit run app.py

The app only loads the pipeline that was trained and saved by
``notebooks/02_modeling.ipynb``. It neither trains a model nor stores any
user data.
"""

import joblib
import streamlit as st

from src import config


@st.cache_resource(show_spinner="Loading model ...")
def load_pipeline():
    """Load the trained pipeline from disk (cached across reruns).

    Returns:
        The fitted scikit-learn pipeline (preprocessing + model).
    """
    return joblib.load(config.MODEL_PATH)


def main() -> None:
    """Render the app."""
    # Must be the first Streamlit command
    st.set_page_config(page_title="Personality Type Predictor",
                       page_icon="\U0001F30A")
    st.title("Personality Type Predictor")
    st.write(
        "Answer a short questionnaire and a machine learning model "
        "predicts your personality type."
    )

    if not config.MODEL_PATH.exists():
        st.error(
            "The model file was not found. Please run the notebooks "
            "`01_eda.ipynb` and `02_modeling.ipynb` first (see README)."
        )
        st.stop()

    pipeline = load_pipeline()
    st.caption(
        "Model loaded. It distinguishes the types: "
        + ", ".join(pipeline.classes_)
    )


if __name__ == "__main__":
    main()
