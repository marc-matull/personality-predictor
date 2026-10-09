"""Streamlit app: predict a personality type from a short questionnaire.

Start it from the project root with::

    streamlit run app.py

The app only loads the pipeline that was trained and saved by
``notebooks/02_modeling.ipynb``. It neither trains a model nor stores any
user data.
"""

from typing import Dict, Optional, Union

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.ocean import get_banner_html, get_footer_html, get_ocean_css
from src.personality_types import PERSONALITY_TYPES
from src.questions import QUESTIONS, SCALE_HINT, SCALE_VALUES

Answers = Dict[str, Union[int, str]]


@st.cache_resource(show_spinner="Loading model ...")
def load_pipeline():
    """Load the trained pipeline from disk (cached across reruns).

    Returns:
        The fitted scikit-learn pipeline (preprocessing + model).
    """
    return joblib.load(config.MODEL_PATH)


def apply_ocean_theme() -> None:
    """Inject the ocean styles and show the animated banner."""
    st.markdown(f"<style>{get_ocean_css()}</style>", unsafe_allow_html=True)
    st.markdown(get_banner_html(), unsafe_allow_html=True)


def render_questionnaire() -> Optional[Answers]:
    """Show the questionnaire form and collect the answers.

    Returns:
        A dictionary with one entry per feature column once the form was
        submitted, otherwise ``None``.
    """
    with st.form("questionnaire"):
        st.subheader("About you")
        age_col, gender_col, hand_col = st.columns(3)
        age = age_col.number_input(
            "Age", min_value=config.MIN_AGE, max_value=config.MAX_AGE,
            value=25, step=1,
        )
        gender = gender_col.selectbox("Gender", config.GENDER_OPTIONS)
        hand = hand_col.selectbox("Writing hand", config.HAND_OPTIONS)

        st.subheader("Questionnaire")
        st.caption(f"How well does each statement describe you? "
                   f"{SCALE_HINT}.")
        answers: Answers = {}
        for column in config.QUESTION_COLUMNS:
            answers[column] = st.radio(
                QUESTIONS[column],
                options=SCALE_VALUES,
                index=2,  # neutral default
                horizontal=True,
                key=f"question_{column}",
            )

        submitted = st.form_submit_button(
            "\U0001F30A Dive in: predict my type"
        )

    if not submitted:
        return None
    answers.update({"age": int(age), "gender": gender, "hand": hand})
    return answers


def build_input_frame(answers: Answers) -> pd.DataFrame:
    """Turn the answers into the one-row DataFrame the pipeline expects.

    The columns must have exactly the names and the order that were used
    for training (``config.FEATURE_COLUMNS``).

    Args:
        answers: Answers of the user, one entry per feature column.

    Returns:
        A DataFrame with a single row.
    """
    return pd.DataFrame([answers])[config.FEATURE_COLUMNS]


def show_result(pipeline, input_frame: pd.DataFrame) -> None:
    """Predict the personality type and display the result.

    Args:
        pipeline: The fitted pipeline.
        input_frame: One-row DataFrame created by ``build_input_frame``.
    """
    prediction = pipeline.predict(input_frame)[0]
    probabilities = pd.Series(
        pipeline.predict_proba(input_frame)[0],
        index=pipeline.classes_,
    ).sort_values(ascending=False)

    info = PERSONALITY_TYPES[prediction]
    st.header(f"{info['creature']} Your personality type: {prediction}")
    st.write(f"Your sea creature: **{info['creature_name']}**. "
             f"{info['description']}")
    st.metric("Model confidence", f"{probabilities[prediction]:.0%}")

    st.subheader("Probability of each type")
    st.bar_chart(probabilities)


def main() -> None:
    """Render the app."""
    # Must be the first Streamlit command
    st.set_page_config(page_title="Personality Type Predictor",
                       page_icon="\U0001F30A")
    apply_ocean_theme()
    st.title("Personality Ocean")
    st.write(
        "Dive into a short questionnaire and a machine learning model "
        "predicts your personality type."
    )

    if not config.MODEL_PATH.exists():
        st.error(
            "The model file was not found. Please run the notebooks "
            "`01_eda.ipynb` and `02_modeling.ipynb` first (see README)."
        )
        st.stop()

    pipeline = load_pipeline()
    answers = render_questionnaire()
    if answers is not None:
        show_result(pipeline, build_input_frame(answers))

    st.markdown(get_footer_html(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
