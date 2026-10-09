"""Question texts shown in the Streamlit app.

The 19 items are a subset of the Big Five (IPIP) personality questionnaire.
Each column name of the dataset is mapped to the statement that the user
rates. Some statements are worded in the opposite direction on purpose
(e.g. ``N2`` or ``E4``); the model learns the meaning of every item itself.
"""

from typing import Dict

QUESTIONS: Dict[str, str] = {
    "N1": "I get stressed out easily.",
    "N2": "I am relaxed most of the time.",
    "N3": "I worry about things.",
    "N4": "I seldom feel blue.",
    "N5": "I am easily disturbed.",
    "N6": "I get upset easily.",
    "N7": "I change my mood a lot.",
    "N8": "I have frequent mood swings.",
    "N9": "I get irritated easily.",
    "N10": "I often feel blue.",
    "E1": "I am the life of the party.",
    "E3": "I feel comfortable around people.",
    "E4": "I keep in the background.",
    "E5": "I start conversations.",
    "E7": "I talk to a lot of different people at parties.",
    "E9": "I don't mind being the center of attention.",
    "E10": "I am quiet around strangers.",
    "A4": "I sympathize with others' feelings.",
    "C4": "I make a mess of things.",
}

# Answer scale shown to the user
SCALE_VALUES = [1, 2, 3, 4, 5]
SCALE_HINT = "1 = does not apply, 5 = applies"
