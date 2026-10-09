"""Descriptions of the four personality types predicted by the model."""

from typing import Dict

# Keys are exactly the class labels of the trained model
PERSONALITY_TYPES: Dict[str, Dict[str, str]] = {
    "Moderate": {
        "description": (
            "Balanced: none of the personality traits is extreme."
        ),
    },
    "Resilient": {
        "description": (
            "Emotionally stable and calm under pressure."
        ),
    },
    "Overcontroller": {
        "description": (
            "Anxious and introverted, tends to hold back."
        ),
    },
    "Undercontroller": {
        "description": (
            "Impulsive and spontaneous, cares less about rules."
        ),
    },
}
