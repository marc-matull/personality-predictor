"""Descriptions of the four personality types predicted by the model."""

from typing import Dict

# Keys are exactly the class labels of the trained model.
# Each type is paired with a sea creature for the ocean theme of the app.
PERSONALITY_TYPES: Dict[str, Dict[str, str]] = {
    "Moderate": {
        "description": (
            "Balanced: none of the personality traits is extreme."
        ),
        "creature": "\U0001F422",
        "creature_name": "Sea turtle",
    },
    "Resilient": {
        "description": (
            "Emotionally stable and calm under pressure."
        ),
        "creature": "\U0001F40B",
        "creature_name": "Whale",
    },
    "Overcontroller": {
        "description": (
            "Anxious and introverted, tends to hold back."
        ),
        "creature": "\U0001F41A",
        "creature_name": "Shell",
    },
    "Undercontroller": {
        "description": (
            "Impulsive and spontaneous, cares less about rules."
        ),
        "creature": "\U0001F42C",
        "creature_name": "Dolphin",
    },
}
