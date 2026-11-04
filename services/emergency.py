DEFAULT_EMERGENCY_PHRASES = [
    {"text": "I need help",       "category": "emergency"},
    {"text": "I am in pain",      "category": "emergency"},
    {"text": "I have an allergy", "category": "emergency"},
    {"text": "I need water",      "category": "emergency"},
    {"text": "I cannot breathe",  "category": "emergency"},
    {"text": "Call a doctor",     "category": "emergency"},
    {"text": "I feel dizzy",      "category": "emergency"},
    {"text": "I need medicine",   "category": "emergency"},
]


def get_default_phrases():
    return list(DEFAULT_EMERGENCY_PHRASES)