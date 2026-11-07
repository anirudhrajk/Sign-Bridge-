"""Symptoms a patient can tap, so nobody needs a sign for them.

Each item is (id, emoji, label, words in the sentence). "urgent" items reach the doctor as an
emergency, however mild the patient says they are.
"""

SYMPTOM_GROUPS = [
    ("Common", [
        ("fever", "🤒", "Fever", "fever"),
        ("cold", "🤧", "Cold", "a cold"),
        ("cough", "😮‍💨", "Cough", "a cough"),
        ("sore-throat", "🗣️", "Sore throat", "a sore throat"),
        ("headache", "🤕", "Headache", "a headache"),
        ("body-pain", "🧍", "Body pain", "body pain"),
        ("tired", "😴", "Tired or weak", "tiredness or weakness"),
        ("dizzy", "😵", "Dizzy", "dizziness"),
        ("infection", "🦠", "Infection", "an infection"),
    ]),
    ("Stomach", [
        ("stomachache", "😖", "Stomachache", "a stomachache"),
        ("nausea", "🤢", "Feel sick", "nausea"),
        ("vomiting", "🤮", "Vomiting", "vomiting"),
        ("diarrhoea", "🚽", "Loose motions", "diarrhoea"),
        ("constipation", "🚫", "Constipation", "constipation"),
        ("no-appetite", "🍽️", "Not eating", "no appetite"),
    ]),
    ("Skin and injury", [
        ("rash", "🔴", "Rash", "a rash"),
        ("itching", "🖐️", "Itching", "itching"),
        ("swelling", "🎈", "Swelling", "swelling"),
        ("wound", "🩹", "Cut or wound", "a cut or wound"),
        ("burn", "🔥", "Burn", "a burn"),
        ("injury", "🦴", "Injury or fall", "an injury or fall"),
    ]),
    ("Other", [
        ("eye", "👁️", "Eye problem", "an eye problem"),
        ("ear", "👂", "Ear problem", "an ear problem"),
        ("tooth", "🦷", "Toothache", "a toothache"),
        ("urine", "💧", "Urine problem", "a urine problem"),
        ("period", "🩸", "Period problem", "a period problem"),
        ("sleep", "🌙", "Cannot sleep", "trouble sleeping"),
        ("allergy", "🌸", "Allergy", "an allergy"),
        ("worried", "😟", "Worried or sad", "worry or low mood"),
    ]),
    ("Urgent", [
        ("breathing", "🫁", "Hard to breathe", "trouble breathing", "urgent"),
        ("chest-pain", "💔", "Chest pain", "chest pain", "urgent"),
        ("bleeding", "🩸", "Heavy bleeding", "heavy bleeding", "urgent"),
        ("fainted", "😵‍💫", "Fainted", "fainting", "urgent"),
        ("seizure", "⚡", "Fit or seizure", "a seizure", "urgent"),
        ("pregnant-pain", "🤰", "Pregnant and in pain", "pain during pregnancy", "urgent"),
    ]),
]


def symptom_groups():
    """[{"name": ..., "symptoms": [{"id", "emoji", "label", "say", "urgent"}]}]"""
    out = []
    for name, items in SYMPTOM_GROUPS:
        out.append({"name": name, "symptoms": [
            {"id": i[0], "emoji": i[1], "label": i[2], "say": i[3], "urgent": len(i) > 4}
            for i in items
        ]})
    return out
