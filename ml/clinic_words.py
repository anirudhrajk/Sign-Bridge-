"""Words a patient is likely to sign at a clinic, chosen by meaning from what each model knows.

The lists were picked without looking at any test score. Change them here, then run
  python ml/clinic_words.py
to write static/models/<lang>/clinic.json, which the sign page reads.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

WORDS = {
    "asl": {
        "Body": "head, face, eye, ear, nose, mouth, lips, tongue, tooth, cheek, chin, hair, arm, finger, feet",
        "How I feel": "hurt, sick, bad, better, fine, hot, thirsty, hungry, sleepy, awake, sad, mad, happy, cry, "
                      "fall, cut, wet, stuck, yucky, taste",
        "What I need": "water, drink, food, toilet, bath, shower, bed, sleep, nap, wait, home, go, stay, call on phone",
        "Talking": "yes, no, not, please, thank you, hello, bye, where, who, why, my, have, finish, now, later, "
                   "tomorrow, yesterday, morning, night, time, before, after, up, down, see, hear, listen, say",
        "People": "mom, dad, child, grandma, grandpa",
    },
    "isl": {
        "Health": "doctor, hospital, medicine, patient, sick, healthy, weak, strong, dead, death, deaf, blind, exercise",
        "How I feel": "bad, good, alright, hot, cold, warm, sad, happy, pleased, quiet, hard, soft",
        "Places and things": "bathroom, bedroom, bed, house, soap, phone, telephone, money, bill",
        "Talking": "hello, thank you, how are you, good morning, good afternoon, good evening, good night, "
                   "I, you, he, she, we, they, it",
        "People": "mother, father, son, daughter, husband, wife, sister, brother, child, baby, family, parent, "
                  "friend, man, woman",
        "Time": "today, tomorrow, yesterday, morning, afternoon, evening, night, hour, minute, week, month, year, "
                "time, monday, tuesday, wednesday, thursday, friday, saturday, sunday",
    },
}
BOOST = {"asl": 8, "isl": 8}   # a clinic word counts this many times more likely than another word


def words(lang):
    """{group: [word, ...]}"""
    return {group: [w.strip() for w in text.split(",")] for group, text in WORDS[lang].items()}


def main():
    for lang in WORDS:
        known = json.loads((ROOT / "static/models" / lang / "model.json").read_text(encoding="utf8"))["labels"]
        groups = words(lang)
        flat = [w for g in groups.values() for w in g]
        missing = [w for w in flat if w not in known]
        if missing or len(set(flat)) != len(flat):
            raise SystemExit(f"{lang}: not in the model {missing}, repeated "
                             f"{sorted({w for w in flat if flat.count(w) > 1})}")
        out = ROOT / "static/models" / lang / "clinic.json"
        out.write_text(json.dumps({"boost": BOOST[lang], "groups": groups}, ensure_ascii=False, indent=1), encoding="utf8")
        print(lang, len(flat), "clinic words of", len(known))


if __name__ == "__main__":
    main()
