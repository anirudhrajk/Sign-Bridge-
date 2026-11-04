"""Readable English for dataset class names that are joined words."""

DISPLAY = {
    "biglarge": "big", "smalllittle": "small", "exmonsoon": "monsoon",
    "goodafternoon": "good afternoon", "goodevening": "good evening",
    "goodmorning": "good morning", "goodnight": "good night",
    "howareyou": "how are you", "thankyou": "thank you",
    "raceethnicity": "race", "storeorshop": "shop", "streetorroad": "road",
    "trainstation": "train station", "trainticket": "train ticket",
    "tshirt": "T-shirt", "youplural": "you (all)", "i": "I", "cellphone": "phone",
    # American Sign Language (Google ISLR dataset)
    "callonphone": "call on phone", "frenchfries": "french fries", "glasswindow": "window",
    "haveto": "have to", "hesheit": "he / she", "minemy": "my", "weus": "we", "icecream": "ice cream",
    "owie": "hurt", "potty": "toilet", "TV": "TV",
}


def display(name):
    return DISPLAY.get(name, name)
