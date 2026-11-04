"""
ISLRTC medical dictionary integration.
Includes a local seed plus a safe optional remote fetch contract.
"""
from flask import current_app

# Safe import: if 'requests' is missing, the app won't crash.
try:
    import requests
except ImportError:
    requests = None

LOCAL_MEDICAL_SIGNS = [
    ("Headache", "Symptoms", "Pain or pressure in the head."),
    ("Fever", "Symptoms", "Elevated body temperature."),
    ("Chest pain", "Symptoms", "Pain in the chest area."),
    ("Nausea", "Symptoms", "Feeling like vomiting."),
    ("Allergy", "Allergies", "An adverse reaction to a substance."),
    ("Water", "Needs", "Request for drinking water."),
    ("Help", "Emergency", "Request for assistance."),
    ("Medicine", "Medication", "Refers to medication."),
    ("Eye", "Body parts", "The eye."),
    ("Stomach", "Body parts", "The abdomen area."),
    ("Blood pressure", "Vitals", "Measurement of blood pressure."),
    ("X-ray", "Procedures", "Radiographic imaging."),
]


def fetch_remote_terms(query):
    """Optional: fetch from ISLRTC API if available."""
    if requests is None:
        # If requests failed to install, silently skip remote fetch
        return []
        
    base = current_app.config.get("ISLRTC_API_BASE")
    if not base: 
        return []
    try:
        r = requests.get(f"{base}/search", params={"q": query}, timeout=3)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception:
        return []


def get_local_seed():
    return [{"label": l, "category": c, "description": d} for l, c, d in LOCAL_MEDICAL_SIGNS]