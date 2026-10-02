import hashlib
import os
import pickle
import secrets

import numpy as np
from flask import Flask, redirect, render_template, request, session

app = Flask(__name__)

# Set SECRET_KEY in your environment (e.g. Vercel project settings) for production.
# Without it a random key is used, so sessions (language choice) reset on restart.
app.secret_key = os.environ.get("SECRET_KEY") or secrets.token_hex(32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MIN_SYMPTOMS = 2

# Pickles can execute code when loaded, so only load files whose SHA-256 matches.
# If you retrain the model, update these (run: sha256sum disease_model.pkl symptoms.pkl).
TRUSTED_HASHES = {
    "disease_model.pkl": "a46017658a7de9dadeed5aa2de47c247ee55e17b4f9727dbb573adfcd3f8fd73",
    "symptoms.pkl": "a0d7925dce18ddc122ee3e4645a2f1c7d2b898d86aa21f2f475239a8d9100a26",
}


def load_pickle(name):
    with open(os.path.join(BASE_DIR, name), "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != TRUSTED_HASHES[name]:
        raise RuntimeError(f"{name} does not match its trusted checksum; refusing to load it.")
    return pickle.loads(raw)


model = load_pickle("disease_model.pkl")
RAW_SYMPTOMS = load_pickle("symptoms.pkl")  # order must match the model's features

LABEL_FIXES = {
    "scurring": "Scurfing",
    "dischromic _patches": "Dyschromic Patches",
    "cold_hands_and_feets": "Cold Hands and Feet",
    "swollen_extremeties": "Swollen Extremities",
}


def make_label(raw):
    key = raw.strip().lower()
    if key in LABEL_FIXES:
        return LABEL_FIXES[key]
    return " ".join(key.replace("_", " ").split()).title()


# Checkboxes send the feature index, so messy raw names never reach the browser.
DISPLAY_SYMPTOMS = sorted(
    ({"id": i, "label": make_label(s)} for i, s in enumerate(RAW_SYMPTOMS)),
    key=lambda s: s["label"],
)

LANGUAGES = {
    "english": {
        "code": "en",
        "title": "🩺 AI Disease Prediction System",
        "tagline": "Smart Healthcare Solution Powered by Machine Learning",
        "badge": "AI Healthcare Assistant",
        "select": "Select Your Symptoms",
        "search": "Search symptoms (English)…",
        "selected": "Selected",
        "button": "Predict Disease",
        "result": "Prediction Result",
        "possible": "Most likely",
        "others": "Other possibilities",
        "min_error": f"Please select at least {MIN_SYMPTOMS} symptoms.",
        "disclaimer": "For educational use only. This model is trained on a small, synthetic dataset and is not a medical diagnosis. Please consult a qualified doctor.",
    },
    "hindi": {
        "code": "hi",
        "title": "🩺 AI रोग पहचान प्रणाली",
        "tagline": "मशीन लर्निंग द्वारा संचालित स्मार्ट स्वास्थ्य समाधान",
        "badge": "AI स्वास्थ्य सहायक",
        "select": "अपने लक्षण चुनें",
        "search": "लक्षण खोजें (अंग्रेज़ी में)…",
        "selected": "चुने गए",
        "button": "रोग की जांच करें",
        "result": "जांच परिणाम",
        "possible": "सबसे संभावित",
        "others": "अन्य संभावनाएँ",
        "min_error": f"कृपया कम से कम {MIN_SYMPTOMS} लक्षण चुनें।",
        "disclaimer": "यह केवल शैक्षणिक उपयोग के लिए है। यह मॉडल एक छोटे, कृत्रिम (सिंथेटिक) डेटासेट पर प्रशिक्षित है और चिकित्सीय निदान नहीं है। कृपया किसी योग्य डॉक्टर से परामर्श करें।",
    },
}


def render(selected=(), results=None, error=False):
    lang = session.get("language", "english")
    if lang not in LANGUAGES:
        lang = "english"
    return render_template(
        "index.html",
        symptoms=DISPLAY_SYMPTOMS,
        text=LANGUAGES[lang],
        lang=lang,
        min_symptoms=MIN_SYMPTOMS,
        selected=set(selected),
        results=results,
        error=error,
    )


@app.route("/")
def home():
    return render()


@app.route("/language", methods=["POST"])
def change_language():
    lang = request.form.get("language", "english")
    session["language"] = lang if lang in LANGUAGES else "english"
    return redirect("/")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return redirect("/")

    n = len(RAW_SYMPTOMS)
    selected = {
        int(v) for v in request.form.getlist("symptoms")
        if v.isascii() and v.isdigit() and int(v) < n
    }
    if len(selected) < MIN_SYMPTOMS:
        return render(selected=selected, error=True), 400

    x = np.zeros((1, n))
    x[0, sorted(selected)] = 1
    proba = model.predict_proba(x)[0]
    top = np.argsort(proba)[::-1][:3]
    results = [
        {"disease": str(model.classes_[i]).strip(), "pct": round(float(proba[i]) * 100, 1)}
        for i in top if proba[i] > 0
    ]
    return render(selected=selected, results=results)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1")
