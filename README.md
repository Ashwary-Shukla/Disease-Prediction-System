# AI Disease Prediction System

Educational Flask app: pick symptoms, get the top-3 likely diseases (RandomForest, 131 symptoms).
**Not a medical diagnosis.** The model is trained on a small, synthetic dataset.

Live demo: https://disease-prediction-system-ka8j.vercel.app/

## Run locally
    pip install -r requirements.txt
    python app.py            # set FLASK_DEBUG=1 for debug mode

## Configuration
- `SECRET_KEY` (env var): required in production so sessions are signed with a private key.

## Model files
`disease_model.pkl` and `symptoms.pkl` were saved with scikit-learn 1.6.1 / NumPy 2.x, so those versions are pinned.
Pickles can run code when loaded, so `app.py` only loads files matching the SHA-256 hashes in `TRUSTED_HASHES`.
After retraining, update the hashes with `sha256sum disease_model.pkl symptoms.pkl`.
