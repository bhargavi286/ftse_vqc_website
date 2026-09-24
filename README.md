# FTSE VQC — Quantum-Assisted Market Prediction

A Flask web app that takes two market indicators (previous close price, trading
volume) and returns a prediction from a Variational Quantum Classifier (VQC)
pipeline, along with a visual chart of the result.

## ✨ What's new in this version
- Full visual redesign: landing page, feature explainer, and a two-panel
  predict dashboard (inputs + live results).
- Input validation with friendly inline error messages (no more server crashes
  on bad input).
- Chart re-styled to match the app's theme, and rendered with a headless-safe
  matplotlib backend (`Agg`) so it won't break on servers like Render.
- Form values and results persist after submission so the page never feels
  like it "reset."
- Fully responsive layout (mobile + desktop).

## Project structure
```
.
├── app.py                 # Flask app + prediction/chart logic
├── save_model.py           # Script used to (re)generate vqc_model.pkl
├── vqc_model.pkl            # Trained model (currently a placeholder — see below)
├── requirements.txt
├── static/
│   └── style.css            # Shared design system for both pages
└── templates/
    ├── index.html            # Landing page
    └── predict.html          # Prediction dashboard
```

## Running locally
```bash
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:5000

## Deploying (Render)
No changes needed to your existing Render setup — it still reads `PORT` from
the environment and binds to `0.0.0.0`, so the current build command
(`pip install -r requirements.txt`) and start command (`python app.py`, or
`gunicorn app:app`) both continue to work.

## ⚠️ Important: the model is currently a placeholder
`save_model.py` trains a dummy `LinearRegression` model on 5 hand-made data
points — it is **not** a real trained Variational Quantum Classifier yet. The
app is fully wired to swap in a real model: once you have a trained VQC (or
any `sklearn`-compatible estimator) saved as `vqc_model.pkl` with a
`.predict([[feature1, feature2]])` interface, it will work with zero code
changes.

## Customizing feature labels
The predict page currently labels the two inputs as "Previous close price"
and "Trading volume" as a sensible default for a FTSE-prediction tool. If your
real model uses different features, update the `<label>` and `.help` text in
`templates/predict.html`.

LIVE DEMO :  https://ftse-vqc-website-1.onrender.com


