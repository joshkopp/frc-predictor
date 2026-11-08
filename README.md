# FRC 2026 (REBUILT) Match Predictor

Streamlit demo of a leakage-safe machine-learning model that predicts FRC match winners from
pre-match team history. Trained on official 2026 matches from The Blue Alliance.

- Winner accuracy on held-out (latest 20%) official matches: ~77%
- Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`
- `streamlit_app.py` is the router and themed nav bar; the two pages live in `app_pages/` (Match Predictor, Alliance Recommender).
- The Alliance Recommender needs a TBA read key: set `TBA_API_KEY` in Streamlit secrets, or paste one on the page.
- `deploy/` holds the exported model (native XGBoost JSON) and per-team ratings. No API keys, no raw data.
- `src/` is just the feature code the app needs. The full training pipeline (validation, leakage
  tests, model comparison) lives in the main project, `frc_ml`.

Data: The Blue Alliance (thebluealliance.com). Not affiliated with FIRST or TBA.
