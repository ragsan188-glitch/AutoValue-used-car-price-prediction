# AutoValue — Used Car Price Estimator

A Streamlit web application that predicts an estimated used-car listing price from vehicle details using the supplied trained scikit-learn pipeline.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at the local URL printed by Streamlit.

## What is included

- `app.py` — Streamlit interface and prediction workflow
- `CAR_PRICE_PREDICTOR.joblib` — trained model pipeline
- `model_categories.json` — model-supported categories and dataset-derived make-to-model mapping
- `.streamlit/config.toml` — theme settings
- `assets/av_mark.png` — minimalist AV browser-tab icon

## Brand/model filtering

The make-to-model map was derived from the supplied AutoScout24 Germany CSV. Only make/model combinations present in the source dataset and supported by the saved model's categories are offered. This reduces invalid combinations in the UI but does not guarantee every selected vehicle configuration existed in the training data.

## Contact links

The website includes the supplied LinkedIn URL and `ragsan188@gmail.com`. The LinkedIn link points to the supplied personal profile URL.

## Notes

- The displayed performance metrics are those reported in the supplied notebook; they have not been independently re-evaluated in this app package.
- Estimates are not guaranteed sale prices. The model does not return calibrated individual prediction intervals.
- Pickle/joblib files should only be loaded from a trusted source.
