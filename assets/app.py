import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "CAR_PRICE_PREDICTOR.joblib"
CATEGORIES_PATH = BASE_DIR / "model_categories.json"
REFERENCE_YEAR = 2026
LINKEDIN_URL = "https://www.linkedin.com/in/ragul-y-s-a05253394/"
CONTACT_EMAIL = "ragsan188@gmail.com"

st.set_page_config(
    page_title="AutoValue | Car Price Estimator",
    page_icon="assets/av_mark.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      :root { color-scheme: dark; }
      .stApp { background: #0a0f1b; color: #edf2ff; }
      [data-testid="stHeader"] { background: rgba(10,15,27,0.94); }
      .block-container { max-width: 1220px; padding-top: 1.7rem; padding-bottom: 3.5rem; }
      .brand-row { display:flex; align-items:center; gap:13px; margin-bottom:1.2rem; }
      .brand-mark { width:44px; height:44px; border:1px solid #38506b; border-radius:13px;
                    display:flex; align-items:center; justify-content:center; color:#9ce8c9;
                    font-size:15px; font-weight:800; letter-spacing:-1px;
                    background:linear-gradient(145deg,#17253a,#101827); }
      .brand-name { color:#f2f5fb; font-size:1.15rem; font-weight:750; letter-spacing:-.025em; }
      .brand-sub { color:#8798b4; font-size:.76rem; margin-top:1px; letter-spacing:.06em; }
      .hero { padding:2rem 2.1rem; border:1px solid #26354b; border-radius:23px;
              background:radial-gradient(circle at 88% 0%,rgba(114,190,163,.10),transparent 35%),
                         linear-gradient(120deg,#131f32 0%,#101725 72%);
              margin-bottom:1.4rem; }
      .eyebrow { color:#9ce8c9; text-transform:uppercase; letter-spacing:.17em;
                 font-size:.72rem; font-weight:750; }
      .hero h1 { color:#f4f7ff; font-size:clamp(2.15rem,4.5vw,3.4rem); line-height:1.08;
                 letter-spacing:-.055em; margin:.65rem 0 .8rem; }
      .hero p { color:#b5c2d8; max-width:730px; line-height:1.7; margin-bottom:0; }
      .panel { border:1px solid #26354b; border-radius:18px; padding:1.35rem 1.4rem;
               background:#111a2a; }
      .price-label { color:#9aabc5; font-size:.88rem; }
      .price-value { color:#9ce8c9; font-size:clamp(2.2rem,5vw,3.5rem);
                     line-height:1.15; font-weight:780; letter-spacing:-.05em; margin:.35rem 0 .5rem; }
      .muted { color:#9baac1; font-size:.88rem; line-height:1.6; }
      .section-heading { color:#f0f4fb; letter-spacing:-.025em; }
      div[data-testid="stMetric"] { background:#111a2a; border:1px solid #26354b;
        border-radius:14px; padding:1rem; }
      div[data-testid="stMetricLabel"] { color:#a7b5cc; }
      .stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"] {
        background:#9ce8c9; color:#08131a; border:0; border-radius:11px;
        font-weight:750; min-height:3rem; transition:all .15s ease; }
      .stButton button[kind="primary"]:hover, .stFormSubmitButton button[kind="primary"]:hover {
        background:#b5f2d9; color:#08131a; }
      div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
        background:#0d1524; border-color:#34435b; border-radius:10px; }
      div[data-testid="stForm"] { border:1px solid #26354b; border-radius:18px; padding:1.25rem; background:#101827; }
      hr { border-color:#26354b; }
      .contact-card { border:1px solid #2a3a50; border-radius:18px; padding:1.5rem;
                      background:linear-gradient(120deg,#121d2e,#101725); }
      .contact-title { color:#f4f7ff; font-size:1.55rem; font-weight:750; letter-spacing:-.035em; margin-bottom:.4rem; }
      .contact-copy { color:#b5c2d8; line-height:1.7; margin-bottom:1rem; max-width:780px; }
      .contact-link { display:inline-block; color:#a3e9cc !important; text-decoration:none !important;
                      border:1px solid #395b58; border-radius:10px; padding:.65rem .9rem;
                      margin:.25rem .4rem .25rem 0; font-weight:650; }
      .contact-link:hover { background:#172c31; border-color:#6bbda0; }
      @media (max-width:640px) {
        .hero { padding:1.4rem; }
        .block-container { padding-left:1rem; padding-right:1rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the trained price model…")
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. Keep it beside app.py."
        )
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_categories():
    with CATEGORIES_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def euro(value):
    return f"€{value:,.0f}"


st.markdown(
    """
    <div class="brand-row">
      <div class="brand-mark">AV</div>
      <div>
        <div class="brand-name">AutoValue</div>
        <div class="brand-sub">USED CAR PRICE ESTIMATOR · GERMANY</div>
      </div>
    </div>
    <div class="hero">
      <div class="eyebrow">Data-driven valuation</div>
      <h1>Know what your car could be worth.</h1>
      <p>Estimate a used car's listing price using a machine-learning model trained on
      AutoScout24 Germany listings. Choose a make to see only models associated with that
      make in the supplied dataset. The result is an estimate, not a guaranteed sale price.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    model = load_model()
    categories = load_categories()
except Exception as exc:
    st.error(f"Could not load the model or its category file: {exc}")
    st.stop()

make_options = categories.get("make", [])
make_model = categories.get("make_model", {})
if not make_options or not make_model:
    st.error("The make-to-model mapping is missing. Please check model_categories.json.")
    st.stop()

default_make = "BMW" if "BMW" in make_options else make_options[0]
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown("### Vehicle details", help="The available choices are based on the supplied AutoScout24 dataset and model categories.")
    st.caption("Choose a make first. The model list updates to match it.")
    make = st.selectbox(
        "Make",
        make_options,
        index=make_options.index(default_make),
        key="vehicle_make",
    )
    model_options = make_model.get(make, [])
    if not model_options:
        st.warning("No supported models were found for this make in the training categories.")
        st.stop()

    # A make-specific key prevents a previous make's model from remaining selected
    # after the make changes.
    with st.form("vehicle_form"):
        c1, c2 = st.columns(2)
        with c1:
            year = st.selectbox(
                "Year of registration",
                list(range(REFERENCE_YEAR, 1989, -1)),
                index=REFERENCE_YEAR - 2018,
            )
            mileage = st.number_input(
                "Mileage (km)", min_value=0, max_value=1_000_000,
                value=80_000, step=5_000,
            )
            hp = st.number_input(
                "Power (HP)", min_value=1, max_value=2_000,
                value=150, step=10,
            )
        with c2:
            model_name = st.selectbox(
                "Model",
                model_options,
                index=0,
                key=f"vehicle_model_{make}",
            )
            fuel = st.selectbox(
                "Fuel type", categories["fuel"],
                index=categories["fuel"].index("Diesel") if "Diesel" in categories["fuel"] else 0,
            )
            gear = st.selectbox(
                "Transmission", categories["gear"],
                index=categories["gear"].index("Automatic") if "Automatic" in categories["gear"] else 0,
            )
            offer_type = st.selectbox(
                "Listing type", categories["offerType"],
                index=categories["offerType"].index("Used") if "Used" in categories["offerType"] else 0,
            )
        submitted = st.form_submit_button(
            "Estimate vehicle price", type="primary", use_container_width=True
        )

    st.caption(
        f"{len(model_options):,} model options available for {make}, based on observed "
        "make/model combinations that are also supported by the trained model."
    )

with right:
    st.markdown("### Estimated value")
    if submitted:
        age = REFERENCE_YEAR - int(year)
        row = pd.DataFrame([{
            "mileage": float(mileage),
            "make": make,
            "model": model_name,
            "fuel": fuel,
            "gear": gear,
            "offerType": offer_type,
            "hp": float(hp),
            "age": float(age),
        }])
        try:
            prediction = float(model.predict(row)[0])
            if not pd.notna(prediction):
                raise ValueError("The model returned a non-finite prediction.")
            st.markdown(
                f'<div class="panel"><div class="price-label">Predicted listing price</div>'
                f'<div class="price-value">{euro(max(0, prediction))}</div>'
                f'<p class="muted">Estimated for a {year} {make} {model_name} · '
                f'{mileage:,} km · {hp} HP</p></div>',
                unsafe_allow_html=True,
            )
            st.session_state["last_prediction"] = {
                "price": prediction, "year": year, "make": make, "model": model_name,
                "mileage": mileage, "hp": hp, "fuel": fuel, "gear": gear, "offerType": offer_type,
            }
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
    elif "last_prediction" in st.session_state:
        result = st.session_state["last_prediction"]
        st.markdown(
            f'<div class="panel"><div class="price-label">Last predicted listing price</div>'
            f'<div class="price-value">{euro(max(0, result["price"]))}</div>'
            f'<p class="muted">{result["year"]} {result["make"]} {result["model"]}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel"><div class="price-label">Your estimate will appear here</div>'
            '<div class="price-value">€ —</div><p class="muted">Complete the vehicle details '
            'and select “Estimate vehicle price”.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### Model performance")
    m1, m2 = st.columns(2)
    m1.metric("R² score", "0.908")
    m2.metric("Mean absolute error", "€1,841")
    st.caption(
        "Reported from the notebook's held-out test split. These metrics describe aggregate "
        "test performance, not the uncertainty of an individual prediction."
    )

st.divider()
with st.expander("How this estimate is produced"):
    st.write(
        "The saved scikit-learn pipeline applies StandardScaler to mileage, horsepower and vehicle age; "
        "OneHotEncoder to make, model, fuel, transmission and listing type; then predicts price with a "
        "100-tree Random Forest regressor."
    )
    st.write(
        "Notebook test results: MAE €1,840.95 · RMSE €5,554.53 · R² 0.90773. "
        "These results were recorded in the supplied notebook and have not been independently reproduced here."
    )
    st.warning(
        "The model does not return a calibrated prediction interval. Actual sale prices can differ due to "
        "condition, trim, service history, accident history, region and market changes—several of which are "
        "not inputs to this model."
    )

st.divider()
st.markdown(
    """
    <div class="contact-card">
      <div class="contact-title">Wanna connect?</div>
      <div class="contact-copy">
        Built with curiosity and a genuine interest in machine learning.
        Have feedback, ideas, or an opportunity to discuss? I'd be happy to connect
        and hear what you think.
      </div>
      <a class="contact-link" href="https://www.linkedin.com/in/ragul-y-s-a05253394/" target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>
      <a class="contact-link" href="mailto:ragsan188@gmail.com">ragsan188@gmail.com ↗</a>
    </div>
    <p class="muted" style="text-align:center;margin-top:1.2rem;">
      AutoValue · Built for demonstration and portfolio use · Not financial or appraisal advice
    </p>
    """,
    unsafe_allow_html=True,
)
