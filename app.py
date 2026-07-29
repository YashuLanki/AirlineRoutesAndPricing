"""Streamlit demo for the airline fare predictor.

Run with: streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import run_training_pipeline
from src.pipeline.hawaii_case_study import (
    MODEL_PATH as HAWAII_MODEL_PATH,
    predict_hawaii_fare,
    train_hawaii_model,
)

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = REPO_ROOT / "artifacts" / "preprocessor.pkl"
METRICS_PATH = REPO_ROOT / "artifacts" / "metrics.json"

AIRLINES = [
    "Air Asia", "Air India", "GoAir", "IndiGo", "Jet Airways",
    "Jet Airways Business", "Multiple carriers",
    "Multiple carriers Premium economy", "SpiceJet", "Trujet", "Vistara",
    "Vistara Premium economy",
]
CITIES = ["Banglore", "Chennai", "Cochin", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "New Delhi"]
TOTAL_STOPS = ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"]
ADDITIONAL_INFO = [
    "No info", "In-flight meal not included", "No check-in baggage included",
    "1 Long layover", "1 Short layover", "2 Long layover", "Business class",
    "Change airports", "Red-eye flight",
]

st.set_page_config(page_title="Airline Fare Predictor", page_icon="✈️")


@st.cache_resource
def load_pipeline():
    return PredictPipeline()


def model_is_trained():
    return MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()


st.title("✈️ Airline Fare Predictor")

tab_india, tab_hawaii = st.tabs(["Indian domestic routes", "🌺 Honolulu → Majuro (personal case study)"])

with tab_india:
    st.caption("Predicts ticket price from flight details using a Random Forest / Decision Tree model, trained on the Kaggle Indian-domestic-flights dataset.")

    if not model_is_trained():
        st.warning("No trained model found yet.")
        if st.button("Train model now", type="primary"):
            with st.spinner("Running the training pipeline (ingest → transform → train)…"):
                r2 = run_training_pipeline()
            st.cache_resource.clear()
            st.success(f"Training complete — test R² = {r2:.4f}")
            st.rerun()
    else:
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text())
            st.caption(f"Current model: **{metrics['best_model']}** — test R² = {metrics['r2_score']:.4f}")

        with st.form("flight_form"):
            col1, col2 = st.columns(2)
            with col1:
                airline = st.selectbox("Airline", AIRLINES)
                source = st.selectbox("Source", CITIES, index=CITIES.index("Banglore"))
                destination = st.selectbox("Destination", CITIES, index=CITIES.index("New Delhi"))
                total_stops = st.selectbox("Total stops", TOTAL_STOPS)
            with col2:
                date_of_journey = st.date_input("Date of journey")
                dep_time = st.time_input("Departure time")
                arrival_time = st.time_input("Arrival time")
                additional_info = st.selectbox("Additional info", ADDITIONAL_INFO)

            duration_col1, duration_col2 = st.columns(2)
            with duration_col1:
                duration_hours = st.number_input("Duration — hours", min_value=0, max_value=48, value=2)
            with duration_col2:
                duration_mins = st.number_input("Duration — minutes", min_value=0, max_value=59, value=50)

            submitted = st.form_submit_button("Predict price", type="primary")

        if submitted:
            if source == destination:
                st.error("Source and destination can't be the same.")
            else:
                data = CustomData(
                    airline=airline,
                    source=source,
                    destination=destination,
                    date_of_journey=date_of_journey.strftime("%d/%m/%Y"),
                    dep_time=dep_time.strftime("%H:%M"),
                    arrival_time=arrival_time.strftime("%H:%M"),
                    duration=f"{duration_hours}h {duration_mins}m",
                    total_stops=total_stops,
                    additional_info=additional_info,
                )
                try:
                    price = load_pipeline().predict(data)
                    st.metric("Predicted price", f"₹{price:,.2f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with st.expander("Retrain model"):
            st.caption("Rerun the full training pipeline against the current dataset.")
            if st.button("Retrain now"):
                with st.spinner("Running the training pipeline (ingest → transform → train)…"):
                    r2 = run_training_pipeline()
                st.cache_resource.clear()
                st.success(f"Training complete — test R² = {r2:.4f}")
                st.rerun()

with tab_hawaii:
    st.caption(
        "United is the only carrier on Honolulu ↔ Majuro (part of its twice-weekly "
        "\"Island Hopper\" service), so there's no airline/route variety to model here — "
        "just how far ahead you book and which days you fly."
    )
    st.info(
        "Trained on **38 real fares** read from Google Flights on 2026-07-29 "
        "(not a historical dataset — a single point-in-time snapshot across two "
        "booking windows). Small sample size — treat this as an illustrative "
        "estimate, not a precise forecast. See the README for details.",
        icon="🌴",
    )

    if not HAWAII_MODEL_PATH.exists():
        st.warning("No trained model found yet.")
        if st.button("Train model now", type="primary", key="train_hawaii"):
            with st.spinner("Fitting a small Linear Regression on the 38 real fares…"):
                r2 = train_hawaii_model()
            st.success(f"Training complete — test R² = {r2:.4f}")
            st.rerun()
    else:
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            hawaii_departure = st.date_input("Departure date", key="hawaii_departure")
        with hcol2:
            hawaii_return = st.date_input("Return date", key="hawaii_return")

        if st.button("Predict fare", type="primary", key="predict_hawaii"):
            if hawaii_return <= hawaii_departure:
                st.error("Return date must be after the departure date.")
            else:
                try:
                    price = predict_hawaii_fare(hawaii_departure, hawaii_return)
                    st.metric("Predicted round-trip fare", f"${price:,.2f}")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        with st.expander("Retrain model"):
            st.caption("Refit the Linear Regression on notebook/data/hawaii_marshall_islands.csv.")
            if st.button("Retrain now", key="retrain_hawaii"):
                with st.spinner("Fitting…"):
                    r2 = train_hawaii_model()
                st.success(f"Training complete — test R² = {r2:.4f}")
                st.rerun()
