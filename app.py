"""Honolulu <-> Majuro fare predictor.

Run with: streamlit run app.py
"""
from datetime import date

import streamlit as st

from src.pipeline.hawaii_case_study import MODEL_PATH, predict_hawaii_fare, train_hawaii_model

DEFAULT_DEPARTURE = date(2026, 11, 15)
DEFAULT_RETURN = date(2026, 11, 22)

st.set_page_config(page_title="Honolulu → Majuro Fare Predictor", page_icon="🌺")

st.title("🌺 Honolulu → Majuro Fare Predictor")
st.caption(
    "United is the only airline on this route, so the one thing that actually "
    "moves the price is how far ahead you book and which days you fly."
)

if not MODEL_PATH.exists():
    with st.spinner("Setting things up for the first time…"):
        train_hawaii_model()

col1, col2 = st.columns(2)
with col1:
    departure = st.date_input("Departure date", value=DEFAULT_DEPARTURE)
with col2:
    return_date = st.date_input("Return date", value=DEFAULT_RETURN)

if return_date <= departure:
    st.error("Return date must be after the departure date.")
else:
    nights = (return_date - departure).days
    st.caption(f"That's a {nights}-night trip.")

    try:
        price = predict_hawaii_fare(departure, return_date)
        st.metric("Predicted round-trip fare", f"${price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

with st.expander("How this works"):
    st.markdown(
        "Trained on **38 real fares** read from Google Flights on 2026-07-29 "
        "(a single point-in-time snapshot across two booking windows — about "
        "1 week out and 14-16 weeks out — not a historical fare-tracking "
        "dataset). With only 38 rows, treat this as a rough, illustrative "
        "estimate rather than a precise forecast, especially for dates far "
        "from those two windows. Full details in the project README.\n\n"
        "Fun fact: Majuro is across the International Date Line from "
        "Honolulu, so this ~5.5 hour nonstop flight lands the *next calendar "
        "day* local time."
    )
    if st.button("Retrain on the current data"):
        with st.spinner("Fitting…"):
            r2 = train_hawaii_model()
        st.success(f"Done — test R² = {r2:.4f}")
        st.rerun()
