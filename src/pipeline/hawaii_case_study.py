"""Personal case study: Honolulu (HNL) <-> Majuro (MAJ) round-trip fares.

Unlike the main pipeline (src/components/*.py), which trains on the
thousands-of-rows Kaggle dataset of Indian domestic routes, this is a small,
honestly-scoped side project: United is the *only* carrier on this route (part
of its twice-weekly "Island Hopper" service), so there's no Airline/Source/
Destination/Stops variety to model - the real driver of price here is how far
in advance you book and which days of the week you fly.

The dataset (notebook/data/hawaii_marshall_islands.csv) is 38 REAL fares
retrieved from Google Flights on 2026-07-29 - not scraped in bulk (this route's
own booking site/OTAs block that), but read directly off Google Flights' date-grid
view across two booking windows (~1 week out and ~14-16 weeks out) in one
session. With n=38 and only two real booking-window clusters in the data, this
is a small, illustrative regression, not a production-grade model - see the
"Personal case study" section of the README for the full caveat.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.utils import load_object, save_object

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "notebook" / "data" / "hawaii_marshall_islands.csv"
MODEL_PATH = REPO_ROOT / "artifacts" / "hawaii_model.pkl"

FEATURE_COLUMNS = ["Days_Until_Departure", "Departure_DOW", "Return_DOW", "Trip_Length_Nights"]


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    departure = pd.to_datetime(df["Departure_Date"], format="%d/%m/%Y")
    ret = pd.to_datetime(df["Return_Date"], format="%d/%m/%Y")
    retrieved = pd.to_datetime(df["Retrieved_On"], format="%d/%m/%Y")

    df["Departure_DOW"] = departure.dt.dayofweek
    df["Return_DOW"] = ret.dt.dayofweek
    df["Trip_Length_Nights"] = (ret - departure).dt.days
    df["Days_Until_Departure"] = (departure - retrieved).dt.days
    return df


def train_hawaii_model():
    """Fit a small Linear Regression on the real HNL<->MAJ fares and save it."""
    try:
        df = _engineer(pd.read_csv(DATA_PATH))
        X, y = df[FEATURE_COLUMNS], df["Price"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        r2 = r2_score(y_test, model.predict(X_test))

        save_object(file_path=str(MODEL_PATH), obj=model)
        return r2
    except Exception as e:
        raise CustomException(e, sys)


def predict_hawaii_fare(departure_date: date, return_date: date, today: date = None) -> float:
    """Predict a HNL<->MAJ round-trip fare (USD) for a given departure/return date."""
    try:
        today = today or date.today()
        model = load_object(str(MODEL_PATH))
        row = pd.DataFrame([{
            "Days_Until_Departure": (departure_date - today).days,
            "Departure_DOW": departure_date.weekday(),
            "Return_DOW": return_date.weekday(),
            "Trip_Length_Nights": (return_date - departure_date).days,
        }])
        return float(model.predict(row[FEATURE_COLUMNS])[0])
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    print(f"Test R^2: {train_hawaii_model():.4f}")
