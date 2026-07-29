import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

STOP_MAP = {
    "non-stop": 0,
    "1 stop": 1,
    "2 stops": 2,
    "3 stops": 3,
    "4 stops": 4,
}


def _parse_duration(duration: str):
    """'2h 50m' / '19h' / '45m' -> (hours, minutes, total_minutes)."""
    hours, minutes = 0, 0
    for part in str(duration).split():
        if part.endswith("h"):
            hours = int(part[:-1])
        elif part.endswith("m"):
            minutes = int(part[:-1])
    return hours, minutes, hours * 60 + minutes


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw Airline.csv schema (Date_of_Journey, Dep_Time, Arrival_Time,
    Duration, Total_Stops as free-form strings) into the numeric/categorical
    columns the preprocessor expects.

    Used by both the training pipeline (data_transformation.py) and the
    prediction pipeline (predict_pipeline.py) so the two can never drift apart.
    """
    df = df.copy()

    journey = pd.to_datetime(df["Date_of_Journey"], format="%d/%m/%Y")
    df["Journey_Day"] = journey.dt.day
    df["Journey_Month"] = journey.dt.month
    df["Journey_Year"] = journey.dt.year

    for col, prefix in [("Dep_Time", "Dep_Time"), ("Arrival_Time", "Arrival_Time")]:
        # Arrival_Time sometimes carries a trailing "DD Mon", e.g. "01:10 22 Mar"
        time_only = df[col].astype(str).str.split(" ").str[0]
        parsed = pd.to_datetime(time_only, format="%H:%M")
        df[f"{prefix}_hour"] = parsed.dt.hour
        df[f"{prefix}_minute"] = parsed.dt.minute

    duration_parts = df["Duration"].apply(_parse_duration)
    df["Duration_hours"] = duration_parts.apply(lambda t: t[0])
    df["Duration_mins"] = duration_parts.apply(lambda t: t[1])
    df["Duration_total_mins"] = duration_parts.apply(lambda t: t[2])

    df["Total_Stops"] = df["Total_Stops"].map(STOP_MAP)
    df["Total_Stops"] = df["Total_Stops"].fillna(df["Total_Stops"].mode()[0])

    df = df.drop(
        columns=["Date_of_Journey", "Dep_Time", "Arrival_Time", "Duration", "Route"],
        errors="ignore",
    )
    return df


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)