import sys
from pathlib import Path

import pandas as pd

from src.exception import CustomException
from src.utils import engineer_features, load_object

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "artifacts" / "model.pkl"
PREPROCESSOR_PATH = REPO_ROOT / "artifacts" / "preprocessor.pkl"


class CustomData:
    """Holds the raw details of a single flight, in the same schema as Airline.csv."""

    def __init__(
        self,
        airline: str,
        source: str,
        destination: str,
        date_of_journey: str,
        dep_time: str,
        arrival_time: str,
        duration: str,
        total_stops: str,
        additional_info: str = "No info",
    ):
        self.airline = airline
        self.source = source
        self.destination = destination
        self.date_of_journey = date_of_journey
        self.dep_time = dep_time
        self.arrival_time = arrival_time
        self.duration = duration
        self.total_stops = total_stops
        self.additional_info = additional_info

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Airline": self.airline,
            "Source": self.source,
            "Destination": self.destination,
            "Date_of_Journey": self.date_of_journey,
            "Dep_Time": self.dep_time,
            "Arrival_Time": self.arrival_time,
            "Duration": self.duration,
            "Total_Stops": self.total_stops,
            "Additional_Info": self.additional_info,
        }])


class PredictPipeline:
    def __init__(self):
        self.model = load_object(MODEL_PATH)
        self.preprocessor = load_object(PREPROCESSOR_PATH)

    def predict(self, custom_data: CustomData) -> float:
        try:
            df = engineer_features(custom_data.to_dataframe())
            transformed = self.preprocessor.transform(df)
            prediction = self.model.predict(transformed)
            return float(prediction[0])
        except Exception as e:
            raise CustomException(e, sys)
