#!/usr/bin/env python3
"""Tool: predict the ticket price for a single flight.

See workflows/predict_fare.md for the operating procedure this implements.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

REQUIRED_FIELDS = [
    "airline",
    "source",
    "destination",
    "date_of_journey",
    "dep_time",
    "arrival_time",
    "duration",
    "total_stops",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Path to a JSON file with flight details")
    parser.add_argument("--airline", help='e.g. "IndiGo"')
    parser.add_argument("--source", help='e.g. "Banglore"')
    parser.add_argument("--destination", help='e.g. "New Delhi"')
    parser.add_argument("--date-of-journey", dest="date_of_journey", help='"DD/MM/YYYY"')
    parser.add_argument("--dep-time", dest="dep_time", help='"HH:MM" (24h)')
    parser.add_argument("--arrival-time", dest="arrival_time", help='"HH:MM" (24h)')
    parser.add_argument("--duration", help='e.g. "2h 50m"')
    parser.add_argument("--total-stops", dest="total_stops", help='"non-stop", "1 stop", "2 stops", ...')
    parser.add_argument("--additional-info", dest="additional_info", default="No info")
    args = parser.parse_args()

    if args.json:
        with open(args.json) as f:
            payload = json.load(f)
    else:
        payload = {field: getattr(args, field) for field in REQUIRED_FIELDS}
        payload["additional_info"] = args.additional_info
        missing = [field for field in REQUIRED_FIELDS if not payload.get(field)]
        if missing:
            parser.error(f"Missing required fields: {', '.join(missing)} (or pass --json)")

    data = CustomData(**payload)
    price = PredictPipeline().predict(data)
    print(f"Predicted price: {price:,.2f}")


if __name__ == "__main__":
    main()
