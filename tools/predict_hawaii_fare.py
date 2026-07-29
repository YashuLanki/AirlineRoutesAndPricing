#!/usr/bin/env python3
"""Tool: predict a Honolulu <-> Majuro round-trip fare.

See workflows/predict_hawaii_fare.md for the operating procedure this implements.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.hawaii_case_study import MODEL_PATH, predict_hawaii_fare, train_hawaii_model


def _parse_date(value):
    return datetime.strptime(value, "%d/%m/%Y").date()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--departure-date", required=True, help='"DD/MM/YYYY"')
    parser.add_argument("--return-date", required=True, help='"DD/MM/YYYY"')
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print("No model found yet, training on the 38 real fares first...")
        r2 = train_hawaii_model()
        print(f"Trained. Test R^2: {r2:.4f}")

    price = predict_hawaii_fare(_parse_date(args.departure_date), _parse_date(args.return_date))
    print(f"Predicted round-trip fare: ${price:,.2f}")


if __name__ == "__main__":
    main()
