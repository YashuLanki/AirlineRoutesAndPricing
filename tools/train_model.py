#!/usr/bin/env python3
"""Tool: run the full training pipeline (ingest -> transform -> train).

See workflows/train_model.md for the operating procedure this implements.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.train_pipeline import run_training_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to a raw CSV in the Airline.csv schema (defaults to notebook/data/Airline.csv)",
    )
    args = parser.parse_args()

    r2 = run_training_pipeline(data_path=args.data_path)
    print(f"Training complete. Test R^2: {r2:.4f}")


if __name__ == "__main__":
    main()
