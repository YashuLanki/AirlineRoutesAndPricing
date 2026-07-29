# Workflow: Train the Airline Fare Prediction Model

## Objective
Produce a trained regression model (and its fitted preprocessor) that predicts
airline ticket price from flight details, and report its test-set R^2.

## When to use this
Whenever the raw dataset changes, or the pipeline code under `src/` has been
modified, and a fresh `model.pkl` + `preprocessor.pkl` is needed.

## Required inputs
- A raw CSV in the `Airline.csv` schema (Airline, Date_of_Journey, Source,
  Destination, Route, Dep_Time, Arrival_Time, Duration, Total_Stops,
  Additional_Info, Price). Defaults to `notebook/data/Airline.csv` if you don't
  pass `--data-path`.

## Tool to run
`tools/train_model.py`

```
python tools/train_model.py [--data-path PATH]
```

## What it does under the hood
1. `src/components/data_ingestion.py` — reads the raw CSV, writes a raw copy plus
   an 80/20 train/test split to `artifacts/`.
2. `src/components/data_transformation.py` — calls `src/utils.py::engineer_features`
   (the same function the prediction tool uses, so training and inference can't
   drift apart), then fits a `ColumnTransformer` (median/most-frequent imputation +
   scaling + one-hot encoding) and saves it to `artifacts/preprocessor.pkl`.
3. `src/components/model_trainer.py` — grid-searches Random Forest and Decision
   Tree regressors, picks the higher-scoring one on the held-out test set, and
   saves it to `artifacts/model.pkl`.

## Expected output
- Console line: `Training complete. Test R^2: 0.NN`
- Files written: `artifacts/data.csv`, `artifacts/train.csv`, `artifacts/test.csv`,
  `artifacts/preprocessor.pkl`, `artifacts/model.pkl`.

## Edge cases / known issues
- If test-set R^2 comes back below 0.6, `ModelTrainer` raises
  `CustomException("No best model found")` — treat that as a signal the dataset
  or feature engineering changed in a way that needs investigation, not
  something to silently retry.
- Raw `Duration` values are sometimes just `"19h"` (no minutes) —
  `engineer_features` handles this, but if you plug in a new data source with a
  different duration format, it will need updating.
- `Arrival_Time` sometimes carries a trailing date (e.g. `"01:10 22 Mar"`) —
  `engineer_features` only reads the `HH:MM` portion and ignores the date.
- `artifacts/` and `logs/` are regenerated outputs — safe to delete and rerun
  this workflow.
