# Workflow: Predict a Single Ticket Price

## Objective
Given the details of one flight, return a predicted ticket price using the
currently trained model.

## Prerequisite
`artifacts/model.pkl` and `artifacts/preprocessor.pkl` must exist — run the
[train_model](train_model.md) workflow first if they don't.

## Required inputs
Either `--json path/to/flight.json` with the fields below, or the equivalent
CLI flags. All fields except `additional_info` are required:

| Field | Example | Notes |
|---|---|---|
| airline | `"IndiGo"` | must match a value seen during training, or it's treated as unknown |
| source | `"Banglore"` | |
| destination | `"New Delhi"` | |
| date_of_journey | `"24/03/2019"` | `DD/MM/YYYY` |
| dep_time | `"22:20"` | 24h `HH:MM` |
| arrival_time | `"01:10"` | 24h `HH:MM` |
| duration | `"2h 50m"` | `"<n>h <n>m"`, `"<n>h"`, or `"<n>m"` |
| total_stops | `"non-stop"` | one of: `non-stop`, `1 stop`, `2 stops`, `3 stops`, `4 stops` |
| additional_info | `"No info"` | optional, defaults to `"No info"` |

## Tool to run
`tools/predict_fare.py`

```
python tools/predict_fare.py --airline IndiGo --source Banglore \
  --destination "New Delhi" --date-of-journey 24/03/2019 \
  --dep-time 22:20 --arrival-time 01:10 --duration "2h 50m" \
  --total-stops non-stop
```

Or with a JSON file:

```
python tools/predict_fare.py --json flight.json
```

## Expected output
Console line: `Predicted price: N,NNN.NN`

## Edge cases / known issues
- Unseen `Airline` / `Source` / `Destination` / `Additional_Info` values are
  handled gracefully: the fitted `OneHotEncoder` uses `handle_unknown="ignore"`,
  so an unseen category just contributes an all-zero encoding rather than
  crashing — expect a less accurate prediction in that case, not an error.
- `total_stops` must match one of the five literal strings above (see
  `STOP_MAP` in `src/utils.py`) — anything else is treated as missing and
  imputed with the most common value seen during training.
- If `artifacts/model.pkl` is missing, this fails immediately with a clear
  file-not-found error — run the training workflow first.
