# Workflow: Predict a Honolulu <-> Majuro Fare

## Objective
Predict the round-trip fare for United's Honolulu (HNL) <-> Majuro (MAJ) route.

## Why the model is this small
United is the only carrier on this route (part of its twice-weekly "Island
Hopper" service), so there's no Airline/Source/Destination/Stops variety to
learn from. The real driver of price here is how far in advance you book and
which days of the week you fly.

## Data source
`notebook/data/hawaii_marshall_islands.csv` — 38 **real** fares read from
Google Flights' date-grid view on 2026-07-29, across two booking windows
(~1 week out and ~14-16 weeks out from that date). This is a single point-in-time
snapshot, not a historical fare-tracking dataset, and n=38 is small — treat
predictions as an illustrative estimate, not a production-grade forecast.

## Required inputs
- `--departure-date "DD/MM/YYYY"`
- `--return-date "DD/MM/YYYY"`

## Tool to run
`tools/predict_hawaii_fare.py`

```
python tools/predict_hawaii_fare.py --departure-date 15/11/2026 --return-date 22/11/2026
```

If `artifacts/hawaii_model.pkl` doesn't exist yet, the tool trains it
automatically on the 38 real fares (a Linear Regression on: days until
departure at the time of asking, departure/return day-of-week, and trip
length in nights).

## Expected output
Console line: `Predicted round-trip fare: $N,NNN.NN`

## Edge cases / known issues
- Predictions for dates far outside the two windows actually observed
  (~1 week and ~14-16 weeks out) are extrapolations — the model has no data
  points in between or beyond, so treat those numbers with more skepticism.
- This model does not know about holidays, fuel surcharges, schedule changes,
  or fare increases since 2026-07-29 — it only knows the patterns present in
  the 38 real quotes it was trained on.
