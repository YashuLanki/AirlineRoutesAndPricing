# Honolulu → Majuro Fare Predictor

## Why this project exists

I'm Marshallese, born and raised in Hawaii. Going home to visit family in the
Marshall Islands should be simple — it's a 5.5 hour flight — but United is the
*only* airline on that route (part of its twice-weekly "Island Hopper"
service), so there's no competition keeping prices down. Fares swing wildly
depending on how far ahead you book, and I wanted to actually understand the
pattern instead of just guessing when to buy.

So I built a small, honest fare predictor for this one route: real data,
a model sized to match how much data I actually had, and a simple web app to
ask "what will this trip cost?" for any pair of dates.

## What it does

Give it a departure and return date, and it predicts the round-trip fare in
USD, using patterns learned from real historical quotes for this exact route.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or from the command line:

```bash
python tools/predict_hawaii_fare.py --departure-date 15/11/2026 --return-date 22/11/2026
```

## The data

Because United has a monopoly on this route, there's no Kaggle dataset or
public API for it — I had to go get real quotes myself. `notebook/data/hawaii_marshall_islands.csv`
holds **38 real fares** I read directly off Google Flights' date-grid view on
2026-07-29, across two booking windows: about 1 week out and about 14-16 weeks
out from that date. That's a single point-in-time snapshot, not a scraped
historical dataset (this route's booking channels block bulk scraping, and
I'm not going to try to defeat that) — so I'm upfront in the code and here
about what it is and isn't.

**What that means for the model**: with n=38 and only two real booking-window
clusters, a complex model would just memorize noise. So instead of a Random
Forest, `src/pipeline/hawaii_case_study.py` fits a small, interpretable Linear
Regression on four engineered features — days until departure, departure
day-of-week, return day-of-week, and trip length in nights — which is the
right amount of model for the amount of real signal in the data.

**Sanity check**: predicting the exact dates of a real quote already in the
dataset (Aug 5 → Aug 12, 2026) comes back at $2,721 against an actual quoted
price of $2,716. Predictions for dates between or beyond the two observed
windows are extrapolations and should be treated with more skepticism than
that — the README isn't hiding that limitation, and the app surfaces it too
("How this works" in the UI).

Also genuinely interesting, and part of why I found this fun to dig into:
because Majuro is across the International Date Line from Honolulu, this
flight lands the *next calendar day* local time despite being under 6 hours
in the air.

## How it's built

```
notebook/data/hawaii_marshall_islands.csv   The 38 real fares, with a Retrieved_On column for traceability
src/pipeline/hawaii_case_study.py           Feature engineering + Linear Regression, train + predict
src/utils.py                                save_object/load_object (pickle helpers)
src/exception.py                            A CustomException that reports file/line on failure
tools/predict_hawaii_fare.py                CLI entry point
workflows/predict_hawaii_fare.md            The operating procedure the CLI tool follows
app.py                                      Streamlit UI over the same predict function the CLI uses
```

The CLI tool and the Streamlit app both call the exact same `predict_hawaii_fare()`
function — there's one prediction path, not two copies that could drift apart.

## Limitations, on purpose left visible

- n=38 from a single retrieval date. This is not a production forecasting
  model, and it doesn't pretend to be one.
- No holidays, fuel surcharges, schedule changes, or fare movements since
  2026-07-29 are accounted for.
- Predictions outside the two observed booking windows are extrapolations.

Being upfront about this is the point — a model this size making modest,
well-caveated claims is more trustworthy than a bigger model overstating what
38 rows can actually support.
