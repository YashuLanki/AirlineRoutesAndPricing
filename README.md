# Predict Fare of Airline Tickets using ML: Project Overview

- **Objective**: Develop a machine learning model to predict airline ticket prices based on features like source, destination, airline, and number of stops.
  - **Dataset Features**: Includes flight details such as airline, date of journey, route, departure and arrival times, source, destination, total stops, additional info, and price.
  - **Approach**: Apply regression algorithms (e.g., Decision Trees, Random Forest) to predict ticket prices, with preprocessing like handling missing values and encoding categorical features.
  - **Model Evaluation**: Evaluate model performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
  - **Data Preprocessing**: Includes handling missing values, encoding categorical features, and scaling numerical data to improve model accuracy.
  - **Technologies Used**: Python, Pandas, Scikit-learn, Matplotlib, and other data science libraries for analysis and model building.

## Architecture

The project separates exploratory work from the production pipeline:

```
notebook/     Exploratory EDA and model-selection notebooks (the "how I got here")
src/          The pipeline as importable, testable Python modules
  components/   data_ingestion.py -> data_transformation.py -> model_trainer.py
  pipeline/     train_pipeline.py (full training run) and predict_pipeline.py (single-flight inference)
  utils.py      Shared feature engineering (engineer_features) used by BOTH pipelines,
                so training and inference can never drift apart
tools/        Thin CLI entry points that run the pipelines end-to-end
workflows/    One markdown SOP per tool: objective, inputs, expected output, edge cases
artifacts/    Generated model/preprocessor/data files (gitignored, regenerated on demand)
```

## Getting Started

```bash
pip install -r requirements.txt

# Train: ingest -> engineer features -> transform -> fit Random Forest & Decision Tree,
# keep whichever scores higher on the held-out test set
python tools/train_model.py

# Predict a fare for one flight
python tools/predict_fare.py --airline IndiGo --source Banglore \
  --destination "New Delhi" --date-of-journey 24/03/2019 \
  --dep-time 22:20 --arrival-time 01:10 --duration "2h 50m" \
  --total-stops non-stop
```

See [`workflows/train_model.md`](workflows/train_model.md) and
[`workflows/predict_fare.md`](workflows/predict_fare.md) for the full input
spec, expected output, and known edge cases for each tool.

### Web demo

`app.py` is a Streamlit UI over the same `predict_pipeline.py` used by the CLI
tool above — dropdowns/date pickers for flight details, a "Train model now"
button if no model exists yet, and the current model's test R² shown at the top.

```bash
streamlit run app.py
```

 ## Code and Resources Used
 
 - **Python Version:** 3.11
 - **Packages:** pandas, numpy, matplotlib, seaborn, scikit-learn.
 - **Dataset:** Dataset provided as part of the Udemy course "Build Data Science Real World Projects in AI, ML, NLP, and Time Series Domain".

## Data Cleaning
Given the provided dataset, I needed to clean up the data so that it was usable for our model. I made the following changes:

  - Converted relevant columns to datetime format to ensure proper handling of date and time information.
  - Extracted relevant components like day, month, and year from date columns.
  - Extracted hour and minute components from time columns.
  - Cleaned and standardized the duration data by splitting it into separate hour and minute components and calculated the total duration in minutes.
  - Applied one-hot encoding to categorical string features and used target-guided encoding to map categorical values to numerical values based on their relationship with the target variable.
  - Applied label encoding to map categorical values to numerical values.
  - Dropped original columns after extracting relevant features and removed unnecessary columns.

## EDA
I visualized the relationships between key features and the target variable, explored the distribution of values across different categories, and performed outlier 
detection to better understand the data’s patterns and anomalies.

<img src="https://github.com/user-attachments/assets/591602d7-ed55-4c9c-8da6-0dbc82351110" alt="Screenshot 2025-01-16 114720" width="300"/>
<img src="https://github.com/user-attachments/assets/8cec1294-08ad-4264-8299-6effea33cbf4" alt="Flights" width="400/">

## Model Building
I split the data into train and test sets with a test size of 20% (`src/components/data_ingestion.py`).

I tried two different models and evaluated them using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. I chose these metrics because they are relatively easy to interpret and outliers aren't particularly damaging to a regression read this way.

Models:
- **Random Forest:** Provides insights into the importance of different features in predicting the target variable (Price).
- **Decision Tree:** Offers a simpler model with clear interpretability, making it easier to understand how features impact the prediction, though its performance is lower than Random Forest here. The focus is on comparing model metrics such as accuracy, mean absolute error, and R-squared.

## Model Performance
The Random Forest model is the better model for predicting airline ticket prices, offering more accurate predictions and a better overall fit for the data.

- **Random Forest:**
  - Training Score: 0.98
  - R2 Score: 0.89
  - MAE: 654.28
  - MSE: 2,280,081.63
  - RMSE: 1509.99
- **Decision Tree:**
  - Training Score: 1.00 (overfits the training set)
  - R2 Score: 0.88
  - MAE: 685.36
  - MSE: 2,453,916.44
  - RMSE: 1566.50

These numbers come directly from running `python tools/train_model.py` against the current pipeline — rerun it after any data or feature-engineering change to get fresh, trustworthy numbers rather than stale ones baked into this README.

## Hypertune ML model
`src/components/model_trainer.py` hypertunes both models via `GridSearchCV` (see `src/utils.py::evaluate_models`) over a small parameter grid — `n_estimators` for Random Forest and `criterion` for Decision Tree — then keeps whichever model scores higher on the held-out test set.

## Personal case study: Honolulu ↔ Majuro

The main model above is trained on the Kaggle Indian-domestic-flights dataset,
which has plenty of airlines, cities, and stops to learn from. As a personal
extension, `src/pipeline/hawaii_case_study.py` predicts round-trip fares for
one specific route I care about: Honolulu (HNL) to Majuro, Marshall Islands (MAJ).

**Why it's a separate, smaller pipeline, not just new categories in the main
model:** United is the *only* carrier on this route (part of its twice-weekly
"Island Hopper" service — the same flight continues on to Kwajalein, Kosrae,
Pohnpei, Chuuk, and Guam). There's no Airline/Source/Destination/Stops variety
to one-hot encode. The real driver of price here is how far in advance you
book and which day of the week you fly.

**Data**: `notebook/data/hawaii_marshall_islands.csv` — 38 real fares read
directly from Google Flights' date-grid view on 2026-07-29, across two booking
windows (~1 week out and ~14-16 weeks out from that date). This is **not** a
scraped historical dataset (this route's booking channels don't expose one,
and bulk-scraping live pricing sites isn't something this project does) — it's
a single point-in-time snapshot, read by hand from one legitimate browsing
session. With n=38 and only two real booking-window clusters, this is a small,
illustrative regression (Linear Regression on days-until-departure,
departure/return day-of-week, and trip length), not a production-grade
forecasting model. A quick sanity check bears this out: predicting the exact
dates of a real quote in the dataset (Aug 5 → Aug 12, 2026) comes back at
$2,721 against an actual quoted price of $2,716 — but predictions for dates
between or beyond the two observed windows are extrapolations and should be
treated with more skepticism.

Also genuinely interesting: because Majuro is across the International Date
Line from Honolulu, this ~5.5 hour nonstop flight lands the *next calendar
day* local time.

```bash
python tools/predict_hawaii_fare.py --departure-date 15/11/2026 --return-date 22/11/2026
```

Or use the "🌺 Honolulu → Majuro" tab in the Streamlit app. See
[`workflows/predict_hawaii_fare.md`](workflows/predict_hawaii_fare.md) for details.

