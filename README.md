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

