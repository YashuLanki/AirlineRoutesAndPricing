# Predict Fare of Airline Tickets using ML: Project Overview

- **Objective**: Develop a machine learning model to predict airline ticket prices based on features like source, destination, airline, and number of stops.
  - **Dataset Features**: Includes flight details such as airline, date of journey, route, departure and arrival times, source, destination, total stops, additional info, and price.
  - **Approach**: Apply regression algorithms (e.g., Decision Trees, Random Forest) to predict ticket prices, with preprocessing like handling missing values and encoding categorical features.
  - **Model Evaluation**: Evaluate model performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
  - **Data Preprocessing**: Includes handling missing values, encoding categorical features, and scaling numerical data to improve model accuracy.
  - **Technologies Used**: Python, Pandas, Scikit-learn, Matplotlib, and other data science libraries for analysis and model building.

 ## Code and Resources Used
 
 - **Python Version:** 3.11
 - **Packages:** pandas, numpy, matplotlib, seaborn, sklearn, pickle.
 - **Dataset:** Dataset provided as part of the Udemy course "Build Data Science Real World Projects in AI, ML, NLP, and Time Series Domain".

## Data Cleaning
Given the provided dataset, I needed to clean up the data so that it was usable for our model. I made the following changes and created the following variables:

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
I split the data into train and tests sets with a test size of 25%.

I tried two different models and evaluated them using using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared. I chose to use these metrics because it is relatively easy to interpret and outliers aren't particularly bad for these models. 

Models: 
- **Random Forest:** Provides insights into the importance of different features in predicting the target variable (Price).
- **Decision Tree:** Offers a simpler model with clear interpretability, making it easier to understand how features impact the prediction, though its performance might be lower compared to Random Forest. The focus here is on comparing model metrics such as accuracy, mean absolute error, and R-squared.

## Model Performance
The Random Forest model is the better model for predicting the price of airline ticket, offering more accurate predictions and a better overall fit for the data. 

- **Random Forest:**
  - Training Score: 0.95
  - r2 Score: 0.81
  - MAE: 1175.16
  - MSE: 3668703.96
  - RMSE: 1915.39
- **Decision Tree:**
  - Training Score: 0.96
  - r2 Score: 0.67
  - MAE: 1423.02
  - MSE: 6348160.96
  - RMSE: 2519.55

