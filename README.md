# Predict Fare of Airline Tickets using ML: Project Overview
- **Objective**: Develop a machine learning model to predict airline ticket prices based on features like source, destination, airline, and number of stops.
- **Dataset**: Includes flight details such as airline, route, departure and arrival times, stops, and pricing information.
- **Approach**: Apply regression algorithms (e.g., Decision Trees, Random Forest) to predict ticket prices, with preprocessing like handling missing values and encoding categorical features.
- **Model Evaluation**: Evaluate model performance using metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
- **Data Preprocessing**: Includes handling missing values, encoding categorical features, and scaling numerical data to improve model accuracy.
- **Technologies Used**: Python, Pandas, Scikit-learn, Matplotlib, and other data science libraries for analysis and model building.

 ## Code and Resources Used
 - **Python Version:** 3.11
 - **Packages:** pandas, numpy, matplotlib, seaborn, sklearn, pickle.
 - **Dataset:** Dataset provided as part of the Udemy course "Build Data Science Real World Projects in AI, ML, NLP, and Time Series Domain".

## Data Cleaning
- Given the provided dataset, I needed to clean up the data so that it was usable for our model. I made the following changes and created the following variables:
  - Converted relevant columns to datetime format to ensure proper handling of date and time information.
  - Extracted relevant components like day, month, and year from date columns.
  - Extracted hour and minute components from time columns.
  - Cleaned and standardized the duration data by splitting it into separate hour and minute components and calculated the total duration in minutes.
  - Applied one-hot encoding to categorical string features, creating binary columns for each unique category.
  - Used target-guided encoding for certain categorical features, mapping categories to numerical values based on their frequency or a custom scheme.
    
 
