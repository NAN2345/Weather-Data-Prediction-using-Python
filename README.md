# Weather Dataset Analysis and Temperature Prediction

## Overview

This project aims to analyze the Weather Dataset, which provides per-hour information about weather conditions at a specific location. The dataset includes various features such as temperature, dew point temperature, relative humidity, wind speed, visibility, pressure, and weather conditions. The primary focus of the project is to perform exploratory data analysis (EDA), visualize weather trends, and predict temperature using machine learning models.

## Dataset

The dataset used for this project is available as a CSV file named "Weather dataset for python.csv". It contains the following columns:

- Date/Time: Date and time of the observation
- Temp_C: Temperature in Celsius
- Dew Point Temp_C: Dew point temperature in Celsius
- Rel Hum_%: Relative humidity in percentage
- Wind Speed_km/h: Wind speed in kilometers per hour
- Visibility_km: Visibility in kilometers
- Press_kPa: Pressure in kilopascals
- Weather Condition: Description of the weather condition

## Analysis Steps

1. **Data Loading and Exploration**: Load the dataset into a Pandas DataFrame and explore its structure, including the first few rows, shape, data types, and basic statistics.
2. **Data Preprocessing and Cleaning**: Handle any missing values, format data types, and preprocess features as needed.
3. **Exploratory Data Analysis (EDA)**: Perform EDA to gain insights into the dataset, visualize distributions, correlations, and trends among weather variables.
4. **Visualization**: Create visualizations such as histograms, line plots, scatter plots, and heatmaps to visualize weather trends and relationships between variables.
5. **Correlation Analysis**: Calculate and visualize the correlation matrix to understand the relationships between different weather variables.
6. **Temperature Prediction**: Use machine learning models to predict temperature based on other weather variables.
   - Three models are used: Linear Regression, Random Forest Regressor, and Decision Tree Regressor.
   - Evaluate the performance of each model using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

## Requirements

To run the code, you need to have the following Python libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Clone the repository to your local machine:

```
git clone https://github.com/your-username/weather-dataset-analysis.git
```

2. Navigate to the project directory:

```
cd weather-dataset-analysis
```

3. Open and run the Jupyter Notebook `Weather_Analysis_Prediction.ipynb` to execute the code cells and analyze the results.

## Acknowledgments

The Weather Dataset used in this project is sourced from [provide the source if applicable].
