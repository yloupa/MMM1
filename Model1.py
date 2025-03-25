import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error
import numpy as np

df = pd.read_csv('marketing_mix.csv')

print(df.head())
print(df.dtypes)
print(df.columns)

df['Date'] = pd.to_datetime(df['Date'])

print(df.dtypes)

df_sorted = df.sort_values(by=['Date']).reset_index(drop=True)

print(df_sorted.head())

X = df[['TikTok', 'Facebook', 'Google Ads']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# Feature names
feature_names = X.columns

# Get coefficients and intercept
coefficients = model.coef_
intercept = model.intercept_

# Create a DataFrame for better visualization
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

print("Intercept (Base Sales):", intercept)
print("\nCoefficients:\n", coef_df)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")


#'model2

df['Date'] = pd.to_datetime(df['Date'])
df['Day_of_Week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Week_of_Year'] = df['Date'].dt.isocalendar().week

# Create lag features (Previous week's sales)
df['Sales_Lag_1'] = df['Sales'].shift(1)  # Previous week's sales
df['Sales_Lag_2'] = df['Sales'].shift(2)
df['Sales_Lag_3'] = df['Sales'].shift(3)

# Fill NaN values (for lag features)
df.fillna(0, inplace=True)

X2 = df[['TikTok', 'Facebook', 'Google Ads', 'Day_of_Week',
        'Month', 'Year', 'Week_of_Year', 'Sales_Lag_1','Sales_Lag_2','Sales_Lag_3']]
y2 = df['Sales']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Initialize the model
model2 = LinearRegression()

# Train the model on the training data
model2.fit(X2_train, y2_train)

# Make predictions on the test set
y2_pred = model2.predict(X2_test)


mae = mean_absolute_error(y2_test, y2_pred)
mse = mean_squared_error(y2_test, y2_pred)
rmse = root_mean_squared_error(y2_test, y2_pred)
r2 = r2_score(y2_test, y2_pred)


# Feature names
feature_names2 = X2.columns

# Get coefficients and intercept
coefficients2 = model2.coef_
intercept2 = model2.intercept_

# Create a DataFrame for better visualization
coef_df2 = pd.DataFrame({
    'Feature': feature_names2,
    'Coefficient': coefficients2
})

print("Intercept (Base Sales):", intercept2)
print("\nCoefficients:\n", coef_df2)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")


# Extract seasonality features
df['Quarter'] = df['Date'].dt.quarter  # Q1–Q4
df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6]).astype(int)  # 1 for Saturday/Sunday

# Create cyclic features for weekly and yearly seasonality
df['Week_Sin'] = np.sin(2 * np.pi * df['Week_of_Year'] / 52)
df['Week_Cos'] = np.cos(2 * np.pi * df['Week_of_Year'] / 52)

df['Month_Sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month'] / 12)

# (Optional) Define major holidays or promotions
holidays = pd.to_datetime(['2019-12-22', '2019-12-29', '2018-12-23','2018-12-30','2020-12-20','2020-12-27'])  # Example dates
df['Is_Holiday'] = df['Date'].isin(holidays).astype(int)

# Define the new feature set
X_seasonal = df[['TikTok', 'Facebook', 'Google Ads',
                 'Month', 'Year', 'Week_of_Year', 'Sales_Lag_1','Sales_Lag_2','Sales_Lag_3',
                 'Quarter', 'Is_Weekend', 'Week_Sin', 'Week_Cos',
                 'Month_Sin', 'Month_Cos', 'Is_Holiday']]
y_seasonal = df['Sales']

# Split the data
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_seasonal, y_seasonal, test_size=0.2, random_state=42)

# Train the model
model_seasonal = LinearRegression()
model_seasonal.fit(X_train_s, y_train_s)

# Make predictions
y_pred_s = model_seasonal.predict(X_test_s)

# Evaluate performance
mae = mean_absolute_error(y_test_s, y_pred_s)
mse = mean_squared_error(y_test_s, y_pred_s)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_s, y_pred_s)

# Feature importance
coef_df_s = pd.DataFrame({
    'Feature': X_seasonal.columns,
    'Coefficient': model_seasonal.coef_
})

# Display results
print("Coefficients:\n", coef_df_s)
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")