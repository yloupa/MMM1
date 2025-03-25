import pandas as pd
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,root_mean_squared_error

df = pd.read_csv('marketing_mix.csv')

print(df.head())
print(df.dtypes)
print(df.columns)

df['Date'] = pd.to_datetime(df['Date'])

for date in df['Date']:
    if date.month in [11, 12]:
        print(date.strftime('%Y-%m-%d'))

