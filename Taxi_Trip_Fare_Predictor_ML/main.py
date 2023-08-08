import pandas as pd
import numpy as np

from sklearn.metrics import explained_variance_score as evs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge


# Data Import
df = pd.read_csv('C:/Users/User/Desktop/Alex Files/Taxi_Trip_Fare_Predictor_ML/taxi_fare/train.csv')
test_df = pd.read_csv('C:/Users/User/Desktop/Alex Files/Taxi_Trip_Fare_Predictor_ML/taxi_fare/test.csv')

# EDA & Preprocessing
print(df.info())

df = df.drop('fare', axis= 1)

print(df)


# Train Test Split
features = df.drop('total_fare', axis= 1)
target = df['total_fare']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size= 0.2, random_state= 42)


# Model Training & Accuracy Score
models = [DecisionTreeRegressor(),
          GradientBoostingRegressor(),
          XGBRegressor(),
          LinearRegression(),
          Ridge()]

for model in models:
    print(model)

    model.fit(X_train, Y_train)

    pred_train = model.predict(X_train)
    print(f'Train accuracy : {evs(Y_train, pred_train)}')

    pred_test = model.predict(X_test)
    print(f'Test Accuracy : {evs(Y_test, pred_test)}\n')