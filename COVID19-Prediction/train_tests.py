import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import signal
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


from sklearn import linear_model

df = pd.read_csv('datasets/brazil_covid19_cities.csv')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df["week"] = df['date'].dt.isocalendar().week
df = df.groupby('state').last().reset_index()

train_df = df[:int(len(df) * (80 / 100))]  # 80% of the data
train_x = train_df[['week', 'cases']]
train_y = train_df['deaths']

test_df = df[int(len(df) * (80 / 100)):]  # 20% of the data
test_x = test_df[['week', 'cases']]
test_y = test_df['deaths']

regr = linear_model.LinearRegression()
regr.fit(train_x.values, train_y)
print(regr.predict([[1, 200]]))
linear_predicted_rating = regr.predict(test_x)
accuracy = metrics.r2_score(test_y, linear_predicted_rating)
print("Accuracy: ", accuracy)


