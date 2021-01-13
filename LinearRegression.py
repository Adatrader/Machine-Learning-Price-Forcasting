import sklearn
from tiingo import TiingoClient
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import requests
plt.style.use('bmh')


now = datetime.now()
dtFormat = now.strftime("%Y-%m-%d")
config = {}
config['session'] = True
config['api_key'] = "<API Key here>"
client = TiingoClient(config)
# Get dataframe for SPY
df = client.get_dataframe("SPY",
                          frequency='1Min',
                          startDate='2020-11-01',
                          endDate=dtFormat)
df = df[['close']]
forecast_future = int(30)  # predict 30 timeframes into the future
# column with data shifted 30 units up
df['prediction'] = df[['close']].shift(-forecast_future)

X = np.array(df.drop(['prediction'], 1))
print(X)
X = preprocessing.scale(X)
X_forecast = X[-forecast_future:]  # set X_forecast equal to last 30
print(X)
print(X_forecast)
X = X[:-forecast_future]  # remove last 30 from X
y = np.array(df['prediction'])
y = y[:-forecast_future]
# Linear regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Training
clf = LinearRegression()
clf.fit(X_train, y_train)
# Testing
confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)
forecast_prediction = clf.predict(X_forecast)
print(forecast_prediction)
# Visualize the data
predictions = forecast_prediction
valid = df[X.shape[0]:]
valid['Prediction'] = predictions
fig = plt.figure(figsize=(8, 4))
plt.title(dtFormat + ' SPY Price Prediction')
plt.xlabel('')
plt.ylabel('Close Price ($)')
strG = dtFormat + ' 14:30:00+00:00'
plt.plot(df.truncate(before=strG)['close'])
plt.plot(valid[['close', 'Prediction']])
plt.legend(['Orig', 'Valid', 'Predicted'])
# plt.show()
fig.savefig('plot.png')
