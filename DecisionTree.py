import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('bmh')

df = pd.read_csv('TSLA_2019_Prices.csv')  # Load Data
df = df[['close']]  # Get close price only
future_days = 30  # Create a variable to predict 'x' days out into the future

# Create a new column (target) shifted 'x' units/days up
df['Prediction'] = df[['close']].shift(-future_days)
# Convert it to a numpy array and remove the last set days
X = np.array(df.drop(['Prediction'], 1))[:-future_days]

# Create the target data set (y) and convert it to a numpy array and get all of the target values except that last 'x' rows/days
y = np.array(df['Prediction'])[:-future_days]

# Split the data into 75% training and 25% testing
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Decision Tree Regressor Model
tree = DecisionTreeRegressor().fit(x_train, y_train)
# Last 'x' rows of the data set
x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

# Decision Tree prediction
tree_prediction = tree.predict(x_future)
print(tree_prediction)

# Plot data
predictions = tree_prediction

valid = df[X.shape[0]:]
valid['Prediction'] = predictions
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price ($)')
plt.plot(df['close'])
plt.plot(valid[['close', 'Prediction']])
plt.legend(['Orig', 'Valid', 'Predicted'])
plt.show()
