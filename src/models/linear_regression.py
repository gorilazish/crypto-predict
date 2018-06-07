import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn import metrics


# Load dataset
# data_filename = 'assets/60min_final.csv'
data_filename = 'assets/15min_prices_tweets_april.csv'
print('Loading prep file: ', data_filename)
df = pd.read_csv(data_filename)
print('File has been loaded \n')

shift = 4
# Split dataset into X and Y
X = df.loc[:,'followers':'standard_deviation']
print(X.head())
X = preprocessing.scale(X)
X_forecast = X[-shift:]
X = X[:-shift]
Y = df['price_change'].shift(-shift)[:-shift]
x_shape, y_shape = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
y_pred = y_pred * 2 # for some reason predicted values are in much smaller scale

# The coefficients
print('Coefficients: \n', reg.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
print('Score: %.2f' % reg.score(X_test, y_test))
print(cross_val_score(reg, X_train, y_train, cv=10))
   
# Plot outputs
fig, ax = plt.subplots()

y_test_dates = df.loc[pd.DataFrame(y_test).index[0]:pd.DataFrame(y_test).index[-1]]
y_test_dates.index = pd.DatetimeIndex(y_test_dates['datetime'])
plt.plot(y_test_dates.index, y_test, color='blue', linewidth=3, label='Real')
plt.plot(y_test_dates.index, y_pred, color='orange', linewidth=3, label='Predicted')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.xlabel('Date')
plt.ylabel('Price change %')
plt.legend()
plt.show()