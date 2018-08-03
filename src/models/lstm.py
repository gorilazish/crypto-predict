import sys
import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from math import sqrt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline

timeframe = '60min' # todo: define global config

def parse(x):
	return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# Load dataset
input_filename = 'data/processed/tweets_prices_60min.csv'
df = pd.read_csv(input_filename, usecols=[2, 3, 5, 6, 9, 13, 18, 19])
print('Data has been loaded\n')

# Move target values to last column
df['close_price'] = df['close']
df.drop(['close'], axis=1, inplace=True)
print(df.head())
values = df.values

# Prepare data for time-series
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
steps = 3
n_features = 8
n_observ = steps * n_features
reframed = series_to_supervised(scaled, steps, 1)

# Split into test and train, and input and output
values = reframed.values
print(pd.DataFrame(reframed).head())
X, Y = values[:, :n_observ], values[:, -1]
x_shape, y_shape = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], steps, n_features))
X_test = X_test.reshape((X_test.shape[0], steps, n_features))

# define base model
def baseline_model():
  neurons = 50
  # design network
  model = Sequential()
  model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam')
  return model

# fit network
lstm_model = baseline_model()
history = lstm_model.fit(X_train, y_train, epochs=500, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('reports/train_test_loss_' + timeframe +'.png')
plt.show()
plt.clf()

predictions = lstm_model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], steps * n_features))
# invert scaling for forecast
inv_pred = np.concatenate((X_test[:, -n_features:-1], predictions), axis=1)
inv_pred = scaler.inverse_transform(inv_pred)
inv_pred = inv_pred[:, -1]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = np.concatenate((X_test[:, -n_features:-1], y_test), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_pred))
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_y, c='blue', label='Real', linewidth=2)
plt.plot(inv_pred, c='orange', label='Predicted', linewidth=2)
plt.ylabel('Price')
plt.xlabel('Test set entries')
plt.legend()
plt.savefig('reports/prediction_'+ timeframe +'.png')
plt.show()
