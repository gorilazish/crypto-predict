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

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

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
# data_filename = 'assets/60min_final.csv'
data_filename = 'assets/15min_prices_tweets_april.csv'
print('\nLoading prep file: ', data_filename)
df = pd.read_csv(data_filename, index_col=0)
print('Data has been loaded\n')
df.index = pd.DatetimeIndex(df.index)
values = df.values

# specify columns to plot
# groups = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# i = 1
# # plot each column
# plt.figure()
# for group in groups:
# 	plt.subplot(len(groups), 1, i)
# 	plt.plot(values[:, group])
# 	plt.title(df.columns[group], y=0.5, loc='right')
# 	i += 1
# plt.show()

# Prepare data for time-series
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]], axis=1, inplace=True)

# Split into test and train, and input and output
values = reframed.values
X, Y = values[:, :-1], values[:, -1]
x_shape, y_shape = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# reshape input to be 3D [samples, timesteps, features]
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# define base model
def baseline_model():
  neurons = 50

  # design network
  model = Sequential()
  model.add(LSTM(neurons, input_shape=(X_train.shape[1], X_train.shape[2])))
  model.add(Dense(1))
  model.compile(loss='mae', optimizer='adam')
  return model

lstm_model = baseline_model()
# fit network
history = lstm_model.fit(X_train, y_train, epochs=10000, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)

# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plt.show()
plt.clf()

predictions = lstm_model.predict(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[2]))
# invert scaling for forecast
inv_pred = np.concatenate((X_test[:, :-1], predictions), axis=1)
inv_pred = scaler.inverse_transform(inv_pred)
inv_pred = inv_pred[:,-1:]
# invert scaling for actual
y_test = y_test.reshape((len(y_test), 1))
inv_y = np.concatenate((X_test[:, :-1], y_test), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1:]

# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_pred))
print('Test RMSE: %.3f' % rmse)
plt.plot(inv_pred, c='orange', label='Predicted', linewidth=3)
plt.plot(inv_y, c='b', label='Real', linewidth=3)
plt.ylabel('Price change %')
plt.xlabel('Test set entries')
plt.legend()
plt.show()
