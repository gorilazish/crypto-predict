import sys
import os
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create the outputs folder - save any outputs you want managed by AzureML here
os.makedirs('./outputs', exist_ok=True)

# Load dataset
# data_filename = 'assets/60min_final.csv'
data_filename = 'assets/15min_prices_tweets_april.csv'
print('Loading prep file: ', data_filename)
df = pd.read_csv(data_filename)
print('File has been loaded \n')

shift = 4
# Split dataset into X and Y
X = df.loc[:,'followers':'standard_deviation']
X_forecast = X[-shift:]
X = X[:-shift]
Y = df['price_change'].shift(-shift)[:-shift]
x_shape, y_shape = X.shape
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
print('Training data length - ', len(X_train))
print('Testing data length - ', len(y_train))
print('\n')

# define base model
def baseline_model():
	model = Sequential()
	model.add(Dense(15, input_dim=y_shape))
	model.add(Dropout(0.1))
	model.add(Dense(7, activation='relu'))
	model.add(Dense(1))
	# Compile model
	print(model.summary())
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

seed = 2
epochs = 1000
batch_size = 128
np.random.seed(seed)

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=epochs, batch_size=batch_size, verbose=1)))
pipeline = Pipeline(estimators)
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
# kfold = KFold(n_splits=2, random_state=seed)
# results = cross_val_score(pipeline, X, Y, cv=kfold)
# predictions = cross_val_predict(pipeline, X, Y, cv=kfold)
# print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

# Plot outputs
fig, ax = plt.subplots()
y_pred = pd.DataFrame(y_pred)
y_pred.index = pd.DataFrame(y_test).index
y_pred = y_pred.shift(shift)[shift:]
y_pred_data = df.loc[y_pred.index[0] : y_pred.index[-1]]
y_pred_data.index = pd.DatetimeIndex(y_pred_data['datetime'])

plt.plot(y_pred_data.index, y_pred_data['price_change'], color='blue', linewidth=3, label='Real price change')
plt.plot(y_pred_data.index, y_pred, color='orange', linewidth=3, label='Predicted price change')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
plt.xlabel('Date')
plt.ylabel('Price change %')
plt.legend()
plt.show()

# from_date = '2018-04-21 10:00:00'
# to_date = '2018-04-24 13:00:00'

# df.index = pd.DatetimeIndex(df['datetime'])
# predictions = pd.DataFrame(predictions).shift(shift)[shift:]
# predictions.index = pd.DatetimeIndex(df['datetime'][:predictions.shape[0]])
# plt.plot(predictions.loc[from_date : to_date].index, predictions.loc[from_date:to_date][0], c='orange', label='Predicted price change', linewidth=3)
# plt.plot(predictions.loc[from_date:to_date].index, df.loc[from_date:to_date]['price_change'], c='b', label='Real price change', linewidth=3)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
# plt.xlabel('Date')
# plt.ylabel('Price change %')
# plt.legend()
# plt.show()