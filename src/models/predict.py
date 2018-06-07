import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def normalize(data):
    return (data - data.min()) / (data.max() - data.min())

def standardize(data):
    return (data - data.mean()) / data.std()

# load dataset
df = pd.read_csv('C:/Users/MANS/Documents/AzureML/P6-predict-crypto/assets/tweets_prices_april_1min.csv')

predict_col = 'Close'
df[predict_col] = df[predict_col].shift(1)
df = df.drop(df.index[0])  # drop first row
df = df.drop(df.index[-1])  # drop last row
time = pd.to_datetime(df['datetime'])
df = df.drop(['datetime'], 1)

df = normalize(df)

X = df[['compound_sum','compound_mean','positive_sum','positive_mean','negative_sum','negative_mean']]
Y = df[predict_col]

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(6, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# fix random seed for reproducibility
seed = 42
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=256, verbose=1)

kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, Y)
prediction = estimator.predict(X)

plt.plot(time, Y, label='expected')
plt.plot(time, prediction, label='prediction')
plt.show()