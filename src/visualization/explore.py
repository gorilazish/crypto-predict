import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os

timeframe = '60min' # todo: make timeframe global variable

input_filename = 'data/processed/tweets_prices_'+ timeframe +'.csv'
df = pd.read_csv(input_filename, index_col=0)
df.index = pd.DatetimeIndex(df.index)
# Selected period of time
start_date = '15/05/2018 10:00:00'
end_date = '16/05/2018 10:00:00'
df = df.loc[(df.index > start_date) & (df.index <= end_date)]

# Scatter feature relation to target label
features = ['followers_positive', 'compound_sum', 'tweet_count']
target = 'close'

for feature in features:
  plt.scatter(x=df[feature], y=df[target], alpha=0.4)
  plt.xlabel(feature)
  plt.ylabel(target)
  plt.savefig('reports/'+ feature +'-'+ target +'_'+ timeframe +'.png')
  plt.clf()





