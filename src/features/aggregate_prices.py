import pandas as pd
import datetime
import numpy as np

print('Loading OHLCV data')
ohlcv_filename = 'data/interim/ohlcv.csv'
df = pd.read_csv(ohlcv_filename, engine="python", index_col=0)
print('OHLCV data was loaded\n')
print(df.head())

def agg_to_timeframe():
  df.index = pd.DatetimeIndex(df.index)
  return df.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})

def get_delta_percent(start, end):
  return round((end - start) / start * 100, 2)

# Convert to selected timeframe
timeframe = '60min'
df = agg_to_timeframe()

# Calculate delta percentage and add column to df
delta = np.array([])
for o, c in zip(df['open'], df['close']):
  delta = np.append(delta, get_delta_percent(o, c))
df['price_change'] = delta

df.to_csv(ohlcv_filename, encoding='utf-8', header=True)
