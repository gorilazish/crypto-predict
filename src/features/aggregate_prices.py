import pandas as pd
import datetime
import numpy as np

timeframe = '15min'
print('Aggregating OHLCV ['+ timeframe +']')
input_filename = 'data/interim/ohlcv_indexed.csv'
output_filename = 'data/interim/ohlcv_aggregated_' + timeframe + '.csv'
df = pd.read_csv(input_filename, engine="python", index_col=0)

def agg_to_timeframe():
  df.index = pd.DatetimeIndex(df.index)
  return df.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})

def get_delta_percent(start, end):
  return round((end - start) / start * 100, 2)

# Convert to selected timeframe
df = agg_to_timeframe()
df = df.fillna(method='ffill')

# Calculate delta percentage and add column to df
delta = np.array([])
for o, c in zip(df['open'], df['close']):
  delta = np.append(delta, get_delta_percent(o, c))
df['price_change'] = delta

df.to_csv(output_filename, encoding='utf-8', header=True)
