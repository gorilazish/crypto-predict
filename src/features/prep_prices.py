import pandas as pd
import datetime
import numpy as np

print('Loading OHLCV data')
# df = package.run('../p6_prep.dprep', dataflow_idx=0)
df = pd.read_csv('../assets/btc_prices_april.csv')
print('OHLCV data was loaded\n')

def agg_to_timeframe(timeframe):
  df.index = pd.DatetimeIndex(pd.to_datetime(df['timestamp'], unit='ms'))
  df['standard_deviation'] = df['close']
  order = df.columns[1:]
  return df.resample(timeframe).agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum', 'standard_deviation': lambda x: round(np.std(x), 2)})[order]

def get_delta_percent(start, end):
  return round((end - start) / start * 100, 2)

# Convert to selected timeframe
timeframe = '60min'
df = agg_to_timeframe(timeframe)

# Calculate delta percentage and add column to df
delta = np.array([])
for o, c in zip(df['open'], df['close']):
  delta = np.append(delta, get_delta_percent(o, c))
df['price_change'] = delta

df.to_csv('../outputs/' + timeframe + '_agg_btc_prices.csv', encoding='utf-8', header=True)