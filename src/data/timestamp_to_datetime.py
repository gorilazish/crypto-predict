import pandas as pd

ohlcv_filename = 'data/interim/ohlcv.csv'
df = pd.read_csv(ohlcv_filename, engine='python', index_col=0)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.rename(columns={'timestamp': 'datetime'}, inplace=True)
df.to_csv(ohlcv_filename, encoding='utf-8', header=True, index=False)
