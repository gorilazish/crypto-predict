import pandas as pd

ROOT_DIR = ''

ohlcv_filename = ROOT_DIR + 'data/interim/ohlcv.csv'
df = pd.read_csv(ohlcv_filename, engine='python')
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.rename(columns={'timestamp': 'datetime'}, inplace=True)
df.to_csv(ohlcv_filename)

