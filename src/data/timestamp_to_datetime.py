import pandas as pd

input_filename = 'data/interim/ohlcv_with_header.csv'
output_filename = 'data/interim/ohlcv_indexed.csv'
df = pd.read_csv(input_filename, engine='python', index_col=0)
df.index = pd.to_datetime(df.index, unit='ms')
df.index.name = 'datetime'
df.to_csv(output_filename, encoding='utf-8', header=True)
