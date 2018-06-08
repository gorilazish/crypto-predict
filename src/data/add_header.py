import pandas as pd
import os

input_filename = 'data/raw/ohlcv.csv'
output_filename = 'data/interim/ohlcv.csv'

header = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.read_csv(input_filename, engine='python', names=header, index_col=0)

df.to_csv(output_filename)