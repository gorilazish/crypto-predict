import pandas as pd
import sys
ROOT_DIR = '../../'

ohlcv_filename = ROOT_DIR + 'data/interim/ohlcv.csv'
twitter_sentiment_filename = ROOT_DIR + 'data/interim/tweets.csv'
output_filename = ROOT_DIR + 'data/processed/aggregated.csv'

df1 = pd.read_csv(twitter_sentiment_filename, index_col=0)
df2 = pd.read_csv(ohlcv_filename, index_col=0)

aggregate = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
aggregate.to_csv(output_filename)
