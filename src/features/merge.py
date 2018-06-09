import pandas as pd
import sys

ohlcv_filename = 'data/interim/ohlcv.csv'
twitter_sentiment_filename = 'data/interim/tweets_aggregated.csv'
output_filename = 'data/processed/tweets_prices.csv'

df1 = pd.read_csv(twitter_sentiment_filename, index_col=0)
df2 = pd.read_csv(ohlcv_filename, index_col=0)

aggregate = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
aggregate.to_csv(output_filename)
