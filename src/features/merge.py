import pandas as pd
import sys

timeframe = '15min'
print('Merging to ', timeframe)
twitter_sentiment_filename = 'data/interim/tweets_aggregated_'+ timeframe +'.csv'
ohlcv_filename = 'data/interim/ohlcv_aggregated_'+ timeframe +'.csv'
output_filename = 'data/processed/tweets_prices_'+ timeframe +'.csv'

df1 = pd.read_csv(twitter_sentiment_filename, index_col=0)
df2 = pd.read_csv(ohlcv_filename, index_col=0)

aggregate = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
print('Merged. Shape: ')
print(aggregate.shape, '/n')
aggregate.to_csv(output_filename)
