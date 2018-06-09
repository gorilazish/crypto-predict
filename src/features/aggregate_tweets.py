import pandas as pd
import regex as re
import sys

timeframe = '60min'
twitter_sentiment_filename = 'data/interim/tweets_expanded.csv'
#cols = ['datetime', 'compound', 'positive', 'neutral', 'negative']
output_filename = 'data/interim/tweets_aggregated.csv'
df = pd.read_csv(twitter_sentiment_filename, index_col=0)

def transform():
  df.index = pd.DatetimeIndex(df.index)
  return df.resample(timeframe).agg({'datetime': 'first', 'followers': 'sum', 'compound_sum': 'sum', 'compound_mean': 'mean', 'followers_compound': 'sum', 'positive_sum': 'sum', 'positive_mean': 'mean', 'followers_positive': 'sum', 'neutral_sum': 'sum', 'neutral_mean': 'mean', 'followers_neutral': 'sum', 'negative_sum': 'sum', 'negative_mean': 'mean', 'followers_negative': 'sum'}).dropna(axis=0, how='any')
aggregated = transform()
aggregated.to_csv(output_filename)
