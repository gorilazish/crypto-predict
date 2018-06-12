import pandas as pd
import regex as re
import os

timeframe = '15min'

def transform(df):
  df.index = pd.DatetimeIndex(df['datetime'])
  df['tweet_count'] = df['datetime']
  order = df.columns
  return df.resample(timeframe).agg({'datetime': 'first', 'followers': 'sum', 'compound_sum': 'sum', 'compound_mean': 'mean', 'followers_compound': 'sum', 'positive_sum': 'sum', 'positive_mean': 'mean', 'followers_positive': 'sum', 'neutral_sum': 'sum', 'neutral_mean': 'mean', 'followers_neutral': 'sum', 'negative_sum': 'sum', 'negative_mean': 'mean', 'followers_negative': 'sum', 'tweet_count': 'count'})[order]

input_filename = "data/interim/tweets_expanded.csv"
output_filename = "data/interim/tweets_aggregated_"+ timeframe +".csv"
columns = ['datetime', 'followers', 'compound_sum', 'compound_mean', 'followers_compound', 'positive_sum', 'positive_mean', 'followers_positive', 'neutral_sum', 'neutral_mean', 'followers_neutral', 'negative_sum', 'negative_mean', 'followers_negative']

def save(data):
  data.to_csv(output_filename,  header='column_names', index=False)

data = pd.read_csv(input_filename, usecols=columns)
data = transform(data)
save(data)