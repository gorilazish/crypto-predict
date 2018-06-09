
import pandas as pd
import regex as re

def transform(df):
  timeframe = '60min'
  df.index = df['datetime']
  order = df.columns
  return df.resample(timeframe).agg({'datetime': 'first', 'followers': 'sum', 'compound_sum': 'sum', 'compound_mean': 'mean', 'followers_compound': 'sum', 'positive_sum': 'sum', 'positive_mean': 'mean', 'followers_positive': 'sum', 'neutral_sum': 'sum', 'neutral_mean': 'mean', 'followers_neutral': 'sum', 'negative_sum': 'sum', 'negative_mean': 'mean', 'followers_negative': 'sum'})[order]

input_filename = "data/interim/tweets_expanded.csv"
output_filename = "data/interim/tweets_aggregated.csv"
columns = ['datetime', 'followers', 'compound_sum', 'compound_mean', 'followers_compound', 'positive_sum', 'positive_mean', 'followers_positive', 'neutral_sum', 'neutral_mean', 'followers_neutral', 'negative_sum', 'negative_mean', 'followers_negative']

data = pd.read_csv(input_filename, usecols=columns, parse_dates=['datetime'])
print(data.head())
data = transform(data)
print(data.head())
data.to_csv(output_filename, header=True, index=False)
