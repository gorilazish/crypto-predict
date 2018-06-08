import pandas as pd
import regex as re
import sys
ROOT_DIR = '../../'

timeframe = '60min'
twitter_sentiment_filename = ROOT_DIR + 'data/interim/data_sentiment2014-01_2018-04.csv'
cols = ['datetime', 'compound', 'positive', 'neutral', 'negative']
output_filename = ROOT_DIR + 'data/interim/tweets.csv'
df = pd.read_csv(twitter_sentiment_filename, usecols=cols, index_col=0)

def transform():
  df.index = pd.DatetimeIndex(df.index)
  return df.resample(timeframe).agg({'compound': 'mean', 'positive': 'mean', 'neutral': 'mean', 'negative': 'mean'}).dropna(axis=0, how='any')

aggregated = transform()
aggregated.to_csv(output_filename)