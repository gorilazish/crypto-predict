import pandas as pd
import regex as re

input_filename = "data/interim/tweets_clean.csv"
output_filename = "data/interim/tweets_expanded.csv"

data = pd.read_csv(input_filename, engine="python", index_col=0)
data = data.drop(['tweet'], 1)
data['followers'] = data['followers_count']
data['followers_compound'] = data['followers'] * data['compound']
data['followers_positive'] = data['followers'] * data['positive']
data['followers_neutral'] = data['followers'] * data['neutral']
data['followers_negative'] = data['followers'] * data['negative']
data['compound_sum'] = data['compound']
data['compound_mean'] = data['compound']
data['positive_sum'] = data['positive']
data['positive_mean'] = data['positive']
data['neutral_sum'] = data['neutral']
data['neutral_mean'] = data['neutral']
data['negative_sum'] = data['negative']
data['negative_mean'] = data['negative']
data.to_csv(output_filename, encoding='utf-8', header=True, index=False)
