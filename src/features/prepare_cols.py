import pandas as pd
import regex as re
import os
import math

input_filename = "data/interim/tweets_clean.csv"
output_filename = "data/interim/tweets_expanded.csv"
chunksize = 100000
counter = 0
total_rows = sum(1 for row in open(input_filename, 'r'))
print('Total rows - ' + str(total_rows))
print('Total chunks - ' + str(math.ceil(total_rows / chunksize)))

def prep_cols (data):
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
  return data.dropna(axis=0, how='any')

def save(data):
    global counter
    # if file does not exist write header
    if not os.path.isfile(output_filename):
       data.to_csv(output_filename,  header='column_names', index=False)
    else: # else it exists so append without writing the header
       data.to_csv(output_filename, mode='a', header=False, index=False)
    print('processing:{} '.format(str(counter)))
    print(round((counter / total_rows) * 100, 2), '%\n')
    counter += chunksize

for chunk in pd.read_csv(input_filename, engine='python', chunksize=chunksize, encoding='utf-8-sig'):
    data = prep_cols(chunk)
    save(data)