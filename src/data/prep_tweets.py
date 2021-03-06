import pandas as pd
import re
import os
import math
from vaderSentiment import vaderSentiment
import csv


input_filename = 'data/raw/live_tweets.csv'
output_filename = 'data/interim/live_tweets_clean.csv'
columns = ['datetime','tweet','followers_count', 'compound', 'positive', 'neutral', 'negative']

counter = 0
chunksize = 10000
total_rows = sum(1 for row in open(input_filename, 'r'))
print('Total chunks - ' + str(total_rows / chunksize))

def clean_tweet_text(text):
    # 1. Lower case
    processed_text = text.lower()
    # 2. Remove mentions
    processed_text = re.sub(r'@[\w\S]+', '', processed_text)
    # 3. Remove hashtag symbols
    processed_text = re.sub('#', '', processed_text)
    # 4. Remove 'rt' from retweeted tweets
    processed_text = re.sub(r'rt ', '', processed_text)
    # 5. Remove alone numbers and special characters
    processed_text = re.sub(r'\s[\W\d:]+|[\W\d:]+\s', ' ', processed_text)
    # 7. remove links
    processed_text = re.sub('https?://[A-Za-z0-9./]+','', processed_text)
    # 8. Remove extra white spaces that were introduced in previous steps
    processed_text = re.sub(r' +', ' ', processed_text)

    return processed_text.strip()

def sentiment(data):
    analyzer = vaderSentiment.SentimentIntensityAnalyzer()
    compound = []
    positive = []
    neutral = []
    negative = []
    for t in data['tweet']:
        vs = analyzer.polarity_scores(str(t))
        compound.append(vs['compound'])
        positive.append(vs['pos'])
        neutral.append(vs['neu'])
        negative.append(vs['neg'])
    data['compound'] = compound
    data['positive'] = positive
    data['neutral'] = neutral
    data['negative'] = negative
    return data

def save(data):
    global counter
    # if file does not exist write header
    if not os.path.isfile(output_filename):
       data[columns].to_csv(output_filename,  header='column_names', index=False)
    else: # else it exists so append without writing the header
       data[columns].to_csv(output_filename, mode='a', header=False, index=False)
    print('processing:{} '.format(str(counter)))
    print(round((counter / total_rows) * 100, 2), '%\n')
    counter += chunksize

def process(data):
    tweets = []
    for t in data['tweet']:
        tweets.append(clean_tweet_text(str(t)))
    data.drop(['tweet'], axis = 1, inplace = True)
    data['tweet'] = tweets
    # data['datetime'] = data['date'] + ' ' + data['time']
    return data

tweets = []
with open(input_filename) as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        tweets.append(row)

df = pd.DataFrame(tweets)
df.columns = df.iloc[0]
df = df[1:]

data = process(df)
data = sentiment(data)
save(data)

# for chunk in pd.read_csv(input_filename, engine='python', chunksize=chunksize, encoding='utf-8-sig'):
#     data = process(chunk)
#     data = sentiment(data)
#     save(data)
