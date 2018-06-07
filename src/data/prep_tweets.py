import pandas as pd
import re
from vaderSentiment import vaderSentiment

input_filename = "../assets/data.csv"
output_filename = "../output/data_clean.csv"
columns = ['datetime','tweet','followers_count', 'compound', 'positive', 'neutral', 'negative']

counter = 0
chunksize = 10000

def clean_tweet_text(text):
    # 1. Lower case
    processed_text = text.lower()
    # 2. Remove mentions
    processed_text = re.sub(r"@[\w\S]+", '', processed_text)
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
        t = str(t)
        vs = analyzer.polarity_scores(t)
        compound.append(vs['compound'])
        positive.append(vs['pos'])
        neutral.append(vs['neu'])
        negative.append(vs['neg'])

    data['compound'] = pd.Series(compound)
    data['positive'] = pd.Series(positive)
    data['neutral'] = pd.Series(neutral)
    data['negative'] = pd.Series(negative)       
    return data

def save(data):
    global counter
    data[columns].to_csv('outputs/' + output_filename, mode='a', encoding='utf-8-sig', header=True, index=False)
    print('processing:{} '.format(str(counter)))
    counter += chunksize
    

def process(data):
    tweets = []  
    for t in data['tweet']:
        tweets.append(clean_tweet_text(str(t)))
    data['tweet'] = pd.Series(tweets)
    data['datetime'] = data['date'] + " " + data['time']
    data = sentiment(data)
    save(data)

df = pd.read_csv(input_filename, engine='python')
process(df)


