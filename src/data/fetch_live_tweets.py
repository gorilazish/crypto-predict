import tweepy
import os
import csv

keywords = ['btc', 'bitcoin', 'btc/usd', 'crypto', 'cryptocurrency', 'ico']
columns = ['datetime', 'tweet', 'followers_count']
TWITTER_APP_KEY = 'lwgx2D7458Vg0xBjDtsBDGbVz'
TWITTER_APP_SECRET = 'BvMW6epZE1IAIVlIBtjTkEhkoUa1xNv1LPagfrQICLmnPzwJ9n'
TWITTER_KEY = '933111809770373120-donuuXGOdJMrZyJ8wUDxJPv2B4KIaTd'
TWITTER_SECRET = 'UZSHq98R5rg4mPkoMHswuAePFYBEc1Pua1VZ8d45lgdJw'
auth = tweepy.OAuthHandler(TWITTER_APP_KEY, TWITTER_APP_SECRET)
auth.set_access_token(TWITTER_KEY, TWITTER_SECRET)
api = tweepy.API(auth)

class StreamListener(tweepy.StreamListener):

  def on_status(self, status):  
    if status.retweeted:
      return
      
    output_file = 'data/raw/live_tweets.csv'
    if not os.path.isfile(output_file):
      with open(output_file, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(columns)

    with open(output_file, 'a') as csv_file:
      writer = csv.writer(csv_file)
      followers_count = status.user.followers_count + status.user.friends_count
      writer.writerow([status.created_at, status.text, followers_count])

  def on_error(self, status_code):
    if status_code == 420:
      return False

stream_listener = StreamListener()
stream = tweepy.Stream(auth=api.auth, listener=stream_listener)
stream.filter(track=keywords)