import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

scaler = MinMaxScaler()
timeframe = '15min'

# Load dataset
data_filename = 'data/processed/tweets_prices_'+ timeframe +'.csv'
df = pd.read_csv(data_filename, index_col=0)
df.index = pd.DatetimeIndex(df.index)
df[['close', 'tweet_count']] = scaler.fit_transform(df[['close', 'tweet_count']])

fig, ax = plt.subplots()

# Selected period of time
start_date = '27/04/2018 00:00:00'
end_date = '28/05/2018 01:00:00'
df = df.loc[(df.index > start_date) & (df.index <= end_date)]
print(df.shape)
# Remove time of day seasonality
day = int((24 * 60) / 15)
diff = list()
i = 0
for count in df['tweet_count']:
  value = df['tweet_count'][i] -  df['tweet_count'][i-day]
  diff.append(value) 
  i += 1

plt.plot(df.index, df['close'], label='Price')
plt.plot(df.index, diff, c='g', label='Tweets volume')
plt.xlabel('Date')
plt.legend()

ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.xticks(rotation=45)

plt.savefig('reports/price_tweet_volume_'+ timeframe +'.png', bbox_inches='tight', dpi='figure')
plt.clf()
