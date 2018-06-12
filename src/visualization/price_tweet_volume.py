import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
timeframe = '15min'

# Load dataset
data_filename = 'data/processed/tweets_prices_'+ timeframe +'.csv'
df = pd.read_csv(data_filename, index_col=0)
df[['close', 'tweet_count']] = scaler.fit_transform(df[['close', 'tweet_count']])

fig, ax = plt.subplots()
df.index = pd.DatetimeIndex(df.index)
# Selected period of time
start_date = '27/04/2018 00:00:00'
end_date = '30/04/2018 00:00:00'
df = df.loc[(df.index > start_date) & (df.index <= end_date)]
plt.plot(df.index, df['close'])
plt.plot(df.index, df['tweet_count'], c='g')
plt.ylabel('Price')
plt.xlabel('Date')

ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.xticks(rotation=45)

plt.savefig('reports/price_tweet_volume_'+ timeframe +'.png', bbox_inches='tight', dpi='figure')
plt.clf()
