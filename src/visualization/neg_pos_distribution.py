import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# Plot Tweet polarity distribution over period
data_filename = 'data/interim/tweets_aggregated.csv'
print('\nLoading prep file: ', data_filename)
df = pd.read_csv(data_filename)
print('File has been loaded \n')

timeframe = '60min'
feature = 'compound'
df.index = pd.DatetimeIndex(df['datetime'])
print(df.loc[df[feature] > 0].shape)
print(df.loc[df[feature] < 0].shape)
print(df.loc[df[feature] == 0].shape)
positive = df.loc[df[feature] > 0].resample(timeframe).agg({'datetime': 'last', feature: 'mean'})
negative = df.loc[df[feature] < 0].resample(timeframe).agg({'datetime': 'last', feature: 'mean'})
negative[feature] = negative[feature].abs()

ax1 = fig.add_subplot(111)
ax1.scatter(positive.index, positive[feature], c='g', alpha=0.4)
ax1.scatter(positive.index, negative[feature], c='r', alpha=0.4)
ax1.set_xlim([positive.index[0], positive.index[-1]])

plt.xlabel('Date')
# plt.ylabel('Compound score')
plt.title('Tweets polarity')
# plt.savefig('../outputs/neg_pos_distribution_' + timeframe + '.png', bbox_inches='tight')
plt.show()
plt.clf()

# Plot followers compound over date
feature = 'followers_compound'
df = df.resample(timeframe).agg({'datetime': 'last',  feature: 'mean'})
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(df.index, df[feature], c='g', alpha=0.4)
ax1.set_xlim([df.index[0], df.index[-1]])

plt.xlabel('Date')
plt.show()
plt.clf()
