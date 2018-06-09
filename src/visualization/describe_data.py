import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

# Load dataset
data_filename = 'data/interim/tweets_aggregated.csv'
print('Loading prep file: ', data_filename)
df = pd.read_csv(data_filename, usecols=['positive_mean'])
print('File has been loaded \n')
# df = df.drop(['datetime'], axis=1)

print(df.describe())
print()

print(pd.DataFrame(preprocessing.scale(df)).describe())
