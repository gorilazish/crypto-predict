from wordcloud import WordCloud, STOPWORDS
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)

def generate_wordcloud (text):
  # Generate a word cloud image
  print('Generating wordcloud')
  wordcloud = WordCloud(max_words=2000, stopwords=stopwords, width=1600, height=800, max_font_size=200).generate(text)
  print('Wordcloud generated \n')
  fig = plt.figure(figsize=(12,10))
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.axis("off")
  print('Saving Wordcloud png')
  fig.savefig('./outputs/cm.png', bbox_inches='tight')