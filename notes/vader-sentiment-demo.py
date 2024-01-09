# VADER = Valence Aware Dictionary and Sentiment Reasoner
#   uses the "bag of words" approach - takes all the words in a
#       sentence and gives them values of positive, negative, or neutral
#       then adds them up to indicate whether the complete statement is positive,
#       negative, or neutral
#   does not account for relationships between words
#   stop words are removed - words like "and" and "the"

# Note: Need to run pip install jupyter and pip install ipywidgets for tqdm
# Seaborn color palette resource: https://www.practicalpythonfordatascience.com/ap_seaborn_palette#cmrmap

import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import os

plt.style.use('ggplot')

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm  # this is for a progress bar

nltk.download('vader_lexicon')

print('\n\n')
# print(os.getcwd())
df = pd.read_csv('C:/Users/Gabriel Diaz Soriano/Documents/Personal Projects/RSS-Feed-Analysis/notes/input/Reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)
# (568454, 10)
# (500, 10)

print(df.head())
#    Id   ProductId          UserId  ...        Time                Summary                                               Text
# 0   1  B001E4KFG0  A3SGXH7AUHU8GW  ...  1303862400  Good Quality Dog Food  I have bought several of the Vitality canned d...
# 1   2  B00813GRG4  A1D87F6ZCVE5NK  ...  1346976000      Not as Advertised  Product arrived labeled as Jumbo Salted Peanut...
# 2   3  B000LQOCH0   ABXLMWJIXXAIN  ...  1219017600  "Delight" says it all  This is a confection that has been around a fe...
# 3   4  B000UA0QIQ  A395BORC6FGVXV  ...  1307923200         Cough Medicine  If you are looking for the secret ingredient i...
# 4   5  B006K2ZZ7K  A1UQRSCLF8GW1T  ...  1350777600            Great taffy  Great taffy at a great price.  There was a wid...
#
# [5 rows x 10 columns]

# Review Star Bar Plot
# ax = (df['Score'].value_counts().sort_index()
#       .plot(kind='bar',
#             title='Count of Reviews by Stars',
#             figsize=(10, 5)))
#
# ax.set_xlabel('Review Stars')
# plt.show()

example = df['Text'][50]
print(f'\n{example}')
# This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.
tokens = nltk.word_tokenize(example)
print(tokens[:10])
# ['This', 'oatmeal', 'is', 'not', 'good', '.', 'Its', 'mushy', ',', 'soft']
tagged = nltk.pos_tag(tokens)
print(tagged[:10])
# [('This', 'DT'), ('oatmeal', 'NN'), ('is', 'VBZ'), ('not', 'RB'), ('good', 'JJ'),
#   ('.', '.'), ('Its', 'PRP$'), ('mushy', 'NN'), (',', ','), ('soft', 'JJ')]

print('\n\n')
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores('I am so happy!'))
# {'neg': 0.0, 'neu': 0.318, 'pos': 0.682, 'compound': 0.6468}
# {'neg': from 0 to 1, 'neu': from 0 to 1, 'pos': from 0 to 1, 'compound': from -1 to 1}

print(sia.polarity_scores('This is the worst thing ever.'))
# {'neg': 0.451, 'neu': 0.549, 'pos': 0.0, 'compound': -0.6249}

print(sia.polarity_scores(example))
# {'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}
print('\n\n')

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myId = row['Id']
    res[myId] = sia.polarity_scores(text)
# print(res)

vaders = pd.DataFrame(res).T  # .T flips everything horizontally in DataFrame
print(f'\n{vaders}\n')
#        neg    neu    pos  compound
# 1    0.000  0.695  0.305    0.9441
# 2    0.138  0.862  0.000   -0.5664
# 3    0.091  0.754  0.155    0.8265
# 4    0.000  1.000  0.000    0.0000
# 5    0.000  0.552  0.448    0.9468
# ..     ...    ...    ...       ...
# 496  0.000  0.554  0.446    0.9725
# 497  0.059  0.799  0.142    0.7833
# 498  0.025  0.762  0.212    0.9848
# 499  0.041  0.904  0.055    0.1280
# 500  0.000  0.678  0.322    0.9811
#
# [500 rows x 4 columns]

vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')  # merge left onto original data frame
print(f'{vaders}\n')
#       Id    neg    neu    pos  ...  Score        Time                          Summary                                               Text
# 0      1  0.000  0.695  0.305  ...      5  1303862400            Good Quality Dog Food  I have bought several of the Vitality canned d...
# 1      2  0.138  0.862  0.000  ...      1  1346976000                Not as Advertised  Product arrived labeled as Jumbo Salted Peanut...
# 2      3  0.091  0.754  0.155  ...      4  1219017600            "Delight" says it all  This is a confection that has been around a fe...
# 3      4  0.000  1.000  0.000  ...      2  1307923200                   Cough Medicine  If you are looking for the secret ingredient i...
# 4      5  0.000  0.552  0.448  ...      5  1350777600                      Great taffy  Great taffy at a great price.  There was a wid...
# ..   ...    ...    ...    ...  ...    ...         ...                              ...                                                ...
# 495  496  0.000  0.554  0.446  ...      5  1201392000                    amazing chips  i rarely eat chips but i saw these and tried t...
# 496  497  0.059  0.799  0.142  ...      5  1196726400                   Best Chip Ever  This is easily the best potato chip that I hav...
# 497  498  0.025  0.762  0.212  ...      4  1186617600  Tangy, spicy, and sweet- oh my!  Kettle Chips Spicy Thai potato chips have the ...
# 498  499  0.041  0.904  0.055  ...      4  1184198400        An indulgence with a bite  Okay, I should not eat potato chips, nor shoul...
# 499  500  0.000  0.678  0.322  ...      5  1183420800                The best I've had  I don't write very many reviews but I have to ...
#
# [500 rows x 14 columns]

# Now we have sentiment score and metadata
print(f'{vaders.head()}\n\n')
#    Id    neg    neu    pos  ...  Score        Time                Summary                                               Text
# 0   1  0.000  0.695  0.305  ...      5  1303862400  Good Quality Dog Food  I have bought several of the Vitality canned d...
# 1   2  0.138  0.862  0.000  ...      1  1346976000      Not as Advertised  Product arrived labeled as Jumbo Salted Peanut...
# 2   3  0.091  0.754  0.155  ...      4  1219017600  "Delight" says it all  This is a confection that has been around a fe...
# 3   4  0.000  1.000  0.000  ...      2  1307923200         Cough Medicine  If you are looking for the secret ingredient i...
# 4   5  0.000  0.552  0.448  ...      5  1350777600            Great taffy  Great taffy at a great price.  There was a wid...
#
# [5 rows x 14 columns]

# Plot Vader Results by Compound Score
# Assumptions: 5-star would tend to be more positive text than 1-star
# ax = sns.barplot(data=vaders, x='Score', y='compound')
# ax.set_title('Compound Score by Amazon Star Review')
# plt.show()

# Plot Vader Results by Individual Score
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', palette='CMRmap', ax=axs[0])  # added plot because newer
sns.barplot(data=vaders, x='Score', y='neu', palette='CMRmap', ax=axs[1])  # version of has the same color
sns.barplot(data=vaders, x='Score', y='neg', palette='CMRmap', ax=axs[2])  # for each bar in plot
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()  # fix overlapping of y-axis values
plt.show()
