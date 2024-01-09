# Roberta Pretrained Model
# Usa a model trained of a large corpus of data
# Transformer-based deeplearning model accounts for the words but also the context related to other words
#
# Demo Resources:
# https://youtu.be/QpzMWQvxXWk?si=57HdVAKANXjuFq1o
# https://www.kaggle.com/code/robikscube/sentiment-analysis-python-youtube-tutorial/notebook
# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm  # progress bar

plt.style.use('ggplot')

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

df = pd.read_csv('C:/Users/Gabriel Diaz Soriano/Documents/Personal Projects/RSS-Feed-Analysis/notes/input/Reviews.csv')
df = df.head(500)
example = df['Text'][50]

# VADER results on example
print(example)
# This oatmeal is not good. Its mushy, soft, I don't like it. Quaker Oats is the way to go.
sia = SentimentIntensityAnalyzer()
print(sia.polarity_scores(example))
# {'neg': 0.22, 'neu': 0.78, 'pos': 0.0, 'compound': -0.5448}


# Run for Roberta Model
encoded_text = tokenizer(example, return_tensors="pt")
print(f'\n{encoded_text}\n')
# {'input_ids': tensor([[    0,   713,  1021, 38615,    16,    45,   205,     4,  3139, 39589,
#            219,     6,  3793,     6,    38,   218,    75,   101,    24,     4,
#           3232,  4218,   384,  2923,    16,     5,   169,     7,   213,     4,
#              2]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#          1, 1, 1, 1, 1, 1, 1]])}

output = model(**encoded_text)
print(f'{output}\n')
# SequenceClassifierOutput(loss=None,
#   logits=tensor([[ 3.1436, -0.7107, -2.6559]],
#   grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)

scores = output[0][0].detach().numpy()  # take output which is a tensor rn and make it a numpy to store locally
scores = softmax(scores)
print(f'{scores}\n')  # format: [negative, neutral, positive]
# [0.97635514 0.02068748 0.00295737]

scores_dict = {
    'roberta_neg': scores[0],
    'roberta_neu': scores[1],
    'roberta_pos': scores[2],
}
print(f'{scores_dict}\n')


# {'roberta_neg': 0.97635514, 'roberta_neu': 0.020687476, 'roberta_pos': 0.0029573706}

# Get polarity of whole dataset using Roberta
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors="pt")
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()  # take output which is a tensor rn and make it a numpy to store locally
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2],
    }
    return scores_dict


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myId = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for k, v in vader_result.items():
            vader_result_rename[f'vader_{k}'] = v
        roberta_result = polarity_scores_roberta(text)
        both = vader_result_rename | roberta_result
        res[myId] = both
        # print(both)
        # break
        # {'vader_neg': 0.0, 'vader_neu': 0.695, 'vader_pos': 0.305, 'vader_compound': 0.9441,
        # 'roberta_neg': 0.009624252, 'roberta_neu': 0.049980428, 'roberta_pos': 0.9403953}
    except RuntimeError:
        print(f'Broke for id {myId}')

# Note: slowness is b/c model was run on cpu, not gpu... gpu optimized

# combine dictionaries <3.10
# print({**vader_result, **roberta_result})

# combine dictionaries 3.10+
# print(vader_result | roberta_result)

results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')
print(f'\n{results_df.head()}\n')
#    Id  vader_neg  vader_neu  vader_pos  ...  Score        Time                Summary                                               Text
# 0   1      0.000      0.695      0.305  ...      5  1303862400  Good Quality Dog Food  I have bought several of the Vitality canned d...
# 1   2      0.138      0.862      0.000  ...      1  1346976000      Not as Advertised  Product arrived labeled as Jumbo Salted Peanut...
# 2   3      0.091      0.754      0.155  ...      4  1219017600  "Delight" says it all  This is a confection that has been around a fe...
# 3   4      0.000      1.000      0.000  ...      2  1307923200         Cough Medicine  If you are looking for the secret ingredient i...
# 4   5      0.000      0.552      0.448  ...      5  1350777600            Great taffy  Great taffy at a great price.  There was a wid...
#
# [5 rows x 17 columns]

# Compare Scores between models and Compare them
sns.pairplot(
    data=results_df,
    vars=['vader_neg', 'vader_neu', 'vader_pos',
          'roberta_neg', 'roberta_neu', 'roberta_pos'],
    hue='Score',
    palette='tab10'
)
plt.show()

# Examples where the model scoring and review score differ the most

# Printing text that the Roberta model determined to be positive but that was
#   given a 1-star review by the reviewer
print(results_df
      .query('Score == 1')
      .sort_values('roberta_pos', ascending=False)['Text']
      .values[0])
# I felt energized within five minutes, but it lasted for about 45 minutes.
# I paid $3.99 for this drink. I could have just drunk a cup of coffee and saved my money.

print('\n')

# Printing text that the VADER model determined to be positive but that was
#   given a 1-star review by the reviewer
print(results_df
      .query('Score == 1')
      .sort_values('vader_pos', ascending=False)['Text']
      .values[0])
# So we cancelled the order.  It was cancelled without any problem.  That is a positive note...

print('\n')

# Negative sentiment 5-star review, Roberta
print(results_df
      .query('Score == 5')
      .sort_values('roberta_neg', ascending=False)['Text']
      .values[0])
# this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault

print('\n')

# Negative sentiment 5-star review, VADER
print(results_df
      .query('Score == 5')
      .sort_values('vader_neg', ascending=False)['Text']
      .values[0])
# this was sooooo deliscious but too bad i ate em too fast and gained 2 pds! my fault

print('\n')

# Hugging Face Transformers Pipeline Sentiment Analysis
from transformers import pipeline

sent_pipeline = pipeline('sentiment-analysis')

print(f"{sent_pipeline('I love sentiment analysis!')}\n")
# [{'label': 'POSITIVE', 'score': 0.9997853636741638}]

print(sent_pipeline("I don't like sand. "
                    "It's coarse and rough and irritating and it gets everywhere. "
                    "Not like here. Here everything is soft and smooth."))
# [{'label': 'NEGATIVE', 'score': 0.9458056092262268}]

# C:\Users\Gabriel Diaz Soriano\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.10_qbz5n2kfra8p0\LocalCache\local-packages\Python310
# TODO: Strategy pattern to choose to use VADER, Roberta, or Hugging Face
# TODO: API ingest endpoint that defaults to getting RSS feed and can be passed the different commands
# TODO: Basic Frontend to display analysis info and RSS feed articles
#
# TODO: Display seaborn graphs on frontend of the sentiments for each model as well as the pair plots for comparison
# TODO: Utilize PyTorch to train own model for sentiment analysis and add it to command pattern and endpoint
