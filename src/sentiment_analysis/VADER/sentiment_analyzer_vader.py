import matplotlib.pyplot as plt
import mpld3
import nltk
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
from nltk.sentiment import SentimentIntensityAnalyzer
from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer


class VaderSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, articles) -> None:
        nltk.download('all')
        self.articles = articles
        self.df = pd.DataFrame(self.articles)
        # self.polarity_scores = []
        self.sia = SentimentIntensityAnalyzer()
        self.plot = {}  # this will be json of matplotlib

    def sentiment_analysis(self):
        # list_arr = df.to_dict('records')

        # make Id column in articles
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={"index": "Id"})
        self.df['Id'] = self.df.index + 1
        res = {}
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            summary = row['summary']
            Id = row['Id']
            res[Id] = self.sia.polarity_scores(summary)

        vaders = pd.DataFrame(res).T
        # make Id column in sentiment dict
        vaders = vaders.reset_index().rename(columns={'index': 'Id'})
        vaders = vaders.merge(self.df, how='left')

        self.articles = vaders.to_dict('records')
        for article in self.articles:
            del article['image']
        return self.articles
        # for article in self.articles:
        #     self.polarity_scores.append(self.sia.polarity_scores(article['summary']))
        # pass

    def print_sentiment_analysis(self) -> None:
        print(self.df)
