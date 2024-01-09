import pandas as pd
import seaborn as sns
import mpld3
import matplotlib.pyplot as plt
from transformers import pipeline
from tqdm.notebook import tqdm
from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer


class HuggingFaceSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, articles) -> None:
        self.articles = articles
        self.df = pd.DataFrame(self.articles)
        self.plot = {}  # this will be json of matplotlib
        self.sent_pipeline = pipeline('sentiment-analysis')

    def sentiment_analysis(self):
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={"index": "Id"})
        self.df['Id'] = self.df.index + 1
        res = {}
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            text = row['summary']
            myId = row['Id']
            res[myId] = self.sent_pipeline(text)

        hugging_face = pd.DataFrame(res).T
        hugging_face = hugging_face.reset_index().rename(columns={'index': 'Id'})
        hugging_face = hugging_face.merge(self.df, how='left')

        self.articles = hugging_face.to_dict('records')
        for article in self.articles:
            del article['image']
        return self.articles

    def print_sentiment_analysis(self) -> None:
        print(self.df)
