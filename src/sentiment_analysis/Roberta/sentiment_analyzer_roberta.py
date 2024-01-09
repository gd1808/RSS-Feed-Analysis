import pandas as pd
import seaborn as sns
import mpld3
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm.notebook import tqdm
from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer


class RobertaSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self, articles) -> None:
        self.articles = articles
        self.df = pd.DataFrame(self.articles)
        self.plot = {}  # this will be json of matplotlib
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def __polarity_scores_roberta(self, text):
        encoded_text = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_text)
        scores = output[0][0].detach().numpy()  # take output which is a tensor rn and make it a numpy to store locally
        scores = softmax(scores)
        scores_dict = {
            'neg': scores[0],
            'neu': scores[1],
            'pos': scores[2],
        }
        return scores_dict

    def sentiment_analysis(self):
        self.df = self.df.reset_index()
        self.df = self.df.rename(columns={"index": "Id"})
        self.df['Id'] = self.df.index + 1
        res = {}
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            try:
                text = row['summary']
                myId = row['Id']
                res[myId] = self.__polarity_scores_roberta(text)
            except RuntimeError:
                print(f'Broke for id {myId}')
        roberta = pd.DataFrame(res).T
        roberta = roberta.reset_index().rename(columns={'index': 'Id'})
        roberta = roberta.merge(self.df, how='left')

        self.articles = roberta.to_dict('records')
        for article in self.articles:
            del article['image']
        return self.articles

    def print_sentiment_analysis(self) -> None:
        print(self.df)
