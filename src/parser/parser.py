import feedparser
from sentiment_analysis.VADER.sentiment_analyzer_vader import VaderSentimentAnalyzer
from sentiment_analysis.Roberta.sentiment_analyzer_roberta import RobertaSentimentAnalyzer
from sentiment_analysis.HuggingFace.sentiment_analyzer_huggingface import HuggingFaceSentimentAnalyzer
from sentiment_analysis.context import Context


class Parser:
    def __init__(self, url) -> None:
        self.url = url
        self.sentiment = None

    def fetch_rss_feed(self) -> feedparser.FeedParserDict:
        # Use feedparser to fetch and parse the RSS feed
        parsed_feed = feedparser.parse(self.url)
        return parsed_feed

    def analyze_articles(self, articles, method):
        match method:
            case 'VADER':
                sentiment_analyzer = Context(VaderSentimentAnalyzer(articles))
                result = sentiment_analyzer.execute_strategy()
                self.sentiment = sentiment_analyzer.strategy
                return result
            case 'Roberta':
                sentiment_analyzer = Context(RobertaSentimentAnalyzer(articles))
                result = sentiment_analyzer.execute_strategy()
                self.sentiment = sentiment_analyzer.strategy
                return result
            case 'HuggingFace':
                sentiment_analyzer = Context(HuggingFaceSentimentAnalyzer(articles))
                result = sentiment_analyzer.execute_strategy()
                self.sentiment = sentiment_analyzer.strategy
                return result
            case _:  # _ is a wildcard, indicates default case
                return None

    def print_sentiment(self) -> None:
        if self.sentiment is not None:
            self.sentiment.print_sentiment_analysis()
