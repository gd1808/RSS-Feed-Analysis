from sentiment_analysis.sentiment_analyzer import SentimentAnalyzer


class Context:
    def __init__(self, strategy: SentimentAnalyzer) -> None:
        self._strategy = strategy

    @property
    def strategy(self) -> SentimentAnalyzer:
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: SentimentAnalyzer) -> None:
        self._strategy = strategy

    def execute_strategy(self):
        result = self._strategy.sentiment_analysis()
        return result
