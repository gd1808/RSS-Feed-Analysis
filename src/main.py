import re
import uvicorn
from fastapi import FastAPI
from parser.parser import Parser

'''
# Helpful Links
# https://www.semrush.com/blog/url-parameters/
# https://stackoverflow.com/questions/1737935/what-is-the-recommended-way-to-pass-urls-as-url-parameters
# https://www.urlencoder.io/python/
# https://docs.python.org/3/library/urllib.parse.html#urllib.parse.urlencode
# https://www.urldecoder.io/python/
# https://www.urlencoder.org/
# https://fastapi.tiangolo.com/tutorial/query-params-str-validations/
# https://stackoverflow.com/questions/64019054/fastapi-service-results-in-404-when-service-is-started-using-uvicorn-run
# 
# Reminder: the command to install requirements is pip install -r requirements.txt
# and to generate requirements.txt file use pip freeze > requirements.txt
'''

app = FastAPI()

sources = {"NPR": "https://feeds.npr.org/510318/podcast.xml"}


def article_cleaner(articles):
    res = []
    for article in articles:
        clean_article = article
        clean_article['summary'] = re.sub(r'<.*?>', ' ', article['summary'])
        clean_article['summary'] = re.sub(r' +', ' ', clean_article['summary'])
        res.append(clean_article)
    return res


@app.get("/")
def read_root():
    res = {"Hello": "World"}
    res.update(sources)
    return res


@app.get("/articles/")
def get_articles_by_url(rss_url: str, method: str | None = None):
    if rss_url is None:
        return {"error": "No URL provided"}
    # rss_url = par.unquote(rss_url)
    parser = Parser(rss_url)
    feed = parser.fetch_rss_feed()
    articles = feed.entries
    clean_articles = article_cleaner(articles)
    if method is not None or '':
        # TODO: more stuff
        clean_articles = parser.analyze_articles(clean_articles, method)
    return clean_articles


# https%3A%2F%2Ffeeds.npr.org%2F510318%2Fpodcast.xml
@app.get("/articles/{source}")
def get_articles_by_source(source: str, method: str | None = None):
    rss_url = None
    if source in sources.keys():
        rss_url = sources[source]
    if rss_url is None:
        return {"error": "Invalid source provided"}
    # rss_url = par.unquote(rss_url)
    parser = Parser(rss_url)
    feed = parser.fetch_rss_feed()
    articles = feed.entries
    clean_articles = article_cleaner(articles)
    if method is not None or '':
        # TODO: more stuff
        clean_articles = parser.analyze_articles(clean_articles, method)
    return clean_articles


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)

    '''
    # # Pre-API Testing
    #
    # # Imports for test
    # import sys
    # import urllib.parse as par
    # import pandas as pd
    # from tqdm.notebook import tqdm
    # from nltk.sentiment import SentimentIntensityAnalyzer
    # 
    # rss_url = sys.argv[1] if len(sys.argv) > 1 else "https://feeds.npr.org/510318/podcast.xml"  # param to be passed
    # method = sys.argv[2] if len(sys.argv) > 2 else 'HuggingFace'  # param to be passed
    #
    # print(f"RSS: {rss_url}")
    # print(f"Method: {method}\n")
    #
    # parser = Parser(rss_url)
    #
    # feed = parser.fetch_rss_feed()
    #
    # print(feed.entries[0].keys())
    # print(f'\n\n{feed.entries[0]}')
    # print(f"\n\n{feed.entries[0]['content']}")
    # print(f"\n\n{feed.entries[0]['summary']}\n\n")
    #
    # # clean_summary = re.sub(r'<.*?>', ' ', feed.entries[1]['summary'])  # remove html tags
    # # clean_summary = re.sub(r' +', ' ', clean_summary)  # remove extra spaces
    # # print(clean_summary)
    #
    # # Extract articles from the feed
    # articles = feed.entries
    #
    # # clean the article summaries for analysis... Note that unclean summary is in ['summary_detail']['value']
    # clean_articles = []
    # for article in articles:
    #     clean_article = article
    #     clean_article['summary'] = re.sub(r'<.*?>', ' ', article['summary'])
    #     clean_article['summary'] = re.sub(r' +', ' ', clean_article['summary'])
    #     clean_articles.append(clean_article)
    #
    # if method == '':
    #     pass
    # else:
    #     clean_articles = parser.analyze_articles(clean_articles, method)
    #
    # for article in clean_articles:
    #     print(f'{article}\n')
    #     break
    #
    # print('\n\n\nSentiment DataFrame Print')
    # parser.print_sentiment()
    '''
