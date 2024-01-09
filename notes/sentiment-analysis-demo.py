# Stale first attempt with deprecated code
import nltk
import nltk.sentiment.sentiment_analyzer


# Analysing for single words
def one_word():
    positive_words = ['good', 'progress', 'luck']
    text = 'Hard Work brings progress and good luck.'.split()
    analysis = nltk.sentiment.util.extract_unigram_feats(text, positive_words)
    print(' ** Sentiment with one word **\n')
    print(analysis)


# Analysing for a pair of words
def paired_words():
    word_sets = [('Regular', 'fit'), ('fit', 'fine')]
    text = 'Regular exercise makes you fit and fine'.split()
    analysis = nltk.sentiment.util.extract_bigram_feats(text, word_sets)
    print('\n*** Sentiment with bigrams ***\n')
    print(analysis)


# Analysing the negation words
def negative_word():
    text = 'Lack of good health can not bring success to students'.split()
    analysis = nltk.sentiment.util.mark_negation(text)
    print('\n**Sentiment with Negative words**\n')
    print(analysis)


one_word()
paired_words()
negative_word()
