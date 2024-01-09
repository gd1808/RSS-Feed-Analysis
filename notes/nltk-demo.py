# Demo Resources:
# https://www.nltk.org/api/nltk.tokenize.punkt.html
# https://www.nltk.org/
# https://www.nltk.org/book/
import nltk

# Part One Downloads
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Part Two Downloads
nltk.download('maxent_ne_chunker')
nltk.download('words')

sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."

# Part One
tokens = nltk.word_tokenize(sentence)
print(tokens)
# tokens = ['At', 'eight', "o'clock", 'on', 'Thursday', 'morning',
#   'Arthur', 'did', "n't", 'feel', 'very', 'good', '.']

tagged = nltk.pos_tag(tokens)
# print('\n{tagged}\n'.format(tagged=tagged)) # cool fstring alternative; con: too verbose
print(f'\n{tagged}\n')
print(tagged[0:6])

# Part Two
entities = nltk.ne_chunk(tagged)  # need to install numpy for the chunker
entities.pprint()
