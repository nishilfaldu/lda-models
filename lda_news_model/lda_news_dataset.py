# Necessary library imports
import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from gensim import corpora, models
from gensim.models.phrases import Phraser, Phrases
from pprint import pprint

# Download the corpora
nltk.download('omw-1.4')

# Inserting data into a DataFrame
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)

# Processing DataFrame
data_text = data[['headline_text']]
data_text['index'] = data_text.index
documents = data_text

# Stemming
def lemmatize_stemming(text):
  return PorterStemmer().stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Preprocessing dataset
def preprocess(text):
  result = []
  for token in gensim.utils.simple_preprocess(text):
    if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
      result.append(lemmatize_stemming(token))
  return result

# Printing a sample
doc_sample = documents[documents['index'] == 4310].values[0][0]
print(doc_sample)

# Printing a tokenized and lemmatized text (checking if it works)
print('original document: ')
words = []
for word in doc_sample.split(' '):
  words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

# More processing of the DataFrame
documents = documents.dropna(subset=['headline_text'])
processed_docs = documents['headline_text'].map(preprocess)

# Creating a dictionary of words
dictionary = gensim.corpora.Dictionary(processed_docs)

# Printing dictionary 
count = 0
for k, v in dictionary.iteritems():
  print(k, v)
  count += 1
  if count > 5:
    break

# Applying filters to dictionary
dictionary.filter_extremes(no_below = 15, no_above = 0.5, keep_n = 100000)

# Generating N-grams
bigram = Phrases(processed_docs, min_count = 1,threshold = 100)
trigrams = Phrases(bigram[processed_docs], min_count = 5, threshold = 100)

# Bag-of-words corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# Creating a TFIDF from the corpus
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

# Printing documents from the generated TFIDF
for doc in corpus_tfidf:
  pprint(doc)
  break

# LDA Model (bag-of-words)
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

# Print topics from the model (bag-of-words) generated above
for idx, topic in lda_model.print_topics(-1):
  print('Topic: {} \nWords: {}'.format(idx, topic))

# LDA Model (TFIDF Model)
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=20, id2word=dictionary, passes=2, workers=4)

# Print topics from the model (TFIDF) generated above
for idx, topic in lda_model_tfidf.print_topics(-1):
  print('Topic: {} \nWords: {}'.format(idx, topic))

# Scoring the topics from the model (bag-of-words) generated above
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
  print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

# Scoring the topics from the model (TFIDF) generated above
for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))

# Testing the model on unseen document 
unseen_document = 'How a Pentagon deal became an identity crisis for Google'
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

