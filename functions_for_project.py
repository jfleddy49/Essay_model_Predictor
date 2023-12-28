import pandas as pd
import numpy as np
import matplotlib as plt
import gensim
import requests
import string
import re

from sklearn.manifold import TSNE

import nltk
nltk.download('punkt')
from nltk import word_tokenize, tokenize, FreqDist
from nltk.tokenize import sent_tokenize

from statistics import variance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from sklearn.neural_network import MLPClassifier
import textstat
import pickle




pattern = r'''(?x)
(?:[A-Z]\.)+
|\w+(?:[-']\w+)*
|\$?\d+(?:\.\d+)?
|\.\.\.
|[.,;"'?()-_`]
'''
def add_metrics(text):
  token = sent_tokenize(text)
  words = word_tokenize(text)
  unique = set(words)
  freq = FreqDist(words)
  m1 = sum(1 for freq in freq.values() if freq == 1)
  m2 = sum(freq ** 2 for freq in freq.values())
  yules = 0
  if m1 != 0:
    yules = 10000 * (m2 - m1) / (m1 * m1)
  sens = []
  total = 0
  for i in token:
    sentence = len(word_tokenize(i))
    total += sentence
    sens += [sentence]
  if len(sens) < 2:
    return total/len(token), 0, len(unique)/len(words), yules
  else:
    return total/len(token), variance(sens), len(unique)/len(words), yules

def remove_abstract(df, col, pattern):
  rem = []
  for i in range(len(df)):
    tokenized_raw = ' '.join(nltk.regexp_tokenize(df[col][i], pattern))
    tokenized_raw = tokenize.sent_tokenize(tokenized_raw)

    if len(tokenized_raw) == 0 or tokenized_raw[0] in string.punctuation:
      df = df.drop(i)
      continue

    rem.append(re.sub('(\\n)+', ' ', df[col][i]))

  df[col] = rem
  return df.reset_index().drop(columns = 'index')

def get_unique_info(df, col, pattern):
  unique = []
  length = []
  vec = []

  for i in range(len(df)):
    tokenized_raw = ' '.join(nltk.regexp_tokenize(df[col][i], pattern))
    tokenized_raw = tokenize.sent_tokenize(tokenized_raw)

    nopunct = []

    for sent in tokenized_raw:
      a = [w for w in sent.split() if w not in string.punctuation]
      nopunct.append(' '.join(a))

    tok_corp = [nltk.word_tokenize(sent) for sent in nopunct]

    model = gensim.models.Word2Vec(tok_corp, min_count = 1, vector_size = 300, window = 5)

    unique_words = list(set([item for sublist in tok_corp for item in sublist]))

    vector_list = model.wv[unique_words]

    unique.append(unique_words)
    length.append(len(unique_words))
    vec.append(vector_list)

  df[col + '_vector_list'] = vec
  df[col + '_unique_words'] = unique
  df[col + '_len_unique_words'] = length

def add_feat(df, col):
  dims = np.zeros(shape = (len(df[col][0]), len(df)))

  for i in range(len(df[col])):
    for j in range(len(df[col][i])):
      dims[j][i] = df[col][i][j]

  for dim in range(len(dims)):
    temp = df.copy()
    temp[col + '_dim_' + str(dim)] = dims[dim]
    df = temp
  return df

def df_maker(text):
  holder = pd.DataFrame(pd.Series(text))
  holder = holder.rename(columns={0 : 'text'})
  return holder

def make_matrix(df, vectorizer):
  df['avg'], df['var'], df['ttr'], df['yules'] = zip(*df['text'].apply(add_metrics))
  df = remove_abstract(df, 'text', pattern)
  get_unique_info(df, 'text', pattern)
  df['one_vec'] = df['text_vector_list'].apply(sum)
  df = add_feat(df, 'one_vec')
  df = df.drop(['one_vec', 'text_vector_list', 'text_unique_words', 'text_len_unique_words'], axis = 1)
  matrix = vectorizer.transform(df['text'])
  for i in range(1, df.shape[1]):
    matrix = hstack([matrix, csr_matrix(df.iloc[:, i]).reshape(-1,1)])
  return matrix


