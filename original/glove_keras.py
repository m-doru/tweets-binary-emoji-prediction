import os
import logging
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Convolution1D, LSTM
from keras.models import Model, Sequential

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 60000 
EMBEDDING_DIM = 200

BASE_DIR = '../data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')

PRETRAINED_EMBEDDINGS_FILE = os.path.join(GLOVE_DIR, 'glove.twitter.27B.{}d.txt'.format(EMBEDDING_DIM))

def read_pretrained_glove_embeddings(path):
  embeddings_index = {}

  with open(path) as f:
    for line in f:
      values = line.rstrip().split()
      word = values[0]
      coefs = np.asarray(values[1:], dtype='float32')
      embeddings_index[word] = coefs

  return embeddings_index

def transform_tweets_to_sequences(X, pad=True):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    logging.info('Found %s unique tokens.' % len(word_index))

    if pad:
      sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return sequences, word_index, tokenizer


def construct_embedding_matrix(word_index, embeddings_index):
  num_words = min(MAX_NB_WORDS, len(word_index))

  embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
  for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

  return embedding_matrix


def pretrained_glove_keras_model_conv(X_train_tweets):
  np.random.seed(777)
  embeddings_index = read_pretrained_glove_embeddings(PRETRAINED_EMBEDDINGS_FILE)

  X_train, word_index, tokenizer = transform_tweets_to_sequences(X_train_tweets, True)

  print('Shape of data tensor:', X_train.shape)

  embedding_matrix = construct_embedding_matrix(word_index, embeddings_index)

  model = Sequential()
  model.add(Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], input_length=X_train.shape[1], weights=[embedding_matrix]))
  model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
  model.add(MaxPooling1D(pool_length=2))
  model.add(Flatten())
  model.add(Dense(250, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model, X_train, tokenizer, MAX_SEQUENCE_LENGTH

def pretrained_glove_keras_model_lstm(X_train_tweets):
  np.random.seed(777)
  embeddings_index = read_pretrained_glove_embeddings(PRETRAINED_EMBEDDINGS_FILE)

  X_train, word_index, tokenizer = transform_tweets_to_sequences(X_train_tweets, True)

  print('Shape of data tensor:', X_train.shape)

  embedding_matrix = construct_embedding_matrix(word_index, embeddings_index)

  model = Sequential()
  model.add(Embedding(embedding_matrix.shape[0], 
                      embedding_matrix.shape[1], 
                      weights=[embedding_matrix],
                      input_length=MAX_SEQUENCE_LENGTH,
                      trainable=False))
                      
  model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
  model.add(MaxPooling1D(pool_length=2))
  model.add(LSTM(100))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  return model, X_train, tokenizer, MAX_SEQUENCE_LENGTH
