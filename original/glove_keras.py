import os
import logging
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Convolution1D
from keras.models import Model

from keras_wrapper import KerasModelWrapper

MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = None
EMBEDDING_DIM = 200
VALIDATION_SPLIT = 0.2

BASE_DIR = '../data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')

POS_FULL_FILEPATH = os.path.join(BASE_DIR, 'train_pos_full.txt')
NEG_FULL_FILEPATH = os.path.join(BASE_DIR, 'train_neg_full.txt')
POS_FILEPATH = os.path.join(BASE_DIR, 'train_pos.txt')
NEG_FILEPATH = os.path.join(BASE_DIR, 'train_neg.txt')

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
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=' ')  # don't delete anything but space
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    logging.info('Found %s unique tokens.' % len(word_index))

    if pad:
      sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return sequences, word_index


def construct_embedding_matrix(word_index, embeddings_index):
  if MAX_NB_WORDS is None:
    num_words = len(word_index) + 1
  else:
    num_words = min(MAX_NB_WORDS, len(word_index)) + 1

  embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
  for word, index in word_index.items():
    if ((MAX_NB_WORDS is None) or index < MAX_NB_WORDS) and (word in embeddings_index):
      embedding_matrix[index] = embeddings_index[word]

  return embedding_matrix


def pretrained_glove_keras_model(X_train_tweets=None):
  if X_train_tweets is None:
    return KerasModelWrapper(None, 'glove_pretrained_{}D_keras_conv1D'.format(EMBEDDING_DIM), 128, 2)
  np.random.seed(777)
  embeddings_index = read_pretrained_glove_embeddings(PRETRAINED_EMBEDDINGS_FILE)

  X_train, word_index = transform_tweets_to_sequences(X_train_tweets, True)

  print('Shape of data tensor:', X_train.shape)

  embedding_matrix = construct_embedding_matrix(word_index, embeddings_index)

  embedding_layer = Embedding(embedding_matrix.shape[0],
                              embedding_matrix.shape[1],
                              weights=[embedding_matrix],
                              input_length=MAX_SEQUENCE_LENGTH,
                              trainable=False)

  # prepare a 1D convnet with global maxpooling
  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences = embedding_layer(sequence_input)
  x = Conv1D(nb_filter=32, filter_length=5, border_mode='same', activation='relu')(embedded_sequences)
  x = MaxPooling1D(2)(x)
  x = Conv1D(128, 5, activation='relu')(x)
  x = MaxPooling1D(2)(x)
  x = Conv1D(nb_filter=128, filter_length=5, activation='relu')(x)
  x = GlobalMaxPooling1D()(x)
  x = Dense(250, activation='relu')(x)

  preds = Dense(1, activation='softmax')(x)

  model = Model(sequence_input, preds)
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  return model
