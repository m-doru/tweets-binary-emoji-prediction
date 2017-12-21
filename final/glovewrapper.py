'''
This code is highly inspired from
https://github.com/keras-team/keras/blob/master/examples/pretrained_word_embeddings.py
'''
import os
import logging
import numpy as np
import pickle

from hashlib import sha1

import keras.models
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout


MAX_SEQUENCE_LENGTH = 100
MAX_NB_WORDS = 60000
EMBEDDING_DIM = 200

BASE_DIR = '../data/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove')
PRETRAINED_EMBEDDINGS_FILE = os.path.join(GLOVE_DIR, 'glove.twitter.27B.{}d.txt'.format(EMBEDDING_DIM))


class GloveKerasWrapper:
  def __init__(self, id, architecture, batch_size, epochs, random_state, serializations_directory):
    '''
    Constructor for the desired keras model using glove embeddings
    :param id: Model's id. Used to serialize the weights
    :param architecture: The desired model architecture. One of: lstm, conv
    :param batch_size: batch size use for training
    :param epochs: number of epochs to train
    :param random_state: random state of the model
    '''
    self.id = id
    self.architecture = architecture
    self.batch_size = batch_size
    self.epochs = epochs
    self.random_seed = random_state
    self.serializations_directory = serializations_directory
    self.model = None

  def get_name(self):
    return self.id

  def fit(self, X, y):
    '''
    Method that either loads the model if a serialization with this data exists or trains the model otherwise. The
    training is done in multiple steps:
    1. The glove pretrained embeddings are read
    2. The textual data is transformed into sequences of integers
    3. The embedding matrix from these integers is created
    4. The model is created with the characteristics of the embeddings matrix
    5. Actual training
    :param X: textual data to train on
    :param y: labels for each of the data entries
    :return: Nothing
    '''
    identifier_x = sha1(X).hexdigest()
    identifier_y = sha1(y).hexdigest()

    model_serialization_path = self.get_model_serialization_path(identifier_x, identifier_y)
    tokenizer_serialization_path = os.path.join(self.get_serialization_directory_path(), '_tokenizer.pkl')

    if os.path.exists(model_serialization_path) and \
            os.path.exists(tokenizer_serialization_path):
      # load the model
      self.model = keras.models.load_model(model_serialization_path)
      logging.info("Model loaded from {}".format(model_serialization_path))
      # load the tokenizer
      with open(tokenizer_serialization_path, 'rb') as f:
        self.tokenizer = pickle.load(f)
    else:
      logging.info("Did not find model. State below.")
      logging.info("Required model serialization path: " + str(model_serialization_path) + ". Exists: " + str(os.path.exists(model_serialization_path)))
      logging.info("Required tokenizer path: " + str(model_serialization_path) + ". Exists: " + str(os.path.exists(tokenizer_serialization_path)))

      print("Did not find model. State below.")
      print("Required model serialization path: " + str(model_serialization_path) + ". Exists: " + str(os.path.exists(model_serialization_path)))
      print("Required tokenizer path: " + str(model_serialization_path) + ". Exists: " + str(os.path.exists(tokenizer_serialization_path)))

      np.random.seed(self.random_seed)
      embeddings_index = self._read_pretrained_glove_embeddings(PRETRAINED_EMBEDDINGS_FILE)
      X_train, word_index, self.tokenizer = self._transform_tweets_to_sequences(X, True)
      embedding_matrix = self._construct_embedding_matrix(word_index, embeddings_index)

      # make sure the serialization directory is created before trying to serialize anything
      self.create_serialization_directory()
      # serialize the tokenizer to be used for computing sequences at predict time
      with open(tokenizer_serialization_path, 'wb') as f:
        pickle.dump(self.tokenizer, f)

      logging.info('Training model {}\nTraining on data tensor of shape {}'.format(self.get_summary(), X_train.shape))

      # create the model
      self.instantiate_model_for_architecture(X_train, embedding_matrix)

      # get the proper labels and train the model
      y = y.copy()
      y[y == -1] = 0
      self.model.fit(X_train, y, epochs=self.epochs, batch_size=self.batch_size)

      # serialize the model
      self.model.save(model_serialization_path)

  def predict(self, X):
    '''
    Method that computes the predictions for the given textual data
    :param X: Textual data to which we will do inference. One text per row
    :return: Probabilities predictions
    '''
    id_x = sha1(X).hexdigest()
    sequences_serialization_path = self.get_sequences_serialization_path(id_x)

    if os.path.exists(sequences_serialization_path + '.npy'):
        X = np.load(sequences_serialization_path + '.npy')
        logging.info("Loaded glove tokenizer sequences from {}".format(sequences_serialization_path))
    else:
        X = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        np.save(sequences_serialization_path, X)
        logging.info("Saved glove tokenizer sequences to {}".format(sequences_serialization_path))

    preds = self.model.predict_proba(X)
    return preds

  def get_summary(self):
    '''
    Returns a string representations of the models parameters used for serialization
    :return: string representation
    '''
    return '_'.join([str(self.id), str(self.architecture), str(self.batch_size), str(self.epochs),
                     str(self.random_seed)])

  def get_serialization_directory_path(self):
    '''
    Function that get the path of directories where the models and embeddings will be serialized
    :return:
    '''
    model_summary = self.get_summary()
    dirs = os.path.join(self.serializations_directory, model_summary)
    return dirs

  def get_model_serialization_path(self, id_x, id_y):
    '''
    Return the path where the model should be serialized
    :param id_x: sha1 hash of the training data
    :param id_y:  sha1 hash of the labels
    :return:
    '''
    dirs_path = self.get_serialization_directory_path()
    serialization_name = '_'.join([id_x, id_y, '.hdf5'])
    path = os.path.join(dirs_path, serialization_name)

    return path


  def get_sequences_serialization_path(self, id_x):
    dirs_path = self.get_serialization_directory_path()
    embeddings_name = '_'.join([id_x, 'glove', 'sequences'])
    path = os.path.join(dirs_path, embeddings_name)

    return path

  def create_serialization_directory(self):
    '''
    Function that creates the directories structure where the model and embeddings will be serialized
    :return:
    '''
    dirs = self.get_serialization_directory_path()
    if not os.path.exists(dirs):
      os.makedirs(dirs)


  def instantiate_model_for_architecture(self, X_train, embeddings_matrix):
    '''
    Creates the keras model specified by the architecture given in constructor
    :param X_train: Sequence embeddings on which the model will be trained
    :param embeddings_matrix: embeddings for each word used in the embedding layer to compute the embedding of the
    tweet
    :return: nothing
    '''
    if self.architecture == 'conv':
      self._instantiate_conv_model(X_train, embeddings_matrix)
    elif self.architecture == 'lstm':
      self._instantiate_lstm_model(X_train, embeddings_matrix)
    else:
      raise Exception("Architecture can be one of [conv, lstm] for {}".format(self.get_summary()))

  def _instantiate_conv_model(self, X_train, embeddings_matrix):
    '''
    Creates a sequential keras model with one Convolutional layer and two fully connected layers at the end
    :param X_train: Sequence embeddings on which the model will be trained
    :param embeddings_matrix: embeddings for each word used in the embedding layer to compute the embedding of the
    :return:
    '''
    self.model = Sequential()
    self.model.add(Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1], input_length=X_train.shape[1],
                         weights=[embeddings_matrix]))
    self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    self.model.add(MaxPooling1D(pool_size=2))
    self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    self.model.add(MaxPooling1D(pool_size=2))
    self.model.add(Dropout(0.2))
    self.model.add(Flatten())
    self.model.add(Dense(250, activation='relu'))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  def _instantiate_lstm_model(self, X_train, embeddings_matrix):
    '''
    Create a sequential model with a Convolutional and a Long-Short-Term-Memory layer with one dense layer at the end
    :param X_train: Sequence embeddings on which the model will be trained
    :param embeddings_matrix: embeddings for each word used in the embedding layer to compute the embedding of the
    :return:
    '''
    self.model = Sequential()
    self.model.add(Embedding(embeddings_matrix.shape[0],
                        embeddings_matrix.shape[1],
                        weights=[embeddings_matrix],
                        input_length=X_train.shape[1],
                        trainable=False))

    self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    self.model.add(MaxPooling1D(pool_size=2))
    self.model.add(LSTM(100))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  def _read_pretrained_glove_embeddings(self, path):
    '''
    Helper function that parses the glove embeddings located at the given path into a dictionary containing mappings
    from vocabulary words to their embedding
    :return: Dictionary mapping vocabulary word to their embedding
    '''
    embeddings_index = {}

    with open(path, 'r') as f:
      for line in f:
        values = line.rstrip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    return embeddings_index

  def _transform_tweets_to_sequences(self, X, pad=True):
    '''
    Helper function that uses Keras functionalities to bijectively map words to integers, hence transforming textual
    tweets in sequences of mappings
    :param pad: whether to return sequences of the same length
    :return: resulted sequences, resulted bijective mapping and the tokenizer used to transform the texts to sequences
    '''
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    logging.info('Found %s unique tokens.' % len(word_index))

    if pad:
      sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return sequences, word_index, tokenizer


  def _construct_embedding_matrix(self, word_index, embeddings_index):
    '''
    Function that takes (word_index - the bijective mapping of words to integers) and (embeddings_index - the mapping
    of words to their embedding) and creates a matrix where line i containes the embeddings corresponding to the word
    mapped to integer i.
    :param word_index: dictionary from words to integers as produced by Keras Tokenizer.word_index after fitting
    :param embeddings_index: dictionary from words to embeddings
    :return: embedding matrix
    '''
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
