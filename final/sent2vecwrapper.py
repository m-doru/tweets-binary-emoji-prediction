import os
import subprocess
from hashlib import sha1
import logging

import numpy as np

import keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers

BASE_DIR = '../data'
SENT2VEC_MODEL_PATH = os.path.join(BASE_DIR, 'twitter_bigrams.bin')
FASTTEXT_PATH = os.path.join('..','sent2vec','fasttext')

class Sent2vecKerasWrapper:
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

  def get_name(self):
    return self.id

  def fit(self, X, y):
    id_x = sha1(X).hexdigest()
    id_y = sha1(y).hexdigest()

    model_serialization_path = self.get_model_serialization_path(id_x, id_y)
    if os.path.exists(model_serialization_path):
      # load the model
      self.model = keras.models.load_model(model_serialization_path)
      logging.info("Model loaded from {}".format(model_serialization_path))
    else:
      np.random.seed(self.random_seed)
      X_train = self._get_sent2vec_embeddings(X)

      logging.info('Training model {}\nTraining on data tensor of shape {}'.format(self.get_summary(), X_train.shape))

      self._instantiate_model_for_architecture()

      y = y.copy()
      y[y == -1] = 0
      self.model.fit(X_train, y, epochs=self.epochs, batch_size=self.batch_size)

      # serialize the model
      self._create_serialization_directory()
      self.model.save(model_serialization_path)

  def predict(self, X):
    X_train = self._get_sent2vec_embeddings(X)
    preds = self.model.predict_proba(X_train)

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

  def get_embeddings_serialization_path(self, id_x):
    '''
    Return the path where the embeddings for multiple training sets will be serialized
    :param id_x: the sha1 of the training set
    :return: the path where the embeddings should be serialized
    '''
    dirs_path = self.get_serialization_directory_path()
    embeddings_name = '_'.join([id_x, 'sent2vec','embeddings'])
    path = os.path.join(dirs_path, embeddings_name)

    return path

  def _create_serialization_directory(self):
    '''
    Function that creates the directories structure where the model and embeddings will be serialized
    :return:
    '''
    dirs = self.get_serialization_directory_path()
    if not os.path.exists(dirs):
      os.makedirs(dirs)

  def _instantiate_model_for_architecture(self):
    if self.architecture == 'dense':
      self._instantiate_dense_model()
    else:
      raise Exception('Architecture can be one of [dense] for {}'.format(self.get_summary()))

  def _instantiate_dense_model(self):
    self.model = Sequential()
    self.model.add(Dense(700, activation='relu', input_shape=(700,),kernel_regularizer=regularizers.l2(0.001)))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(700, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(700, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(700, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    self.model.add(Dropout(0.2))
    self.model.add(Dense(1, activation='sigmoid'))
    self.model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

  def _get_sent2vec_embeddings(self, tweets):
    '''
    Function that either computes and serializes the sent2vec embeddings for the given tweets or loads them if they are
    already computed and serialized
    :param tweets: tweets for which the embeddings will be computed
    :return: embeddings of the given tweets
    '''
    id_tweets = sha1(tweets).hexdigest()

    SWAP_FILE_TWEETS = '.sent2vec_tweets.swp'
    tweets_embeddings_path = self.get_embeddings_serialization_path(id_tweets)

    # if the embeddings are already computed, simply return them
    if os.path.exists(tweets_embeddings_path):
      return np.load(tweets_embeddings_path + '.npy')

    # otherwise, write the tweets in a file and use sent2vec command line tool to compute embeddings for it
    with open(SWAP_FILE_TWEETS, 'w', encoding='utf8') as f:
      for tweet in tweets:
        f.write(tweet.strip() + '\n')
      f.flush()

    # make sure the directory where we serialize is already created
    self._create_serialization_directory()

    # call the sent2vec command line to compute embeddings for tweets
    print(subprocess.call(FASTTEXT_PATH +
                          ' print-sentence-vectors ' +
                          SENT2VEC_MODEL_PATH +
                          ' < ' +
                          SWAP_FILE_TWEETS +
                          ' > ' +
                          tweets_embeddings_path, shell=True))

    embeddings = []

    # read the embeddings im memory
    with open(tweets_embeddings_path, 'r') as f:
      for line in f:
        try:
          values = line.strip().split()
          coefs = np.asarray(values, dtype='float32')
          embeddings.append(coefs)
        except Exception:
          pass

    try:
      os.remove(SWAP_FILE_TWEETS)
    except Exception as e:
      print("Exception occured in removing sent2vec swap file.", e)

    tweets_embeddings = np.asarray(embeddings)
    np.save(tweets_embeddings_path, tweets_embeddings)

    return tweets_embeddings
