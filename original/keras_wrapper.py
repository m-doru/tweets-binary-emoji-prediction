import os
import subprocess
import numpy as np
from hashlib import sha1

import pandas as pd

import keras.models
from keras.preprocessing.sequence import pad_sequences

from glove_keras import pretrained_glove_keras_model_conv
from glove_keras import pretrained_glove_keras_model_lstm

BASE_DIR = '../data'
SENT2VEC_MODEL_PATH = os.path.join(BASE_DIR, 'twitter_bigrams.bin')
FASTTEXT_PATH = os.path.join('..','sent2vec','fasttext')

SAVED_WEIGHTS_ID = {
        'glove_pretrained_keras_conv1D' : 1,
        'glove_pretrained_keras_lstm' : 1,
        'sent2vec_trained_keras_model' : 1
    }

WEIGHTS = {
        'glove_pretrained_keras_conv' : [
            'glove_pretrained_keras_conv1Db3f0c8231a6e61f3cb04db2b2f471483c0a92e0a9f432f218bf120035bffdaf50f445752e7d1284e.hdf5',
            'glove_pretrained_keras_conv1Dbdb04936c9c5075ad8d61674adaee3f46bd7c07614d270d9c34452fdcf3b67b19e9e9019836da72e.hdf5',
            'glove_pretrained_keras_conv1De0ccad8d964eb3d15c13d1245ea29c81aeb935ea55420816821a7d2c8671a506b02449c45176e011.hdf5'
            ],
        'glove_pretrained_keras_lstm' : [
            'glove_pretrained_keras_lstmb3f0c8231a6e61f3cb04db2b2f471483c0a92e0a9f432f218bf120035bffdaf50f445752e7d1284e.hdf5',
            'glove_pretrained_keras_lstmbdb04936c9c5075ad8d61674adaee3f46bd7c07614d270d9c34452fdcf3b67b19e9e9019836da72e.hdf5',
            'glove_pretrained_keras_lstme0ccad8d964eb3d15c13d1245ea29c81aeb935ea55420816821a7d2c8671a506b02449c45176e011.hdf5',
            ],
        'sent2vec_trained_keras_model' : [
            'sent2vec_trained_keras_model1e9b52e19aa040c71d56f7d68dd0e3b8936b9d6e55420816821a7d2c8671a506b02449c45176e011.hdf5',
            'sent2vec_trained_keras_model4b42eaae54a4e72a0d446dba2c950c88a72a881d14d270d9c34452fdcf3b67b19e9e9019836da72e.hdf5',
            'sent2vec_trained_keras_modelcad4c3b3baf31f9c5e89299c759230b4f13222c469d294f895b23d35dd9def07a216c658b88148d1.hdf5',
            'sent2vec_trained_keras_modelffa64cc19310a2854bf5b135b5d7917e02e9e7819f432f218bf120035bffdaf50f445752e7d1284e.hdf5',
            ]
    }

class KerasModelWrapper:
  def __init__(self, model, id, batch_size, epochs, architecture=None, tokenizer=None, max_seq_len=None):
    self.model = model
    self.id = id
    self.fit_params = {
      'batch_size':batch_size,
      'epochs':epochs
    }
    self.architecture = architecture 
    self.tokenizer = tokenizer
    self.max_seq_len = max_seq_len
    if model is not None:
        self.model_weights = model.get_weights()

  def fit(self, X, y):
    y = y.copy()
    y[y==-1] = 0

    if self.architecture == 'sent2vec':
      X = self._get_sent2vec_embeddings(X)
      self.model.set_weights(self.model_weights)
    elif self.architecture == 'glove_conv':
      self.model, X, self.tokenizer, self.max_seq_len = pretrained_glove_keras_model_conv(X)
    elif self.architecture == 'glove_lstm':
      self.model, X, self.tokenizer, self.max_seq_len = pretrained_glove_keras_model_lstm(X)


    identifier_x = sha1(X).hexdigest()
    identifier_y = sha1(y).hexdigest()

    model_save_filepath = self.id + identifier_x + identifier_y + '.hdf5'
    #idx = SAVED_WEIGHTS_ID[self.id]
    #SAVED_WEIGHTS_ID[self.id] += 1
    #model_save_filepath = WEIGHTS[self.id][idx]

    if os.path.isfile(model_save_filepath):
      self.model = keras.models.load_model(model_save_filepath)
      print('Model loaded from {}'.format(model_save_filepath))
    else:
      self.model.fit(X, y, **self.fit_params)
      self.model.save(model_save_filepath)

  def predict(self, X):
    if self.architecture == 'sent2vec':
        X = self._get_sent2vec_embeddings(X)
    elif self.architecture in ['glove_conv','glove_lstm']:
        X = self.tokenizer.texts_to_sequences(X)
        X = pad_sequences(X, maxlen=self.max_seq_len)

    preds = self.model.predict(X)
    preds[preds==0] = -1
    return preds

  def score(self, X, y):
    if self.sent2vec:
        X = self._get_sent2vec_embeddings(X)
    else:
        X = self.tokenizer.text_to_sequences(X)
        X = pad_sequences(X, maxlen=self.max_seq_len)

    y = y.copy()
    y[y==-1]=0
    preds = self.model.predict(X)

    return (np.count_nonzero(preds == y) * 1.0) / len(y)
  
  def get_name(self):
    return self.id

  def _get_sent2vec_embeddings(self, tweets):
    SWAP_FILE_TWEETS = 'sent2vec_tweets'
    SWAP_FILE_EMBEDDINGS = 'sent2vec_embeddings'

    with open(SWAP_FILE_TWEETS, 'w', encoding='utf8') as f:
      for tweet in tweets:
        f.write(tweet.strip() + '\n')
      f.flush()

    print(subprocess.call(FASTTEXT_PATH +
                    ' print-sentence-vectors ' +
                    SENT2VEC_MODEL_PATH +
                    ' < ' +
                    SWAP_FILE_TWEETS +
                    ' > ' +
                    SWAP_FILE_EMBEDDINGS, shell=True))


    df = pd.read_csv(SWAP_FILE_EMBEDDINGS, sep=' ', header=None)
    df.drop(df.columns[-1], axis=1, inplace=True)
    embeddings = df.values.astype('float64')

    os.remove(SWAP_FILE_TWEETS)
    os.remove(SWAP_FILE_EMBEDDINGS)

    return embeddings.copy(order='C')
