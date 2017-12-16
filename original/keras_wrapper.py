import os
import numpy as np
import keras.models
from hashlib import sha1
from sklearn import preprocessing

from glove_keras import pretrained_glove_keras_model

BASE_DIR = '../data'
SENT2VEC_MODEL_PATH = os.path.join(BASE_DIR, 'twitter_bigrams.bin')
FASTTEXT_PATH = os.path.join('..','sent2vec','fasttext')

class KerasModelWrapper:
  def __init__(self, model, id, batch_size, epochs, sent2vec=False):
    self.model = model
    self.model_weights = model.get_weights()
    self.id = id
    self.fit_params = {
      'batch_size':batch_size,
      'epochs':epochs
    }
    self.sent2vec = sent2vec

  def fit(self, X, y):
    if self.sent2vec:
      X = self._get_sent2vec_embeddings(X)
      self.model.set_weights(self.model_weights)
    else:
      self.model = pretrained_glove_keras_model(X, y)


    X = preprocessing.scale(X)

    identifier_x = sha1(X).hexdigest()
    identifier_y = sha1(y).hexdigest()

    model_save_filepath = self.id + identifier_x + identifier_y + '.hdf5'

    if os.path.isfile(model_save_filepath):
      self.model = keras.models.load_model(model_save_filepath)
    else:
      self.model.fit(X, y, **self.fit_params)
      self.model.save(model_save_filepath)

  def predict(self, X):
    return self.model.predict(X)

  def score(self, X, y):
    preds = self.model.predict(X)

    return (np.count_nonzero(preds == y) * 1.0) / len(y)

  def _get_sent2vec_embeddings(self, tweets):
    SWAP_FILE_TWEETS = 'sent2vec_tweets'
    SWAP_FILE_EMBEDDINGS = 'sent2vec_embeddings'

    with open(SWAP_FILE_TWEETS, 'w', encoding='utf8') as f:
      for tweet in tweets:
        f.write(tweet.strip() + '\n')
      f.flush()

    subprocess.call(FASTTEXT_PATH +
                    'print-sentence-vectors' +
                    SENT2VEC_MODEL_PATH +
                    '<' +
                    SWAP_FILE_TWEETS +
                    '>' +
                    SWAP_FILE_EMBEDDINGS, shell=True)


    df = pd.read_csv(SWAP_FILE_EMBEDDINGS, sep=' ', header=None)
    df.drop(df.columns[-1], axis=1, inplace=True)
    df.drop(df.index[-1], inplace=True)
    embeddings = df.values.astype('float64')

    os.remove(SWAP_FILE_TWEETS)
    os.remove(SWAP_FILE_EMBEDDINGS)

    return embeddings
