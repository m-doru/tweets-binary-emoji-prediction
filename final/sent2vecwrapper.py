import os
import subprocess

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

  def get_summary(self):
    '''
    Returns a string representations of the models parameters used for serialization
    :return: string representation
    '''
    return '_'.join([self.id, self.architecture, self.batch_size, self.epochs, self.random_seed])

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

  def get_embeddings_serialization_path(self, id_x, id_y):
    pass

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

    embeddings = np.genfromtxt(SWAP_FILE_EMBEDDINGS)

    try:
      os.remove(SWAP_FILE_TWEETS)
      os.remove(SWAP_FILE_EMBEDDINGS)
    except Exception as e:
      print("Exception occured in removing sent2vec swap file.", e)

    return embeddings
