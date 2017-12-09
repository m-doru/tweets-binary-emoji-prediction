import numpy as np
import os.path
import pandas as pd
import pickle

RANDOM_STATE = 777
SUFFIX = '.pkl'

def load_data(pos_path, neg_path, verbose=False):
  '''
  Loads the feature vectors from disk. The file should contain one feature vector per line with the elements
  separated by space.
  :param pos_path: Path to feature vectors corresponding to tweets with label 1
  :param neg_path: Path to feature vectors corresponding to tweets with label 0
  :param suffle: Boolean. Whether the data should be shuffled or not
  :return: X, y where X is a numpy matrix with a feature vector pe line and y is a vector with the label for each line
  '''
  if os.path.isfile(pos_path + SUFFIX) and \
    os.path.isfile(neg_path + SUFFIX):
    # with open(pos_path + SUFFIX, 'rb') as f:
    #   X_pos = pickle.load(f)
    # with open(neg_path + SUFFIX, 'rb') as f:
    #   X_neg = pickle.load(f)
    X_pos = pd.read_pickle(pos_path + SUFFIX)
    X_neg = pd.read_pickle(neg_path + SUFFIX)
    print("Finished loading pickles")
  else:
    X_pos = pd.read_csv(pos_path, sep=' ', header=None)
    empty_col = X_pos.columns[-1]
    X_pos.drop(empty_col, axis=1, inplace=True)
    X_neg = pd.read_csv(neg_path, sep=' ', header=None)
    X_neg.drop(empty_col, axis=1, inplace=True)
    print("Finished loading from txt", '\n', 'Dumping to pickle...')
    pd.to_pickle(X_pos, pos_path+SUFFIX)
    pd.to_pickle(X_neg, neg_path+SUFFIX)

    print("Finished dumping to pickle")

  X_neg = X_neg.values
  X_pos = X_pos.values
  y_pos = np.ones(X_pos.shape[0])
  y_neg = np.zeros(X_neg.shape[0])
  X = np.concatenate((X_pos, X_neg))
  y = np.concatenate((y_pos, y_neg))
  return X, y
