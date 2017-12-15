import os
import numpy as np
from util import construct_dataset_from_files, construct_test_from_file
from fastText import load_model, train_unsupervised

MODEL_FILE = os.path.join('processed_data', 'trained_model_for_wrappers.bin')

TRAINING_FILES = {os.path.join('..', 'data', 'twitter-datasets', 'train_pos_full.txt'): 1,
                  os.path.join('..', 'data', 'twitter-datasets', 'train_neg_full.txt'): -1}

TEST_FILE = os.path.join('..', 'data', 'twitter-datasets', 'test_data.txt')

SWAP_FILE = os.path.join('processed_data', 'swap_unsupervised')

parameters_fastText_unsupervised = {'lr': 0.05,  # try to vary this further
                             'dim': 30,  # try to vary this further
                             'ws': 5,
                             'epoch': 12,  # to try to vary this, also
                             'minCount': 1,
                             'minCountLabel': 0,
                             'minn': 2,
                             'maxn': 7,
                             'neg': 5,
                             'wordNgrams': 5,
                             'loss': 'hs',
                             'bucket': 10000000,
                             'thread': 8,
                             'lrUpdateRate': 100,
                             't': 0.0001,
                             'verbose': 2}

def create_transformer_model(params):
    X_train, _ = construct_dataset_from_files(TRAINING_FILES, split_size=None)
    X_test, _ = construct_test_from_file(TEST_FILE)

    X = np.append(X_train, X_test)

    with open(SWAP_FILE, 'w') as f:
        for tweet in X:
            f.write(tweet.strip() + '\n')

    model = train_unsupervised(SWAP_FILE, **params)
    model.save_model(MODEL_FILE)
    os.remove(SWAP_FILE)  # DELETE SWAP FILE AFTER USAGE

class TweetToVector:
    def __init__(self, model_file):
        self.model = load_model(model_file)

    def transform_tweet_to_vector(self, tweet):
        return self.model.get_sentence_vector(tweet)

class Sklearn_Wrapper:
    def __init__(self, sklearner, name):
        self.learner = sklearner
        self.transformer = TweetToVector(MODEL_FILE)
        self.name = name

    def get_name(self):
        return self.name

    def fit(self, X, y):
        # this will override the previous fit call, if any
        X_train = np.array([self.transformer.transform_tweet_to_vector(tweet) for tweet in X])
        # transform text to a vector
        self.learner.fit(X_train, y) # use sklearner on the transformed data

    def predict(self, X):
        return self.learner.predict()

    def score(self, X, y):
        return self.learner.score(X, y)

if __name__ == "__main__":
    create_transformer_model(parameters_fastText_unsupervised)