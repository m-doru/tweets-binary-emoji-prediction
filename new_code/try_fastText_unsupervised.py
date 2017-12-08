import os

import numpy as np

from fastText import train_unsupervised
from fastText import load_model
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import logging

LEARNING_SET = os.path.join('processed_data', 'unsupervised_total.txt')
TRAINING_SET = os.path.join('processed_data', 'train_total_full.txt')
MODEL_PATH = os.path.join('processed_data', 'model_unsupervised')

SWAP_FILE = os.path.join('processed_data', 'swap_file.txt')

LABEL_IDENTIFIER = '__label__'
SEPARATOR = ' , '

logging.basicConfig(filename='app_log_unsupervised.log',level=logging.INFO)


def construct_unsupervised_dataset_from_file(filename):
    X = []

    with open(filename, 'r') as f:
        for line in f:
            X.append(line)

    X = np.array(X)
    np.random.shuffle(X)

    return X

def construct_supervised_dataset_from_file(filename):
    X = []
    y = []

    with open(filename, 'r') as f:
        for line in f:
            splits = line.strip().split(SEPARATOR, 1)
            label = splits[0].split(LABEL_IDENTIFIER)[1]
            tweet = splits[1]
            X.append(tweet)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    p = np.random.permutation(len(y))

    X = X[p]
    y = y[p]

    return X, y


params = {"model": "skipgram",
          "lr": 0.05,
          "dim": 100,
          "ws": 5,
          "epoch": 20,
          "minCount": 5,
          "minCountLabel": 0,
          "minn": 2,
          "maxn": 7,
          "neg": 5,
          "wordNgrams": 2,
          "loss": "ns",
          "bucket": 2000000,
          "thread": 12,
          "lrUpdateRate": 100,
          "t": 1e-4}

def learn_unsupervised():
    X = construct_unsupervised_dataset_from_file(LEARNING_SET)

    with open(SWAP_FILE, 'w') as f:
        for tweet in X:
            f.write(tweet.strip() + '\n')

    model = train_unsupervised(SWAP_FILE, **params)
    os.remove(SWAP_FILE)  # DELETE SWAP FILE AFTER USAGE
    return model

def transform_dataset(X, model):
    result = []
    for tweet in X:
        result.append(model.get_sentence_vector(tweet.strip()))

    return np.array(result)


logging.info("################################################################")

# model = learn_unsupervised()
# model.save_model(MODEL_PATH)
model = load_model(MODEL_PATH)
logging.info("Finished model unsupervised")
print("Finished model unsupervised")

X, y = construct_supervised_dataset_from_file(TRAINING_SET)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)
X_train, X_test = transform_dataset(X_train, model), transform_dataset(X_test, model)
print("Finished constructing dataset.")
logging.info("Finished constructing dataset.")

logistic = linear_model.LogisticRegression(C=1e5, penalty='l2', verbose=True, n_jobs=8)
logistic.fit(X_train, y_train)
accuracy = logistic.score(X_test, y_test)
print(accuracy)
