import logging
import os

import numpy as np
from sklearn.model_selection import train_test_split


def get_training_files_for_params(full_dataset, prefix, training_files_full, training_files_partial):
    if full_dataset:
        return {os.path.join('data', prefix + base_file): label for base_file, label in training_files_full.items()}

    return {os.path.join('data', prefix + base_file): label for base_file, label in training_files_partial.items()}


def shuffle_dataset(X, y):
    np.random.seed(0)
    p = np.random.permutation(len(y))

    X = X[p]
    y = y[p]

    return X, y


def construct_dataset_from_files(filenames, split_size=0.1):
    logging.info("Loading data from files: " + str(filenames))
    print("Loading data from files: " + str(filenames))

    X = []
    y = []

    for filename, label in filenames.items():
        with open(filename, 'r', encoding='utf8') as f:
            for tweet in f:
                X.append(tweet)
                y.append(int(label))

    X = np.array(X)
    y = np.array(y)

    X, y = shuffle_dataset(X, y)

    if split_size is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=100)
        return X_train, y_train, X_test, y_test

    return X, y


def construct_test_from_file(filename):
    tweets = []
    ids = []

    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            id, tweet = line.split(',', 1)
            ids.append(id)
            tweets.append(tweet)

    return np.array(tweets), np.array(ids)


def create_submission(filename, predictions, ids):
    with open(filename, 'w', encoding='utf8') as f:
        f.write('Id,Prediction\n')
        for id, prediction in zip(ids, predictions):
            f.write(str(id) + ',' + str(prediction) + '\n')
