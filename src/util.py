import logging
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split


def get_training_files_for_params(full_dataset, prefix, training_files_full, training_files_partial):
    '''
    Function that returns the correct training files, for both cases we want to use the small dataset or 
    the big dataset.
    :param full_dataset: boolean parameter which specifies if full dataset is used or small one 
    :param prefix: the prefix of the file (might have been processed before, look into process_input)
    :param training_files_full: the dictionary with the full files and associated label for each tweet in the file
    :param training_files_partial: the dictionary with the partial files and associated label for each tweet in the file
    :return: the dictionary with the correct files
    '''
    if full_dataset:
        return {os.path.join('data', prefix + base_file): label for base_file, label in training_files_full.items()}

    return {os.path.join('data', prefix + base_file): label for base_file, label in training_files_partial.items()}


def shuffle_dataset(X, y):
    '''
    Function that shuffles a dataset.
    :param X: the instances of the dataset
    :param y: the labels of the dataset
    :return: the shuffled dataset, as a pair (X_suhffled, y_shuffled)
    '''
    np.random.seed(0)
    p = np.random.permutation(len(y))

    X = X[p]
    y = y[p]

    return X, y


def construct_dataset_from_files(filenames, split_size=0.1):
    '''
    Function that constructs a dataset from files. If split_size is None, then no split is performed and
    function returns a pair (X, y). If split_size is a float value, then the dataset is splitted into train 
    set and test set, with test set size being equal to split_size * (full_dataset_size). 
    :param filenames: the dictionary which has filenames as the keys and as values the labels for the tweets 
        in each filen.
    :param split_size: the split size of the set 
    :return: (X, y) is split_size is None, the full dataset, otherwise (X_train, y_train, X_test, y_test)
    '''
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
    '''
    Function that constructs a dataset representing the final test set used for predictions from the specified file.
    :param filename: the filename where the test set is stored
    :return: (X_test, ids), an pair of arrays representing the test set and associated id for each tweet
    '''
    tweets = []
    ids = []

    with open(filename, 'r', encoding='utf8') as f:
        for line in f:
            id, tweet = line.split(',', 1)
            ids.append(id)
            tweets.append(tweet)

    return np.array(tweets), np.array(ids)


def create_submission(filename, predictions, ids):
    '''
    Function that creates a submission, given the predictions and associated ids.
    :param filename: the filename where we store the submission
    :param predictions: the predictions for the test set
    :param ids: the ids of the tweets
    :return: None
    '''
    with open(filename, 'w', encoding='utf8') as f:
        f.write('Id,Prediction\n')
        for id, prediction in zip(ids, predictions):
            f.write(str(id) + ',' + str(prediction) + '\n')

def construct_predictions_from_pickles(path_train, path_test):
    '''
    Function that creates the prediction of the neural networks models to be used for the second tier classifier
    '''
    with open(path_train, 'rb') as f:
        [X_train, y_train] = pickle.load(f)

    with open(path_test, 'rb') as f:
        X_test = pickle.load(f)

    return X_train, y_train, X_test


