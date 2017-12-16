import logging
import os
import itertools as it
from sklearn.linear_model import LogisticRegression

import numpy as np
from stacking import stacking_test_accuracy, stacking_submission
from fastText_keras import fastText_keras
from plain_fastText import fastText_plain, fastText_plain_submission, FastTextClassifier
from util import construct_dataset_from_files, construct_test_from_file, create_submission
from sent2vec_keras import trained_sent2vec_keras_model
from keras_wrapper import KerasModelWrapper

logging.basicConfig(filename=os.path.join('logs', 'logs.log'), level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

TRAINING_FILES_FULL = {'train_pos_full.txt': 1,
                       'train_neg_full.txt': -1}

TRAINING_FILES_PARTIAL = {'train_pos.txt': 1,
                          'train_neg.txt': -1}

TEST_FILE = os.path.join('..', 'data', 'twitter-datasets', 'test_data.txt')
SUBMISSION_FILE = os.path.join('processed_data', 'submission.csv')

params_keras = {
    'ngram_range': 1,
    'max_features': 20000,
    'maxlen': 60,
    'batch_size': 32,
    'embedding_dims': 50,
    'epochs': 5,
    'nb_words': None}

parameters_fastText_plain = {'lr': 0.05,  # try to vary this further
                             'dim': 30,  # try to vary this further
                             'ws': 5,
                             'epoch': 12,  # to try to vary this, also
                             'minCount': 1,
                             'minCountLabel': 0,
                             'minn': 2,
                             'maxn': 7,
                             'neg': 5,
                             'wordNgrams': 5,
                             'loss': 'softmax',
                             'bucket': 10000000,
                             'thread': 8,
                             'lrUpdateRate': 100,
                             't': 0.0001,
                             'verbose': 2}

PRETRAINED_FILE = os.path.join('processed_data', 'wiki.en.bin')

parameters_pretrained_fastText_plain = {'lr': 0.05,  # try to vary this further
                                        'dim': 30,  # try to vary this further
                                        'ws': 5,
                                        'epoch': 12,  # to try to vary this, also
                                        'minCount': 1,
                                        'minCountLabel': 0,
                                        'minn': 2,
                                        'maxn': 7,
                                        'neg': 5,
                                        'wordNgrams': 5,
                                        'loss': 'softmax',
                                        'bucket': 10000000,
                                        'thread': 8,
                                        'lrUpdateRate': 100,
                                        't': 0.0001,
                                        'verbose': 2,
                                        'pretrainedVectors': str(PRETRAINED_FILE)}

multiple_params_fastText = {'lr': [0.05],  # try to vary this further
                            'dim': [30, 50, 100, 150, 200, 300, 400],  # try to vary this further
                            'ws': [5],
                            'epoch': [12],  # to try to vary this, also
                            'minCount': [1],
                            'minCountLabel': [0],
                            'minn': [2],
                            'maxn': [7],
                            'neg': [5],
                            'wordNgrams': [5],
                            'loss': ['softmax'],
                            'bucket': [50000000],
                            'thread': [8],
                            'lrUpdateRate': [100],
                            't': [0.0001],
                            'verbose': [2]}

params_glove_keras_model_conv = {'model':None,
                            'id':'glove_pretrained_keras_conv1D',
                            'batch_size':128,
                            'epochs':2,
                            'architecture':'glove_conv'}
params_glove_keras_model_lstm = {'model':None,
                            'id':'glove_pretrained_keras_lstm',
                            'batch_size':128,
                            'epochs':2,
                            'architecture':'glove_lstm'}


def get_training_files_for_params(full_dataset, prefix):
    if full_dataset:
        return {os.path.join('data', prefix + base_file):label for base_file, label in TRAINING_FILES_FULL.items()}

    return {os.path.join('data', prefix + base_file): label for base_file, label in TRAINING_FILES_PARTIAL.items()}

SUBMISSION = False
FULL_DATASET = True 
PREFIX = 'no_dups_'

np.random.seed(0)

################################ Start running code ##############################

def create_ensemble_classifiers():
    classifiers = []

    classifier = FastTextClassifier(parameters_fastText_plain, "fastText_plain_" + str(parameters_fastText_plain))
    classifiers.append(classifier)

    classifier = KerasModelWrapper(**params_glove_keras_model_conv)
    classifiers.append(classifier)

    classifier = KerasModelWrapper(**params_glove_keras_model_lstm)
    classifiers.append(classifier)

    classifier = trained_sent2vec_keras_model()  
    classifiers.append(classifier)

    return classifiers


def create_ensemble_second_tier_classifier():
    return LogisticRegression()


def run_test_accuracy():
    training_files = get_training_files_for_params(full_dataset=FULL_DATASET, prefix=PREFIX)
    X_train, y_train, X_test, y_test = construct_dataset_from_files(training_files, split_size=0.1)

    # varNames = sorted(multiple_params_fastText)
    # combinations = [dict(zip(varNames, prod)) for prod in it.product(*(multiple_params_fastText[varName] for varName in varNames))]
    #
    # for possible_params in combinations:
    #     fastText_plain(X_train, y_train, X_test, y_test, possible_params)

    # pretrained_fastText_plain(X_train, y_train, X_test, y_test, parameters_fastText_plain)

    # fastText_keras(X_train, y_train, X_test, y_test)
    # fastText_plain(X_train, y_train, X_test, y_test, parameters_fastText_plain)

    classifiers = create_ensemble_classifiers()
    second_classifier = create_ensemble_second_tier_classifier()
    stacking_test_accuracy(second_classifier, classifiers, X_train, y_train, X_test, y_test)


def run_submission():
    training_files = get_training_files_for_params(full_dataset=FULL_DATASET, prefix=PREFIX)
    X_train, y_train = construct_dataset_from_files(training_files, split_size=None)
    tweets, ids = construct_test_from_file(TEST_FILE)

    # predictions = fastText_plain_submission(X_train, y_train, tweets, parameters_fastText_plain)

    classifiers = create_ensemble_classifiers()
    second_classifier = create_ensemble_second_tier_classifier()
    predictions = stacking_submission(second_classifier, classifiers, X_train, y_train, tweets)

    create_submission(SUBMISSION_FILE, predictions, ids)

def run():
    if SUBMISSION:
        run_submission()
    else:
        run_test_accuracy()

run()
