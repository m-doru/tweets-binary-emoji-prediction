import logging
import os
import numpy as np
from sklearn.linear_model import LogisticRegression

from params_wrappers import *
from fastText_wrapper import FastTextClassifier
from sent2vecwrapper import Sent2vecKerasWrapper
from glovewrapper import GloveKerasWrapper

from stacking import stack_classifiers
from util import construct_dataset_from_files, construct_test_from_file, create_submission, \
    get_training_files_for_params

logging.basicConfig(filename=os.path.join('logs', 'log_stacking.log'), level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")

TRAINING_FILES_FULL = {'train_pos_full.txt': 1,
                       'train_neg_full.txt': -1}

TRAINING_FILES_PARTIAL = {'train_pos.txt': 1,
                          'train_neg.txt': -1}

TEST_FILE = os.path.join('..', 'data', 'twitter-datasets', 'test_data.txt')
SUBMISSION_FILE = os.path.join('processed_data', 'submission.csv')

FULL_DATASET = True
PREFIX = 'no_dups_'

################## PRETRAIN OR NOT ##############
USE_PRETRAIN = True
#################################################



np.random.seed(0)


############################## Start writing code for creating classifiers ##############################

def create_stacking_classifiers():
    '''
    Method that creates all the classifiers to be trained during stacking.
    :return: a list of classifiers
    '''
    classifiers = []

    classifier = GloveKerasWrapper(**params_glove_keras_model_conv)
    classifiers.append(classifier)

    classifier = GloveKerasWrapper(**params_glove_keras_model_conv_2)
    classifiers.append(classifier)

    classifier = GloveKerasWrapper(**params_glove_keras_model_conv_3)
    classifiers.append(classifier)

    classifier = GloveKerasWrapper(**params_glove_keras_model_lstm)
    classifiers.append(classifier)

    classifier = GloveKerasWrapper(**params_glove_keras_model_lstm_2)
    classifiers.append(classifier)

    classifier = Sent2vecKerasWrapper(**params_sent2vec_keras_model_dense)
    classifiers.append(classifier)

    classifier = FastTextClassifier(params_fastText, "fastText")
    classifiers.append(classifier)

    return classifiers


def create_ensemble_second_tier_classifier():
    '''
    Method that creates the classifiers that is used on the second tier of stacking. 
    '''
    return LogisticRegression()


def run():
    '''
    Method that runs everything. First, read the training set, then create stacking classifiers, 
    afterwards train the whole stack and finally write the predictions into the submission file.
    :return: 
    '''
    training_files = get_training_files_for_params(full_dataset=FULL_DATASET, prefix=PREFIX,
                                                   training_files_full=TRAINING_FILES_FULL,
                                                   training_files_partial=TRAINING_FILES_PARTIAL)

    X_train, y_train = construct_dataset_from_files(training_files, split_size=None)
    X_test, ids = construct_test_from_file(TEST_FILE)

    classifiers = create_stacking_classifiers()
    second_classifier = create_ensemble_second_tier_classifier()
    predictions = stack_classifiers(second_classifier, classifiers, X_train, y_train, X_test)

    create_submission(SUBMISSION_FILE, predictions, ids)


#################################### Simple call to the run method ########################


run()
