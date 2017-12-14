import os

from util import construct_dataset_from_files
from fastText_keras import fastText_keras
from plain_fastText import fastText_plain
from fastText_pretrained import pretrained_fastText_plain
import logging
import itertools as it
import numpy as np

logging.basicConfig(filename=os.path.join('logs', 'logs.log'),level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


TRAINING_FILES_FULL = {os.path.join('..', 'data', 'twitter-datasets', 'train_pos_full.txt') : 1,
                  os.path.join('..', 'data', 'twitter-datasets', 'train_neg_full.txt'): -1}

TRAINING_FILES_PARTIAL = {os.path.join('..', 'data', 'twitter-datasets', 'train_pos.txt') : 1,
                  os.path.join('..', 'data', 'twitter-datasets', 'train_neg.txt'): -1}

TRAINING_FILES_PROCESSED_FULL = {os.path.join('processed_data', 'no_dups_train_pos_full.txt') : 1,
                  os.path.join('processed_data', 'no_dups_train_neg_full.txt'): -1}

TRAINING_FILES_PROCESSED_PARTIAL = {os.path.join('processed_data', 'no_dups_train_pos.txt') : 1,
                  os.path.join('processed_data', 'no_dups_train_neg.txt'): -1}

params_keras = {
    'ngram_range': 1,
    'max_features': 20000,
    'maxlen': 60,
    'batch_size': 32,
    'embedding_dims': 50,
    'epochs': 5,
    'nb_words':None}

parameters_fastText_plain = {'lr':0.05, # try to vary this further
          'dim':30, # try to vary this further
          'ws':5,
          'epoch':10, # to try to vary this, also
          'minCount':1,
          'minCountLabel':0,
          'minn':2,
          'maxn':7,
          'neg':5,
          'wordNgrams':5,
          'loss':'softmax',
          'bucket':10000000,
          'thread':8,
          'lrUpdateRate':100,
          't':0.0001,
          'verbose':2}

PRETRAINED_FILE = os.path.join('processed_data', 'wiki.en.bin')

parameters_pretrained_fastText_plain = {'lr':0.05, # try to vary this further
          'dim':30, # try to vary this further
          'ws':5,
          'epoch':10, # to try to vary this, also
          'minCount':1,
          'minCountLabel':0,
          'minn':2,
          'maxn':7,
          'neg':5,
          'wordNgrams':5,
          'loss':'softmax',
          'bucket':10000000,
          'thread':8,
          'lrUpdateRate':100,
          't':0.0001,
          'verbose':2,
          'pretrainedVectors':str(PRETRAINED_FILE)}

multiple_params_fastText = {'lr':[0.05], # try to vary this further
          'dim':[30], # try to vary this further
          'ws':[5],
          'epoch':[10], # to try to vary this, also
          'minCount':[1,2],
          'minCountLabel':[0],
          'minn':[2],
          'maxn':[7],
          'neg':[5],
          'wordNgrams':[2,5],
          'loss':['softmax'],
          'bucket':[50000000],
          'thread':[8],
          'lrUpdateRate':[100, 1000],
          't':[0.0001, 0.001],
          'verbose':[2]}

def get_training_files_for_params(full_dataset, processed):
    if full_dataset and processed:
        return TRAINING_FILES_PROCESSED_FULL
    if full_dataset and (not processed):
        return TRAINING_FILES_FULL
    if (not full_dataset) and processed:
        return TRAINING_FILES_PROCESSED_PARTIAL
    if (not full_dataset) and (not processed):
        return TRAINING_FILES_PARTIAL

FULL_DATASET = False
PROCESSED = True

np.random.seed(0)

################################ Start running code ##############################
training_file = get_training_files_for_params(full_dataset=FULL_DATASET, processed=PROCESSED)
X_train, y_train, X_test, y_test = construct_dataset_from_files(training_file)

# varNames = sorted(multiple_params_fastText)
# combinations = [dict(zip(varNames, prod)) for prod in it.product(*(multiple_params_fastText[varName] for varName in varNames))]
#
# for possible_params in combinations:
#     fastText_plain(X_train, y_train, X_test, y_test, possible_params)

# pretrained_fastText_plain(X_train, y_train, X_test, y_test, parameters_fastText_plain)

fastText_keras(X_train, y_train, X_test, y_test)
# fastText_plain(X_train, y_train, X_test, y_test, parameters_pretrained_fastText_plain)