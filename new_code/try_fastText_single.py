from fastText import train_supervised
from sklearn.model_selection import train_test_split
import numpy as np
import logging
import itertools as it
import os

logging.basicConfig(filename='fastText_trainings_single.log',level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

TRAINING_SET = os.path.join('processed_data', 'train_total_full.txt')

SWAP_FILE = os.path.join('processed_data', 'swap_file_single.txt')
LABEL_IDENTIFIER = '__label__'
SEPARATOR = ' , '

class FastTextClassifier:
    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, X, y):
        with open(SWAP_FILE, 'w') as f:
            for tweet, label in zip(X, y):
                f.write(LABEL_IDENTIFIER + label + SEPARATOR + tweet.strip() + '\n')

        self.model = train_supervised(SWAP_FILE, **self.params)
        os.remove(SWAP_FILE) # DELETE SWAP FILE AFTER USAGE

    def predict(self, X):
        result = []

        for tweet in X:
            prediction = (self.model.predict(tweet, 1)[0][0]).split(LABEL_IDENTIFIER)[1]
            result.append(prediction)

        return np.array(result)

    def score(self, X, y):
        predicted = self.predict(X)

        return (np.count_nonzero(predicted == y) * 1.0) / len(y)

def construct_dataset_from_file(filename):
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

def get_accuracy(params, X_train, y_train, X_test, y_test):

    classifier = FastTextClassifier(params)
    classifier.fit(X_train, y_train)

    training_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    logging.info("Parameters: " + str(params))
    logging.info("Training accuracy: " + str(training_accuracy))
    logging.info("Test accuracy: " + str(test_accuracy))

    print("Training accuracy: " + str(training_accuracy))
    print("Test accuracy: " + str(test_accuracy))

parameters = {'lr':[0.05],
          'dim':[10, 20, 30, 50, 100],
          'ws':[5],
          'epoch':[10, 15],
          'minCount':[1],
          'minCountLabel':[0],
          'minn':[2],
          'maxn':[7],
          'neg':[5],
          'wordNgrams':[4],
          'loss':['softmax'],
          'bucket':[10000000],
          'thread':[8],
          'lrUpdateRate':[100],
          't':[0.0001],
          'verbose':[2]}

varNames = sorted(parameters)
combinations = [dict(zip(varNames, prod)) for prod in it.product(*(parameters[varName] for varName in varNames))]

logging.info("################################################################")

# np.random.seed(100)
# logging.info("Set np random seed to 100.")

X, y = construct_dataset_from_file(TRAINING_SET)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=100)
print("Finished constructing dataset.")
logging.info("Finished constructing dataset.")

for possible_params in combinations:
    print("Parameters: " + str(possible_params))
    get_accuracy(possible_params, X_train, y_train, X_test, y_test)
