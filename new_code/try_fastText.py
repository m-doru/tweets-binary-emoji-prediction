from fastText import train_supervised
from sklearn.model_selection import cross_val_score
import numpy as np
import logging
import os

logging.basicConfig(filename='app_log.log',level=logging.INFO)

TRAINING_SET = os.path.join('processed_data', 'train_total_full.txt')

SWAP_FILE = os.path.join('processed_data', 'swap_file.txt')
LABEL_IDENTIFIER = ' __label__'

class FastTextClassifier:
    def __init__(self, params):
        self.model = None
        self.params = params

    def fit(self, X, y):
        with open(SWAP_FILE, 'w') as f:
            for tweet, label in zip(X, y):
                f.write(tweet + LABEL_IDENTIFIER + label + '\n')

        self.model = train_supervised(SWAP_FILE, **self.params)
        os.remove(SWAP_FILE) # DELETE SWAP FILE AFTER USAGE

    def predict(self, X):
        result = []

        for tweet in X:
            prediction = self.model.predict(tweet, 1)[0][0]
            result.append(prediction)

        return np.array(result)

    def score(self, X, y):
        predicted = np.array([self.predict(tweet) for tweet in X])
        return np.count_nonzero(predicted == y)

def construct_dataset_from_file(filename):
    X = []
    y = []

    with open(filename, 'r') as f:
        for line in f:
            tweet, label = line.strip().split(LABEL_IDENTIFIER)
            X.append(tweet)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = np.random.shuffle(X)
    y = np.random.shuffle(y)

    return X, y

parameters = {'lr':0.1,
          'dim':100,
          'ws':5,
          'epoch':5,
          'minCount':2,
          'minCountLabel':0,
          'minn':0,
          'maxn':0,
          'neg':5,
          'wordNgrams':2,
          'loss':'softmax',
          'bucket':2000000,
          'thread':12,
          'lrUpdateRate':100,
          't':0.0001,
          'verbose':2}

logging.info("################################################################")

X, y = construct_dataset_from_file(TRAINING_SET)
print("Finished constructing dataset.")
logging.info("Finished constructing dataset.")

classifier = FastTextClassifier(parameters)

scores = cross_val_score(classifier, X, y, cv=5)
logging.info("Parameters: " + str(parameters))
logging.info("Mean accuracy: " + str(scores.mean()))
logging.info("Scores detailed: " + str(scores))

print("Mean accuracy: " + str(scores.mean()))
print("Scores detailed: " + str(scores))



