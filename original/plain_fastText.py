from fastText import train_supervised
import numpy as np
import logging
import os

logging.basicConfig(filename=os.path.join('logs', 'fastText_trainings.log'),level=logging.INFO,
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


SWAP_FILE = os.path.join('processed_data', 'swap_file.txt')
LABEL_IDENTIFIER = '__label__'
SEPARATOR = ' , '

class FastTextClassifier:
    def __init__(self, params, name):
        self.model = None
        self.params = params
        self.name = name

    def get_name(self):
        return self.name

    def fit(self, X, y):
        with open(SWAP_FILE, 'w', encoding='utf8') as f:
            for tweet, label in zip(X, y):
                f.write(LABEL_IDENTIFIER + str(label) + SEPARATOR + tweet.strip() + '\n')

        self.model = train_supervised(SWAP_FILE, **self.params)
        os.remove(SWAP_FILE) # DELETE SWAP FILE AFTER USAGE

    def predict(self, X):
        result = []

        for tweet in X:

            prediction = (self.model.predict(tweet.strip(), 1)[0][0])[9:]
            result.append(prediction)

        return np.array(result)

    def predict_proba(self, X):
        result = []

        for tweet in X:
            labels, probs = self.model.predict(tweet.strip(), 2)

            if labels[0] == '__label__-1':
                result.append(probs)
            else:
                result.append([probs[1], probs[0]])

        return np.array(result)

    def score(self, X, y):
        correct = 0

        for tweet, label in zip(X, y):
            prediction = (self.model.predict(tweet.strip(), 1)[0][0])[9:]
            if prediction == str(label):
                correct += 1

        return (correct * 1.0) / len(y)

def fastText_plain(X_train, y_train, X_test, y_test, params):
    logging.info("\n################################################################")
    logging.info("Parameters: " + str(params))
    print("Parameters: " + str(params))

    classifier = FastTextClassifier(params)
    classifier.fit(X_train, y_train)

    training_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)
    logging.info("Training accuracy: " + str(training_accuracy))
    logging.info("Test accuracy: " + str(test_accuracy))

    print("Training accuracy: " + str(training_accuracy))
    print("Test accuracy: " + str(test_accuracy))

    return classifier

def fastText_plain_submission(X_train, y_train, X_test, params):
    classifier = FastTextClassifier(params)
    classifier.fit(X_train, y_train)

    return classifier.predict(X_test)