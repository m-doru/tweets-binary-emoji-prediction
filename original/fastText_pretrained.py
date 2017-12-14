from fastText import train_supervised, load_model
import numpy as np
import logging
import os


SWAP_FILE = os.path.join('processed_data', 'swap_file.txt')
LABEL_IDENTIFIER = '__label__'
SEPARATOR = ' , '
MODEL_FILE = os.path.join('processed_data', 'wiki.en.bin')

class FastTextClassifier:
    def __init__(self, params):
        self.model = None
        self.params = params

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

    def load_model(self):
        self.model = load_model(MODEL_FILE)

    def score(self, X, y):
        correct = 0

        for tweet, label in zip(X, y):
            prediction = (self.model.predict(tweet.strip(), 1)[0][0])[9:]
            if prediction == str(label):
                correct += 1

        return (correct * 1.0) / len(y)

def pretrained_fastText_plain(X_train, y_train, X_test, y_test, params):
    logging.info("\n################################################################")
    logging.info("Using pretrained vectors!")
    logging.info("Parameters: " + str(params))

    print("\n################################################################")
    print("Using pretrained vectors!")
    print("Parameters: " + str(params))

    classifier = FastTextClassifier(params)
    classifier.load_model()
    classifier.fit(X_train, y_train)

    training_accuracy = classifier.score(X_train, y_train)
    test_accuracy = classifier.score(X_test, y_test)

    logging.info("Training accuracy: " + str(training_accuracy))
    logging.info("Test accuracy: " + str(test_accuracy))

    print("Training accuracy: " + str(training_accuracy))
    print("Test accuracy: " + str(test_accuracy))

    return classifier