import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from util import construct_predictions_from_pickles, construct_test_from_file, create_submission

import logging

logging.basicConfig(filename=os.path.join('logs', 'logs_second_tier.log'), level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
PICKLE_TRAIN = os.path.join('..', 'pred_pickles2','classifiers_folds_predictions.pkl')
PICKLE_TEST = os.path.join('..', 'pred_pickles2', 'classifiers_test_predictions.pkl')

TEST_FILE = os.path.join('..', 'data', 'twitter-datasets', 'test_data.txt')
SUBMISSION_FILE = os.path.join('processed_data', 'submission.csv')

def compute_score(predictions, labels):
    return (np.count_nonzero(predictions == labels) * 1.0 ) / len(labels)

X, y, X_test = construct_predictions_from_pickles(PICKLE_TRAIN, PICKLE_TEST)
_, ids = construct_test_from_file(TEST_FILE)

print(X.shape)

clf = GradientBoostingClassifier(n_estimators=200, \
        max_depth=3,
        verbose=True)

clf.fit(X, y)

predictions = clf.predict(X_test)

create_submission(SUBMISSION_FILE, predictions, ids)
