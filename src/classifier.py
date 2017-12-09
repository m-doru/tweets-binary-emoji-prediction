from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

import numpy as np

from utils import *
from Logger import Logger

LOG = Logger('logs.txt')

def logistic_regression_classifier(X_train, X_test, y_train, y_test):
  LOG('Starting logistic regression')
  LOG.log_shapes(X_train, X_test)

  clf = LogisticRegression(C=0.1,n_jobs=2, verbose=True)

  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  LOG.log_accuracy(accuracy)
  # create plot of data and how it was labeled
  # create_plot

def svm_classifier(X_train, X_test, y_train, y_test):
  LOG('Starting svm classifier')

  clf = svm.SVC(kernel='linear', C=0.1, verbose=True)

  clf.fit(X_train, y_train)

  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  LOG.log_accuracy(accuracy)
  # create plot

def nn_classifier(X_train, X_test, y_train, y_test):
  LOG("Starting NN classifier")
  LOG.log_shapes(X_train, X_test)

  grid_params = {
    'hidden_layer_sizes':[(100,), (300,), (200,200)],
    'alpha' : 10.0 ** -np.arange(1, 7),
    'solver':['sgd'],
    'momentum':[0.9],
    'nesterovs_momentum':[True],
  }

  nn = MLPClassifier(verbose=True)

  clf = GridSearchCV(nn, grid_params, cv=5, n_jobs=4, verbose=True)

  clf.fit(X_train, y_train)

  LOG(clf.cv_results_)
  LOG(clf.best_estimator_)

  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)

  LOG.log_accuracy(accuracy)

def random_forest_clf(X_train, X_test, y_train, y_test):
  print("Starting random forest clf")

  grid_params = {
    'n_estimators':[50,200,500],
    'max_depth':[10,20,30]
  }

  rfc = RandomForestClassifier(n_jobs=4, random_state=777)

  clf = GridSearchCV(rfc, n_jobs=4, verbose=True)

  clf.fit(X_train, X_test)

  print(clf.cv_results_)
  print(clf.best_estimator_)

  y_pred = clf.predict(X_test)
  print('Accuracy', accuracy_score(y_test, y_pred))


def run(pos_path, neg_path):
  LOG.log_paths(pos_path, neg_path)
  X, y = load_data(pos_path, neg_path)

  # scaling the data to 0 mean and 1 variance
  X = preprocessing.scale(X)
  LOG("Scaled data to 0 mean and 1 variance")

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=777, stratify=y)

  #for classifier in [logistic_regression_classifier, svm_classifier, nn_classifier]:
  #  classifier(X_train, X_test, y_train, y_test)
  nn_classifier(X_train, X_test, y_train, y_test)

if __name__=='__main__':
  run('../feature_vectors/pos_feat_vect.bin','../feature_vectors/neg_feat_vect.bin')
