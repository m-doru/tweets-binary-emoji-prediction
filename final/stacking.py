import logging
import pickle

import numpy as np
from sklearn.model_selection import KFold

NUMBER_FOLDS = 3


def construct_test_predictions_filename(classifiers):
    '''
    Function that constructs the filename that identifies the predictions given by a set of classifiers.
    :param classifiers: the list of classifiers that did the prediction
    :return: the name of the file where we should serialize the predictions of the classifiers
    '''
    filename = 'test_predictions_'

    for classifier in classifiers:
        filename += (str(classifier.get_name()) + '_')

    return filename[-1]


def create_folds_predictions(classifiers, X_train, y_train):
    '''
    Function that will create predictions by each classifier for the training set. For this, we will split
    the training set in k folds, and for each one of the folds, will train all classifiers on the other k-1 folds and
    predict on the current fold. 
    :param classifiers: the classifiers to train on the folds
    :param X_train: the instances of the dataset
    :param y_train: the labels of the dataset
    :return: (predictions, labels), where predictions is a n x p numpy array, where n is the number of instances 
     in the dataset and p is the number of classifiers, and labels is a numpy array of size n x 1, representing the
     true labels. 
    '''
    np.random.seed(7)
    kFold = KFold(n_splits=NUMBER_FOLDS, shuffle=True, random_state=0)

    classifiers_fold_predictions = [[] for _ in range(len(classifiers))]
    folds_real_labels = []

    for train_indices, test_indices in kFold.split(X_train):
        X_train_fold = X_train[train_indices]
        y_train_fold = y_train[train_indices]

        X_test_fold = X_train[test_indices]
        y_test_fold = y_train[test_indices]

        folds_real_labels += y_test_fold.tolist()

        for index, classifier in enumerate(classifiers):
            classifier.fit(X_train_fold, y_train_fold)  # train on train folds
            predictions = classifier.predict(X_test_fold)  # predict on predict fold

            classifiers_fold_predictions[index] = classifiers_fold_predictions[index] + predictions.tolist()
            # append data to predictions list

    print("Fitting classifiers to full dataset")
    for classifier in classifiers:
        classifier.fit(X_train, y_train)

    return np.column_stack(classifiers_fold_predictions), np.array(folds_real_labels)


def create_test_predictions(classifiers, X_test):
    '''
    Function that creates predictions for the test set, using the already trained classifiers. 
    :param classifiers: the classifiers to use for predictions
    :param X_test: the test set
    :return: n x p numpy array, where n is the number of instances in the dataset and p is the number of classifiers,
    representing the predictions of the classifiers on the test instances.
    '''
    classifiers_test_predictions = []

    for classifier in classifiers:
        predictions_test = classifier.predict(X_test)
        classifiers_test_predictions.append(predictions_test)

        logging.info("Created test predictions for " + str(classifier.get_name()))
        print("Created test predictions for " + str(classifier.get_name()))

    return np.column_stack(classifiers_test_predictions)


def stack_classifiers(second_classifier, classifiers, X_train, y_train, X_test):
    '''
    Function that trains the classifiers on the training set and outputs the predictions on the training 
    dataset using folds method (note that predictions are in fact probabilities). Then, we train a second
    tier classifier on top of the predictions of the classifiers, and will predict the output, given
    the predictions of the classifiers in first tier, of the second tier classifier. 
    :param second_classifier: the second tier classifier
    :param classifiers: the first tier classifiers
    :param X_train: the training set instances
    :param y_train: the training set labels
    :param X_test: the test set instances
    :return: the predictions of the second tier classifier on the test set.
    '''
    classifiers_fold_predictions, folds_real_labels = create_folds_predictions(classifiers, X_train, y_train)
    second_classifier.fit(classifiers_fold_predictions, folds_real_labels)

    final_train_accuracy = second_classifier.score(classifiers_fold_predictions, folds_real_labels)

    logging.info("Final stacking train accuracy: " + str(final_train_accuracy))
    print("Final stacking train accuracy: " + str(final_train_accuracy))

    first_tier_classifiers_test_predictions = create_test_predictions(classifiers, X_test)

    with open(construct_test_predictions_filename(classifiers), 'wb') as f:
        pickle.dump(first_tier_classifiers_test_predictions, f)

    return second_classifier.predict(first_tier_classifiers_test_predictions)
