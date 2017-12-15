from sklearn.model_selection import KFold
import numpy as np
import logging

NUMBER_FOLDS = 3


def create_predictions_for_classifier(classifier, X_train, y_train, X_test):
    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    return predictions

def create_folds_predictions(classifiers, X_train, y_train):
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
            predictions = create_predictions_for_classifier(classifier, X_train_fold, y_train_fold, X_test_fold)

            classifiers_fold_predictions[index] = classifiers_fold_predictions[index] + predictions.tolist()

    return np.column_stack(classifiers_fold_predictions), np.array(folds_real_labels)

def compute_score(predictions, labels):
    return (np.count_nonzero(predictions == labels) * 1.0 ) / len(labels)

def create_folds_and_test_predictions(classifiers, X_train, y_train, X_test, y_test):
    classifiers_fold_predictions, folds_real_labels = create_folds_predictions(classifiers, X_train, y_train)

    classifiers_test_predictions = []
    test_real_labels = y_test

    for classifier in classifiers:
        predictions_test = create_predictions_for_classifier(classifier, X_train, y_train, X_test)
        classifiers_test_predictions.append(predictions_test)

        classifier_score = compute_score(predictions_test, y_test)
        classifier_name = classifier.get_name()

        logging.info("Classifier %s accuracy: %f." % (classifier_name, classifier_score))
        print("Classifier %s accuracy: %f." % (classifier_name, classifier_score))

    return classifiers_fold_predictions, folds_real_labels, np.column_stack(classifiers_test_predictions), np.array(test_real_labels)

def stacking_test_accuracy(second_classifier, classifiers, X_train, y_train, X_test, y_test):
    classifiers_fold_predictions, folds_real_labels, classifiers_test_predictions, test_real_labels =\
        create_folds_and_test_predictions(classifiers, X_train, y_train, X_test, y_test)

    second_classifier.fit(classifiers_fold_predictions, folds_real_labels)

    final_train_accuracy = second_classifier.score(classifiers_fold_predictions, folds_real_labels)
    final_test_accuracy = second_classifier.score(classifiers_test_predictions, test_real_labels)

    logging.info("Final stacking train accuracy: " + str(final_train_accuracy))
    logging.info("Final stacking test accuracy: " + str(final_test_accuracy))

    print("Final stacking train accuracy: " + str(final_train_accuracy))
    print("Final stacking test accuracy: " + str(final_test_accuracy))

def stacking_submission(second_classifier, classifiers, X_train, y_train, X_test):
    classifiers_fold_predictions, folds_real_labels = create_folds_predictions(classifiers, X_train, y_train)

    second_classifier.fit(classifiers_fold_predictions, folds_real_labels)

    final_train_accuracy = second_classifier.score(classifiers_fold_predictions, folds_real_labels)

    logging.info("Final stacking train accuracy: " + str(final_train_accuracy))
    print("Final stacking train accuracy: " + str(final_train_accuracy))

    return second_classifier.predict(X_test)
