import os
from hashlib import sha1

import numpy as np

from fastText import train_supervised, load_model

SWAP_FILE = os.path.join('.', 'swap_file.txt')
LABEL_IDENTIFIER = '__label__'
SEPARATOR = ' , '


class FastTextClassifier:
    def __init__(self, params, name):
        '''
        Function that instantiates a fasttext classifier. 
        :param params: parameters used for instantiating classifier
        :param name: name of the classifier
        '''
        self.model = None
        self.params = params
        self.name = name
        self.folder = self._get_folder()

    def _get_folder(self):
        '''
        Function that returns the folder in which the classifier will serialize all its weights and models. 
        It depends on the model name and parameters
        '''
        folder_name = self.name + '_'

        for param, value in list(sorted(self.params.items(), key=lambda x: x[0])):
            folder_name += (str(param) + ':' + str(value) + ',')

        return folder_name[:-1]  # don't consider the last comma

    def _get_identifier_for_model_file(self, X, y):
        '''
        Function that constructs a unique identifier filename for each dataset.
        :param X: the set of instances in the dataset
        :param y: the set of labels in the dataset
        :return: the filename identifier for the dataset model.
        '''

        identifier_x = sha1(X).hexdigest()
        identifier_y = sha1(y).hexdigest()

        return identifier_x + identifier_y + '.hdf5'

    def get_name(self):
        '''
        Function that returns the name of the classifier
        :return: the name of the classifier
        '''
        return self.name

    def fit(self, X, y):
        '''
        Method that trains the fasttext clasifier on the provided dataset. If the file with the serialized model 
        of the classifier trained on the exact same dataset exists, then just load the model, otherwise 
        train the classifier and save model.
        :param X: the set of instances in the dataset
        :param y: the set of labels in the dataset
        :return: None
        '''
        file_identifier = self._get_identifier_for_model_file(X, y)
        serialized_file = os.path.join('serialized_models', self.folder, file_identifier)

        if os.path.isfile(serialized_file):
            print("Loading the model from file " + str(serialized_file))
            # file already exists, fit means just loading the model
            self.model = load_model(serialized_file)

        else:
            print("Training the classifier " + str(self.name))
            # means that file does not exist, we have to train the model

            with open(SWAP_FILE, 'w', encoding='utf8') as f:
                for tweet, label in zip(X, y):
                    f.write(LABEL_IDENTIFIER + str(label) + SEPARATOR + tweet.strip() + '\n')

            self.model = train_supervised(SWAP_FILE, **self.params)
            os.remove(SWAP_FILE)  # DELETE SWAP FILE AFTER USAGE

            # lastly, save model. Firstly, create folder for classifier, if it doesn't exist

            os.makedirs(self.folder, exist_ok=True)
            self.model.save_model(serialized_file)

    def predict(self, X):
        '''
        Method that predicts the probabilities for label 1 for each instance in the dataset
        :param X: the instances in the dataset
        :return: the probabilities for label 1, for each instance
        '''
        result = []

        for tweet in X:
            labels, probs = self.model.predict(tweet.strip(), 2)

            if labels[0] == '__label__-1':
                result.append(probs[1])
            else:
                result.append(probs[0])

        return np.array(result)
