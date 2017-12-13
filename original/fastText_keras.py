# '''This example demonstrates the use of fasttext for text classification
# Based on Joulin et al's paper:
# Bags of Tricks for Efficient Text Classification
# https://arxiv.org/abs/1607.01759
# Results on IMDB datasets with uni and bi-gram embeddings:
#     Uni-gram: 0.8813 test accuracy after 5 epochs. 8s/epoch on i7 cpu.
#     Bi-gram : 0.9056 test accuracy after 5 epochs. 2s/epoch on GTx 980M gpu.
# '''
#
from __future__ import print_function
#
import numpy as np
#
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Convolution1D, MaxPooling1D
from keras.layers import Embedding, Flatten
from keras.layers import GlobalAveragePooling1D
#
# import logging
#
# logging.basicConfig(filename='fasttext_keras.log',level=logging.INFO,
#         format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
#
# def create_ngram_set(input_list, ngram_value=2):
#     """
#     Extract a set of n-grams from a list of integers.
#     >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
#     {(4, 9), (4, 1), (1, 4), (9, 4)}
#     >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
#     [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
#     """
#     return set(zip(*[input_list[i:] for i in range(ngram_value)]))
#
#
# def add_ngram(sequences, token_indice, ngram_range=2):
#     """
#     Augment the input list of list (sequences) by appending n-grams values.
#     Example: adding bi-gram
#     >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
#     >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
#     >>> add_ngram(sequences, token_indice, ngram_range=2)
#     [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
#     Example: adding tri-gram
#     >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
#     >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
#     >>> add_ngram(sequences, token_indice, ngram_range=3)
#     [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
#     """
#     new_sequences = []
#     for input_list in sequences:
#         new_list = input_list[:]
#         for i in range(len(new_list) - ngram_range + 1):
#             for ngram_value in range(2, ngram_range + 1):
#                 ngram = tuple(new_list[i:i + ngram_value])
#                 if ngram in token_indice:
#                     new_list.append(token_indice[ngram])
#         new_sequences.append(new_list)
#
#     return new_sequences
#
# def tokenize(X_train, X_test, nb_words):
#     print('\nTokenization of tweets...')
#     if nb_words == None:
#         tokenizer = Tokenizer(filters='')
#     else:
#         tokenizer = Tokenizer(nb_words=nb_words, filters='')
#     tokenizer.fit_on_texts(X_train)
#     word_index = tokenizer.word_index
#     max_features = len(word_index)
#     print('Found %s unique tokens.' % max_features)
#
#     train_sequences = tokenizer.texts_to_sequences(X_train)
#     test_sequences = tokenizer.texts_to_sequences(X_test)
#
#     return train_sequences, test_sequences
#
# def fastText_keras(X_train, y_train, X_test, y_test, **params):
#     ngram_range = params['ngram_range']
#     max_features = params['max_features']
#     maxlen = params['maxlen']
#     batch_size = params['batch_size']
#     embedding_dims = params['embedding_dims']
#     epochs = params['epochs']
#     nb_words=params['nb_words']
#
#     print(len(X_train), 'train sequences')
#     print(len(X_test), 'test sequences')
#     print('Average train sequence length: {}'.format(np.mean(list(map(len, X_train)), dtype=int)))
#     print('Average test sequence length: {}'.format(np.mean(list(map(len, X_test)), dtype=int)))
#
#     X_train, X_test = tokenize(X_train, X_test, nb_words)
#
#     if ngram_range > 1:
#         print('Adding {}-gram features'.format(ngram_range))
#         # Create set of unique n-gram from the training set.
#         ngram_set = set()
#         for input_list in X_train:
#             for i in range(2, ngram_range + 1):
#                 set_of_ngram = create_ngram_set(input_list, ngram_value=i)
#                 ngram_set.update(set_of_ngram)
#
#         # Dictionary mapping n-gram token to a unique integer.
#         # Integer values are greater than max_features in order
#         # to avoid collision with existing features.
#         start_index = max_features + 1
#         token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
#         indice_token = {token_indice[k]: k for k in token_indice}
#
#         # max_features is the highest integer that could be found in the dataset.
#         max_features = np.max(list(indice_token.keys())) + 1
#
#         # Augmenting x_train and x_test with n-grams features
#         x_train = add_ngram(X_train, token_indice, ngram_range)
#         x_test = add_ngram(X_test, token_indice, ngram_range)
#         print('Average train sequence length: {}'.format(np.mean(list(map(len, x_train)), dtype=int)))
#         print('Average test sequence length: {}'.format(np.mean(list(map(len, x_test)), dtype=int)))
#
#     print('Pad sequences (samples x time)')
#     X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#     X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#     print('X_train shape:', X_train.shape)
#     print('X_test shape:', X_test.shape)
#
#     print('Build model...')
#
#     model = Sequential()
#     model.add(Embedding(max_features,embedding_dims,input_length=X_train.shape[1]))
#     model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
#     model.add(MaxPooling1D(pool_length=2))
#     model.add(Flatten())
#     model.add(Dense(250, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#
#     model.fit(X_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
#           validation_data=(X_test, y_test))
#
#     training_metrics = model.evaluate(X_train, y_train)
#     test_metrics = model.evaluate(X_test, y_test)
#
#     logging.info("Training metrics: " + str(training_metrics))
#     logging.info("Test metrics: " + str(test_metrics))
#
#     print("Training metrics: " + str(training_metrics))
#     print("Test metrics: " + str(test_metrics))



###############################

import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import _pickle as cPickle


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for i in range(len(new_list) - ngram_range + 1):
            for ngram_value in range(2, ngram_range + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


def train_test_features(X_train, y_train, X_test, y_test, n_gram=False, pretrained=True, nb_words=None):
    """
    Function that does the cleaning,the tokenization, add the n-grams, build the weight matrix(pretrained)
    depending on the arguments
    Arguments: full (True to use the full dataset)
               n_gram (True to use 2-grams and False to use 1-grams) we didn't use more than 2-grams
               because of the limit of 140 characters in twitter
               pretrained (True to use the glove200 pretraining)
               nb_words (None to take all the words otherwise takes an integer value N which will take the N most frequent words)
    Returns : train_sequences (List of list of word indexes for each tweet for the training)
              test_sequences (List of list of word indexes for each tweet for the test)
              labels (Labels)
              max_features (Maximum word index in the list of list train_sequences)
              embedding_matrix (if pretrained=True returns the embedding_matrix builded by the help of glove)

              This code was inspired from : https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
              and from : https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py

    """
    np.random.seed(0)



    print('\nTokenization of tweets...')
    if nb_words == None:
        tokenizer = Tokenizer(filters='')
    else:
        tokenizer = Tokenizer(nb_words=nb_words, filters='')
    tokenizer.fit_on_texts(X_train)
    word_index = tokenizer.word_index
    max_features = len(word_index)
    print('Found %s unique tokens.' % max_features)

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    maxlen = 30

    if n_gram:
        maxlen = 60
        ngram_range = 2
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in train_sequences:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting X_train and X_test with n-grams features
        train_sequences = add_ngram(train_sequences, token_indice, ngram_range)
        test_sequences = add_ngram(test_sequences, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(np.mean(list(map(len, train_sequences)), dtype=int)))
        print('Average test sequence length: {}'.format(np.mean(list(map(len, test_sequences)), dtype=int)))

    print('Pad sequences (samples x time)')
    train_sequences = sequence.pad_sequences(train_sequences, maxlen=maxlen)
    test_sequences = sequence.pad_sequences(test_sequences, maxlen=maxlen)
    print('train_sequences shape:', train_sequences.shape)
    print('test_sequences shape:', test_sequences.shape)

    if pretrained:

        print('Extracting word vectors from glove200.txt...')
        embeddings_index = {}
        f = open('embeddings/glove200.txt', 'rb')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))

        print('Creating the embedding matrix...')
        embedding_matrix = np.zeros((max_features + 1, 200))
        for word, i in word_index.items():
            if i > max_features:
                continue
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        print('Embedding matrix created.')
        return train_sequences, y_train, test_sequences, y_test, max_features, embedding_matrix
    else:
        return train_sequences, y_train, test_sequences, y_test, max_features


def dumpFeatures(full, n_gram, pretrained, nb_words, namefile):
    """
    Function that dumps the output of the function above : train_test_features()
    Arguments: full (True to use the full dataset)
               n_gram (True to use 2-grams and False to use 1-grams) we didn't use more than 2-grams
               because of the limit of 140 characters in twitter
               pretrained (True to use the glove200 pretraining)
               namefile (the name of the file that will contain the output)
               nb_words (None to take all the words otherwise takes an integer value N which will take the N most frequent words)

    """
    if pretrained:
        train_sequences, labels, test_sequences, max_features, embedding_matrix = train_test_features(full, n_gram,
                                                                                                      pretrained,
                                                                                                      nb_words)
        cPickle.dump([train_sequences, labels, test_sequences, max_features, embedding_matrix], open(namefile, 'wb'))
    else:
        train_sequences, labels, test_sequences, max_features = train_test_features(full, n_gram, pretrained, nb_words)
        cPickle.dump([train_sequences, labels, test_sequences, max_features], open(namefile, 'wb'))
    return

def fastText_keras(X_train, y_train, X_test, y_test, **args):
    X_train, y_train, X_test, y_test, max_features = train_test_features(X_train, y_train, X_test, y_test, n_gram=True, pretrained=False)

    model = Sequential()
    model.add(Embedding(max_features + 1, 20, input_length=X_train.shape[1]))
    model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
    model.add(MaxPooling1D(pool_length=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())


    model.fit(X_train, y_train,
              batch_size=32,
              nb_epoch=5,
              validation_data=(X_test, y_test))

    training_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    print("Training metrics: " + str(training_metrics))
    print("Test metrics: " + str(test_metrics))

    return model