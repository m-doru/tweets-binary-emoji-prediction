import logging
import os

import numpy as np
from keras.layers import Convolution1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle

MODEL_FILE = os.path.join('processed_data', 'glove.twitter.27B.200d.txt')
SERIALIZED_PARAMS_FILE = os.path.join('processed_data', 'serialized', 'all_params.bin')
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = None
EMBEDDING_DIM = 30

def read_pretrained_model():
    global EMBEDDING_DIM
    embeddings_index = {}
    with open(MODEL_FILE) as f:
        coefs = []

        # first_line = True
        for line in f:
            # if first_line: # drop first line
            #     first_line = False
            #     continue

            try:
                values = line.strip().split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except Exception:
                pass

        EMBEDDING_DIM = len(coefs) # update EMBEDDING_DIM depending on input file!

    print('Found %s word vectors.' % len(embeddings_index))
    logging.info('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index

def construct_embedding_matrix(word_index, embeddings_index):
    if MAX_NB_WORDS is None:
        num_words = len(word_index) + 1
    else:
        num_words = min(MAX_NB_WORDS, len(word_index)) + 1 # redundant, already done by tokenizer

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM)) # embedding matrix
    for word, index in word_index.items():
        if ((MAX_NB_WORDS is None) or index < MAX_NB_WORDS) and (word in embeddings_index):
            embedding_matrix[index] = embeddings_index[word]

    return embedding_matrix

def transform_tweets_to_sequences(X, pad=True):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters=' ') # don't delete anything but space
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    logging.info('Found %s unique tokens.' % len(word_index))

    if pad:
        sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return sequences, word_index

def fastText_keras(X_train, y_train, X_test, y_test):
    global EMBEDDING_DIM

    X_total = np.append(X_train, X_test)
    y_total = np.append(y_train, y_test)

    if os.path.isfile(SERIALIZED_PARAMS_FILE):
        embedding_index, X_total, y_total, word_index, embedding_matrix, EMBEDDING_DIM = \
            pickle.load(open(SERIALIZED_PARAMS_FILE, 'rb'))

    else:
        embedding_index = read_pretrained_model()
        X_total, word_index = transform_tweets_to_sequences(X_total, pad=True)
        embedding_matrix = construct_embedding_matrix(word_index, embedding_index)

        pickle.dump((embedding_index, X_total, y_total, word_index, embedding_matrix, EMBEDDING_DIM),
                    open(SERIALIZED_PARAMS_FILE, 'wb'))

    train_size = X_train.shape[0]
    X_train = X_total[:train_size]
    X_test = X_total[train_size:]

    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu')(embedded_sequences)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(250, activation='relu')(x)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=128, nb_epoch=2, validation_data=(X_test, y_test))

    training_accuracy = model.evaluate(X_train, y_train)
    test_accuracy = model.evaluate(X_test, y_test)

    logging.info("Training accuracy: " + str(training_accuracy))
    logging.info("Test accuracy: " + str(test_accuracy))

    print("Training accuracy: " + str(training_accuracy))
    print("Test accuracy: " + str(test_accuracy))

    return model