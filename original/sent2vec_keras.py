import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop
from keras import regularizers

from keras_wrapper import KerasModelWrapper

def trained_sent2vec_keras_model():
  np.random.seed(777)
  model = Sequential()

  model.add(Dense(700, activation='relu', input_shape=(X_train.shape[1],),kernel_regularizer=regularizers.l2(0.001)))
  model.add(Dropout(0.2))
  model.add(Dense(700, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(Dropout(0.2))
  model.add(Dense(700, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(Dropout(0.2))
  model.add(Dense(700, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  # Compile model
  model.summary()

  model.compile(loss='binary_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

  return KerasModelWrapper(model, 'sent2vec_trained_keras_model', 128, 2)


