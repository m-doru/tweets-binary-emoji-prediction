# TODO!! Changed the batch size to 256 from 128, might influence a lot

SERIALIZED_MODELS_PATH = 'serialized_models'

params_fastText = {'lr': 0.05,
                   'dim': 30,
                   'ws': 5,
                   'epoch': 12,
                   'minCount': 1,
                   'minCountLabel': 0,
                   'minn': 2,
                   'maxn': 7,
                   'neg': 5,
                   'wordNgrams': 5,
                   'loss': 'softmax',
                   'bucket': 10000000,
                   'thread': 8,
                   'lrUpdateRate': 100,
                   't': 0.0001,
                   'verbose': 2}

params_glove_keras_model_conv = {'id': 'glove_pretrained_keras_conv1D',
                                 'architecture': 'conv',
                                 'batch_size': 256,
                                 'epochs': 2,
                                 'random_state': 777,
                                 'serializations_directory': SERIALIZED_MODELS_PATH
                                 }
params_glove_keras_model_conv_2 = {'id': 'glove_pretrained_keras_conv1D_2',
                                   'architecture': 'conv',
                                   'batch_size': 256,
                                   'epochs': 2,
                                   'random_state': 778,
                                   'serializations_directory': SERIALIZED_MODELS_PATH
                                   }

params_glove_keras_model_conv_3 = {'id': 'glove_pretrained_keras_conv1D_2',
                                   'architecture': 'conv',
                                   'batch_size': 256,
                                   'epochs': 2,
                                   'random_state': 779,
                                   'serializations_directory': SERIALIZED_MODELS_PATH
                                   }

params_glove_keras_model_lstm = {'id': 'glove_pretrained_keras_lstm',
                                 'architecture': 'lstm',
                                 'batch_size': 256,
                                 'epochs': 2,
                                 'random_state': 777,
                                 'serializations_directory': SERIALIZED_MODELS_PATH
                                 }

params_glove_keras_model_lstm_2 = {'id': 'glove_pretrained_keras_lstm_2',
                                   'architecture': 'lstm',
                                   'batch_size': 256,
                                   'epochs': 2,
                                   'random_state': 778,
                                   'serializations_directory': SERIALIZED_MODELS_PATH
                                   }

params_sent2vec_keras_model_dense = {'id': 'sent2vec_pretrained_keras_dense',
                                     'architecture': 'dense',
                                     'batch_size': 256,
                                     'epochs': 2,
                                     'random_state': 776,
                                     'serializations_directory': SERIALIZED_MODELS_PATH
                                     }
