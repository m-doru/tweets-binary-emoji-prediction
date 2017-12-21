# TODO!! Changed the batch size to 256 from 128, might influence a lot

SERIALIZED_MODELS_PATH = 'serialized_models'

params_glove_keras_model_conv = {'id':'glove_pretrained_keras_conv1D',
                                 'architecture':'conv',
                                 'batch_size':256,
                                 'epochs':2,
                                 'random_state':777,
                                 'serializations_directory':SERIALIZED_MODELS_PATH
                                 }
params_glove_keras_model_conv_2 = {'id':'glove_pretrained_keras_conv1D_2',
                                 'architecture':'conv',
                                 'batch_size':256,
                                 'epochs':2,
                                 'random_state':778,
                                 'serializations_directory':SERIALIZED_MODELS_PATH
                                 }

params_glove_keras_model_conv_3 = {'id':'glove_pretrained_keras_conv1D_2',
                                   'architecture':'conv',
                                   'batch_size':256,
                                   'epochs':2,
                                   'random_state':779,
                                   'serializations_directory':SERIALIZED_MODELS_PATH
                                   }

params_glove_keras_model_lstm = {'id':'glove_pretrained_keras_lstm',
                                   'architecture':'lstm',
                                   'batch_size':256,
                                   'epochs':2,
                                   'random_state':777,
                                   'serializations_directory':SERIALIZED_MODELS_PATH
                                   }

params_glove_keras_model_lstm_2 = {'id':'glove_pretrained_keras_lstm_2',
                                 'architecture':'lstm',
                                 'batch_size':256,
                                 'epochs':2,
                                 'random_state':778,
                                 'serializations_directory':SERIALIZED_MODELS_PATH
                                 }

params_sent2vec_keras_model_dense = {'id':'sent2vec_pretrained_keras_dense',
                                   'architecture':'dense',
                                   'batch_size':256,
                                   'epochs':2,
                                   'random_state':776,
                                   'serializations_directory':SERIALIZED_MODELS_PATH
                                   }
