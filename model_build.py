"""
This a submodule on the *Implementation Level* that realises the functions called in the pipeline-function
(:code:`model_build_and_training_pipeline()`) on the *Main Level*.
"""

from math import ceil

import keras
import numpy as np
from keras.models import (Sequential, Model)
from keras.layers import (Input, SimpleRNN, GRU, LSTM, Dense, Bidirectional, Dropout, Conv1D, concatenate)
from keras.callbacks import (EarlyStopping, ModelCheckpoint, CSVLogger)
from keras.metrics import CategoricalAccuracy
from utils import data_generator, data_generator_for_conc, write_model_information_to_file, get_datetime_from_model_dir
from absl import logging as absl_logging


def build_simple_rnn_model(window_len: int, num_indicators: int, num_labels: int,
                           dropout: float, optimizer: str, model_dir: str) -> keras.models.Sequential:
    """
    Builds the neuronal network architecture based on a simple recurrent layer and a fully connected layer. Compiles
    the built model with the loss-function (:code:`categorical_crossentropy`) and the specified optimizer
    (:code:`optimizer`).

    :param window_len: Length of the input data samples
    :param num_indicators: Number of features
    :param num_labels: Number of target labels
    :param dropout: Dropout rate of the recurrent layer
    :param optimizer: Training optimizer method
    :param model_dir: Directory where the model information is saved to
    :return: Compiled model object
    """
    # instantiate model object
    model = Sequential(name='simple_rnn')

    # instantiate and add recurrent input layer with 32 units and integrated dropout according to dropout rate
    model.add(SimpleRNN(units=32, dropout=dropout,
                        input_shape=(window_len, num_indicators), name='rnn_input'))

    # instantiate and add fully connected layer with unit number according to number of target labels
    model.add(Dense(units=num_labels, activation='softmax', name='dense_output'))

    # compile model with categorical crossentropy loss and specified optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[CategoricalAccuracy()])

    # print the model architecture summary
    model.summary()

    # extract date and time from model directory path
    date, dtime = get_datetime_from_model_dir(model_dir=model_dir)

    # write model information to txt-file and save it to model directory
    info_dict = {'Date': date,
                 'Time': dtime,
                 'Model - Name': "simple_rnn",
                 'Model - Loss Function': "categorical_crossentropy",
                 'Model - Optimizer': optimizer,
                 'Model - Metrics': "categorical_accuracy",
                 'First Layer': "rnn_input",
                 'First Layer - Units': "32",
                 'First Layer - Dropout-Rate': str(dropout),
                 'First Layer - Input-Shape': (str(window_len) + ', ' + str(num_indicators)),
                 'Second Layer': "dense_output",
                 'Second Layer - Units': str(num_labels),
                 'Second Layer - Activation': "softmax"}
    write_model_information_to_file(model_dir=model_dir, info_text=info_dict)
    return model


def build_simple_gru_model(window_len: int, num_indicators: int, num_labels: int,
                           dropout: float, optimizer: str, model_dir: str) -> keras.models.Sequential:
    """
    Builds the neuronal network architecture based on a gated recurrent layer and a fully connected layer. Compiles
    the built model with the loss-function (:code:`categorical_crossentropy`) and the specified optimizer
    (:code:`optimizer`).

    :param window_len: Length of the input data samples
    :param num_indicators: Number of features
    :param num_labels: Number of target labels
    :param dropout: Dropout rate of the gated recurrent layer
    :param optimizer: Training optimizer method
    :param model_dir: Directory where the model information is saved to
    :return: Compiled model object
    """
    # instantiate model object
    model = Sequential(name='simple_gru')

    # instantiate and add gated recurrent input layer with 32 units and integrated dropout according to dropout rate
    model.add(GRU(units=32, dropout=dropout,
                  input_shape=(window_len, num_indicators), name='gru_input'))

    # instantiate and add fully connected layer with unit number according to number of target labels
    model.add(Dense(units=num_labels, activation='softmax', name='dense_output'))

    # compile model with categorical crossentropy loss and specified optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[CategoricalAccuracy()])

    # print the model architecture summary
    model.summary()

    # extract date and time from model directory path
    date, dtime = get_datetime_from_model_dir(model_dir=model_dir)

    # write model information to txt-file and save it to model directory
    info_dict = {'Date': date,
                 'Time': dtime,
                 'Model - Name': "simple_gru",
                 'Model - Loss Function': "categorical_crossentropy",
                 'Model - Optimizer': optimizer,
                 'Model - Metrics': "categorical_accuracy",
                 'First Layer': "gru_input",
                 'First Layer - Units': "32",
                 'First Layer - Dropout-Rate': str(dropout),
                 'First Layer - Input-Shape': (str(window_len) + ', ' + str(num_indicators)),
                 'Second Layer': "dense_output",
                 'Second Layer - Units': str(num_labels),
                 'Second Layer - Activation': "softmax"}
    write_model_information_to_file(model_dir=model_dir, info_text=info_dict)
    return model


def build_simple_lstm_model(window_len: int, num_indicators: int, num_labels: int, dropout: float,
                            optimizer: str, model_dir: str) -> keras.models.Sequential:
    """
    Builds the neuronal network architecture based on a long-short-term-memory layer and a fully connected layer.
    Compiles the built model with the loss-function (:code:`categorical_crossentropy`) and the specified
    optimizer (:code:`optimizer`).

    :param window_len: Length of the input data samples
    :param num_indicators: Number of features
    :param num_labels: Number of target labels
    :param dropout: Dropout rate of the long-short-term-memory layer
    :param optimizer: Training optimizer method
    :param model_dir: Directory where the model information is saved to
    :return: Compiled model object
    """
    # instantiate model object
    model = Sequential(name='simple_lstm')

    # instantiate and add long-short-term-memory input layer with 32 units and integrated dropout rate
    model.add(LSTM(units=32, dropout=dropout, input_shape=(window_len, num_indicators), name='lstm_input'))

    # instantiate and add fully connected layer with unit number according to number of target labels
    model.add(Dense(units=num_labels, activation='softmax', name='dense_output'))

    # compile model with categorical crossentropy loss and specified optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[CategoricalAccuracy()])

    # print the model architecture summary
    model.summary()

    # extract date and time from model directory path
    date, dtime = get_datetime_from_model_dir(model_dir=model_dir)

    # write model information to txt-file and save it to model directory
    info_dict = {'Date': date,
                 'Time': dtime,
                 'Model - Name': "simple_lstm",
                 'Model - Loss Function': "categorical_crossentropy",
                 'Model - Optimizer': optimizer,
                 'Model - Metrics': "categorical_accuracy",
                 'First Layer': "lstm_input",
                 'First Layer - Units': "32",
                 'First Layer - Dropout-Rate': str(dropout),
                 'First Layer - Input-Shape': (str(window_len) + ', ' + str(num_indicators)),
                 'Second Layer': "dense_output",
                 'Second Layer - Units': str(num_labels),
                 'Second Layer - Activation-Function': 'softmax'}
    write_model_information_to_file(model_dir=model_dir, info_text=info_dict)
    return model


def build_conv_lstm_model(window_len: int, num_indicators: int, num_labels: int, dropout: float,
                          optimizer: str, model_dir: str) -> keras.models.Sequential:
    """
    Builds the neuronal network architecture based on a convolutional layer, a bidirectional long-short-term-memory
    layer and a fully connected layer. Compiles the built model with the loss-function
    (:code:`categorical_crossentropy`) and the specified optimizer (:code:'optimizer').

    :param window_len: Length of the input data samples
    :param num_indicators: Number of features
    :param num_labels: Number of target labels
    :param dropout: Dropout rate of the long-short-term-memory layer
    :param optimizer: Training optimizer method
    :param model_dir: Directory where the model information is saved to
    :return: Compiled model object
    :return:
    """
    # instantiate model object
    model = Sequential(name='conv_lstm')

    # instantiate and add one-dimensional convolutional input layer with 32 units and kernel size 5
    model.add(Conv1D(filters=32, kernel_size=5,
                     input_shape=(window_len, num_indicators),
                     activation='relu', name='conv_input'))

    # instantiate and add dropout layer according to specified dropout rate
    model.add(Dropout(rate=dropout, name='dropout_1'))

    # instantiate and add bidirectional long-short-term-memory input layer with 32 units and integrated dropout rate
    model.add(Bidirectional(LSTM(units=32, dropout=dropout), name='bidirectional_lstm_1'))

    # instantiate and add fully connected layer with unit number according to number of target labels
    model.add(Dense(units=num_labels, activation='softmax', name='dense_output'))

    # compile model with categorical crossentropy loss and specified optimizer
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[CategoricalAccuracy()])

    # print the model architecture summary
    model.summary()

    # extract date and time from model directory path
    date, dtime = get_datetime_from_model_dir(model_dir=model_dir)

    # write model information to txt-file and save it to model directory
    info_dict = {'Date': date,
                 'Time': dtime,
                 'Model - Name': "conv_lstm",
                 'Model - Loss Function': "categorical_crossentropy",
                 'Model - Optimizer': optimizer,
                 'Model - Metrics': "categorical_accuracy",
                 'First Layer': "conv_input",
                 'First Layer - Filters': "64",
                 'First Layer - Kernel-Size': "5",
                 'First Layer - Input-Shape': (str(window_len) + ', ' + str(num_indicators)),
                 'First Layer - Activation-Function': "relu",
                 'Second Layer': "dropout_1",
                 'Second Layer - Dropout-Rate': str(dropout),
                 'Third Layer': "bidirectional_lstm_1",
                 'Third Layer - Units': "32",
                 'Third Layer - Dropout-Rate': str(dropout),
                 'Fourth Layer': "dense_output",
                 'Fourth Layer - Units': str(num_labels),
                 'Fourth Layer - Activation-Function': 'softmax'}
    write_model_information_to_file(model_dir=model_dir, info_text=info_dict)
    return model


def train_and_save_model_with_gen(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                                  model: keras.models.Sequential, model_path: str, history_path: str,
                                  early_stopping_patience: int = 2, epochs: int = 100, batch_size: int = 128) \
        -> tuple[keras.models.Sequential, keras.callbacks.History]:
    """
    Trains the specified model object (:code:`model`) on the provided training data samples (:code:`x_train`) and
    training labels (:code:`y_train`). Employs generators to provide data samples in batches (:code:`batch_size`) during
    the training process. Evaluates validation data samples (:code:`x_val`) and labels (:code:`y_val`) at the end
    of each training epoch. Restricts total number of training iterations (:code:`epochs`) with
    an early stopping patience (:code:`early_stopping_patience`).

    The best performing trained model is saved to the model directory (:code:`model_path`) and the training history
    information is saved to the history directory (:code:`history_path`).

    :param x_train: Training data samples
    :param y_train: Training labels
    :param x_val: Validation data samples
    :param y_val: Validation labels
    :param model: Model to train
    :param model_path: Directory where trained model is saved to
    :param history_path: Directory where training history information is saved to
    :param early_stopping_patience: Early stopping patience of training process
    :param epochs: Maximum number of training epochs
    :param batch_size: Size of provided data sample batches
    :return: Trained model and history
    """
    # get data generator for training data samples and labels
    gen_train = data_generator(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs)

    # get data generator for validation data samples and labels
    gen_val = data_generator(x=x_val, y=y_val, batch_size=batch_size, epochs=epochs)

    # initialize early stopping mechanism
    early = EarlyStopping(monitor='val_loss', patience=early_stopping_patience)

    # initialize model checkpoint (to save only the model with the lowest validation loss)
    check = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)

    # initialize csv logger to save training history
    csv_logger = CSVLogger(history_path, separator=',', append=False)

    # calculate number of training and validation steps
    steps_train = ceil(len(x_train) / batch_size)
    steps_val = ceil(len(x_val) / batch_size)

    # disable absl INFO and WARNING log messages
    absl_logging.set_verbosity(absl_logging.ERROR)

    # train model with generators and save model with the lowest validation loss
    history = model.fit(gen_train, epochs=epochs, steps_per_epoch=steps_train, validation_data=(gen_val),
                        validation_steps=steps_val, callbacks=[early, check, csv_logger])
    return model, history

