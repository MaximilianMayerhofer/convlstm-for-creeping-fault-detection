"""
This a submodule on the *Implementation Level* that realises the functions called in the pipeline-function
(:code:`model_analysis_pipeline()`) on the *Main Level*.
"""

from os import mkdir

import numpy as np
import matplotlib.pyplot as plt
import keras

from global_variables import indicator_columns
from utils import make_model_subdir


def analyse_weights(model: keras.models.Sequential, model_dir: str, indicators: list, labels: list):
    """
    Plots the learnable weights of each layer and each unit of the specified model (:code:`model`) to separate plots for
    analysis purpose.

    The figures are saved to subdirectory of the model's directory (:code:`model_dir`).

    :param model: Model object for which weights are plotted
    :param model_dir: Directory of model
    :param indicators: Feature column names
    :param labels: Label column names
    """
    # make subdirectory in model-directory to save plots
    model_weights_dir = make_model_subdir(model_directory=model_dir, sub_dir_title='weights', sep='/')

    # get layers of model
    layers = model.layers

    # iterate over all layers of model
    for layer in layers:
        # get weight array of current layer
        weights = np.array(layer.get_weights())
        # plot weights according to layer type
        if layer.__class__.__name__ == 'Conv1D':
            plot_Conv1D_layer_weights(weights=weights,
                                      model_weights_dir=model_weights_dir,
                                      indicators=indicators)
        elif layer.__class__.__name__ == 'Bidirectional':
            plot_Bidirectional_LSTM_layer_weights(weights=weights,
                                                  model_weights_dir=model_weights_dir)
        elif layer.__class__.__name__ == 'Dense':
            plot_Dense_layer_weights(weights=weights,
                                     model_weights_dir=model_weights_dir,
                                     labels=labels)


def plot_Conv1D_layer_weights(weights, model_weights_dir: str, indicators: list):
    """
    Plots the learnable weights (:code:`weights`) of one-dimensional convolution layer for each filter separately.

    Figures are saved to subdirectory of the weight's directory specified in :code:`model_weights_dir`.

    :param weights: Array of weights of convolution layer
    :param model_weights_dir: Directory of weights in model's directory
    :param indicators: Feature column names
    """
    # make subdirectory in weights-directory to save layer-specific plots
    try:
        model_weights_conv1d_dir = model_weights_dir + '/Conv1D'
        mkdir(model_weights_conv1d_dir)
    except FileExistsError:
        model_weights_conv1d_dir = model_weights_dir + '/Conv1D'

    # plot bias
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[1], 'o')
    plt.title('Conv1D - Bias')
    plt.savefig(model_weights_conv1d_dir + '/bias.pdf')
    plt.close()

    # plot learnable weights for each filter of the one-dimensional convolutional layer
    for idx in range(weights[0].shape[-1]):
        fig, ax = plt.subplots(figsize=(17, 10))
        ax.plot(weights[0][:, :, idx], 'o', label=indicators)
        ax.legend(loc='center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=4)
        plt.title('Conv1D - Weights: Filter #' + str(idx))
        plt.savefig(model_weights_conv1d_dir + '/weights_filter_' + str(idx) + '.pdf')
        plt.close()


def plot_Bidirectional_LSTM_layer_weights(weights, model_weights_dir: str):
    """
    Plots the learnable weights (:code:`weights`) of bidirectional long-short-term-memory layer for each unit and gate
    separately.

    The figures are saved to subdirectory of the weight's directory (code:`model_weights_dir`).

    :param weights: Array of weights of long-short-term-memory layer
    :param model_weights_dir: Directory of weights in model's directory
    """
    # make subdirectory in weights-directory to save layer-specific plots
    try:
        model_weights_bidirectional_dir = model_weights_dir + '/Bidirectional'
        mkdir(model_weights_bidirectional_dir)
    except FileExistsError:
        model_weights_bidirectional_dir = model_weights_dir + '/Bidirectional'

    # get units of LSTM layer of first direction
    units = weights[1].shape[0]

    # plot input bias of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[2][:units], 'o')
    plt.title('Bidirectional LSTM_forward - Input Bias')
    plt.savefig(model_weights_bidirectional_dir + '/forward_input_bias.pdf')
    plt.close()
    # plot forget bias of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[2][units:units*2], 'o')
    plt.title('Bidirectional LSTM_forward - Forget Bias')
    plt.savefig(model_weights_bidirectional_dir + '/forward_forget_bias.pdf')
    plt.close()
    # plot cell state bias of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[2][units*2:units * 3], 'o')
    plt.title('Bidirectional LSTM_forward - Cell State Bias')
    plt.savefig(model_weights_bidirectional_dir + '/forward_cell_state_bias.pdf')
    plt.close()
    # plot output bias of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[2][units * 3:], 'o')
    plt.title('Bidirectional LSTM_forward - Output Bias')
    plt.savefig(model_weights_bidirectional_dir + '/forward_output_bias.pdf')
    plt.close()
    # plot input x-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[0][:, :units], 'o')
    plt.title('Bidirectional LSTM_forward - Input X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_input_x_weights.pdf')
    plt.close()
    # plot forget x-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[0][:, units:units*2], 'o')
    plt.title('Bidirectional LSTM_forward - Forget X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_forget_x_weights.pdf')
    plt.close()
    # plot cell state x-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[0][:, units*2:units*3], 'o')
    plt.title('Bidirectional LSTM_forward - Cell State X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_cell_state_x_weights.pdf')
    plt.close()
    # plot output x-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[0][:, units*3:], 'o')
    plt.title('Bidirectional LSTM_forward - Output X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_output_x_weights.pdf')
    plt.close()
    # plot input h-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[1][:, :units], 'o')
    plt.title('Bidirectional LSTM_forward - Input H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_input_h_weights.pdf')
    plt.close()
    # plot forget x-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[1][:, units:units * 2], 'o')
    plt.title('Bidirectional LSTM_forward - Forget H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_forget_h_weights.pdf')
    plt.close()
    # plot cell state h-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[1][:, units * 2:units * 3], 'o')
    plt.title('Bidirectional LSTM_forward - Cell State H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_cell_state_h_weights.pdf')
    plt.close()
    # plot output h-weights of first direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[1][:, units * 3:], 'o')
    plt.title('Bidirectional LSTM_forward - Output H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/forward_output_h_weights.pdf')
    plt.close()

    # get units of LSTM layer of second direction
    units = weights[4].shape[0]

    # plot input bias of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[5][:units], 'o')
    plt.title('Bidirectional LSTM_backward - Input Bias')
    plt.savefig(model_weights_bidirectional_dir + '/backward_input_bias.pdf')
    plt.close()
    # plot forget bias of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[5][units:units * 2], 'o')
    plt.title('Bidirectional LSTM_backward - Forget Bias')
    plt.savefig(model_weights_bidirectional_dir + '/backward_forget_bias.pdf')
    plt.close()
    # plot cell state bias of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[5][units * 2:units * 3], 'o')
    plt.title('Bidirectional LSTM_backward - Cell State Bias')
    plt.savefig(model_weights_bidirectional_dir + '/backward_cell_state_bias.pdf')
    plt.close()
    # plot output bias of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[5][units * 3:], 'o')
    plt.title('Bidirectional LSTM_backward - Output Bias')
    plt.savefig(model_weights_bidirectional_dir + '/backward_output_bias.pdf')
    plt.close()
    # plot input x-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[3][:, :units], 'o')
    plt.title('Bidirectional LSTM_backward - Input X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_input_x_weights.pdf')
    plt.close()
    # plot forget x-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[3][:, units:units * 2], 'o')
    plt.title('Bidirectional LSTM_backward - Forget X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_forget_x_weights.pdf')
    plt.close()
    # plot cell state x-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[3][:, units * 2:units * 3], 'o')
    plt.title('Bidirectional LSTM_backward - Cell State X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_cell_state_x_weights.pdf')
    plt.close()
    # plot output x-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[3][:, units * 3:], 'o')
    plt.title('Bidirectional LSTM_backward - Output X-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_output_x_weights.pdf')
    plt.close()
    # plot input h-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[4][:, :units], 'o')
    plt.title('Bidirectional LSTM_backward - Input H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_input_h_weights.pdf')
    plt.close()
    # plot forget x-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[4][:, units:units * 2], 'o')
    plt.title('Bidirectional LSTM_backward - Forget H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_forget_h_weights.pdf')
    plt.close()
    # plot cell state h-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[4][:, units * 2:units * 3], 'o')
    plt.title('Bidirectional LSTM_backward - Cell State H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_cell_state_h_weights.pdf')
    plt.close()
    # plot output h-weights of second direction
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[4][:, units * 3:], 'o')
    plt.title('Bidirectional LSTM_backward - Output H-Weights')
    plt.savefig(model_weights_bidirectional_dir + '/backward_output_h_weights.pdf')
    plt.close()


def plot_Dense_layer_weights(weights, model_weights_dir: str, labels: list):
    """
    Plots the learnable weights (:code:`weights`) of fully connected layer.

    The figures are saved to subdirectory of the weight's directory (:code:`model_weights_dir`).

    :param weights: Array of weights of fully connected layer
    :param model_weights_dir: Directory of weights in model's directory
    :param labels: Label column names
    """
    # make subdirectory in weights-directory to save layer-specific plots
    try:
        model_weights_dense_dir = model_weights_dir + '/Dense'
        mkdir(model_weights_dense_dir)
    except FileExistsError:
        model_weights_dense_dir = model_weights_dir + '/Dense'

    # plot bias
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[1], 'o', label=labels)
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=4)
    plt.title('Dense - Bias')
    plt.savefig(model_weights_dense_dir + '/bias.pdf')
    plt.close()

    # plot weights
    fig, ax = plt.subplots(figsize=(17, 10))
    ax.plot(weights[0], 'o', label=labels)
    ax.legend(loc='center', bbox_to_anchor=(0.5, -0.10), shadow=False, ncol=4)
    plt.title('Dense - Weights')
    plt.savefig(model_weights_dense_dir + '/weights.pdf')
    plt.close()
