"""
This a submodule on the *Implementation Level* that realises the functions called in the
:code:`model_evaluation_pipeline()` and :code:`model_testing_pipeline()` functions on the *Main Level*.
"""

import time
from os import mkdir

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
import shutil

from sklearn.metrics import classification_report, cohen_kappa_score, matthews_corrcoef, log_loss, confusion_matrix, \
    ConfusionMatrixDisplay
from global_variables import best_model_dir, best_history_dir, indicator_columns
from utils import write_model_information_to_file, make_model_subdir, write_testing_metrics_to_file, \
    write_validation_metrics_to_file


def plot_learning_history(history: pd.DataFrame, history_dir: str, title: str):
    """
    Plots the *accuracy* and *loss* (:code:`history`) of the training and validation data samples over all training
    epochs.

    The plotted figure is saved to the specified history directory (:code:`history_dir`) with its corresponding name
    (:code:`title`).

    :param history: Training history comprising evaluation metrics for each training epoch
    :param history_dir: Directory where figure is saved to
    :param title: Title of plotted figure
    """
    # if a history dataframe is specified
    if history is not None:
        # plot training accuracy and loss, and validation accuracy and loss
        plt.plot(history['categorical_accuracy'])
        plt.plot(history['val_categorical_accuracy'])
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])

        # title plot, label axis, and wire legend
        plt.title(title)
        plt.ylabel('accuracy/ loss')
        plt.xlabel('epoch')
        plt.legend(['train-acc', 'val-acc', 'train-loss', 'val-loss'], loc='upper left')

        # save plotted figure to .pdf-file in the specified history directory
        plt.savefig(history_dir + '_' + title + '.pdf')

        # display the plot
        plt.show()


def validate_model(x_val: np.ndarray, y_val: np.ndarray,
                   model: keras.models.Sequential = None, model_dir: str = None,
                   validation_model: keras.models.Sequential = None, val_model_dir: str = None):
    """
    Evaluates the specified (validation) model (:code:`model, validation_model`) on the validation data samples
    (:code:`x_val`) and validation labels (:code:`y_val`) regarding the *accuracy* and *loss*.

    The evaluation metrics are printed out and saved to an evaluation .txt-file in the validation subdirectory of the
    model directory (:code:`model_dir, val_model_dir`).

    :param x_val: Validation data samples
    :param y_val: Validation labels
    :param model: Model to evaluate
    :param model_dir: Directory of model
    :param validation_model: Validation-Model to evaluate
    :param val_model_dir: Directory of validation-model
    """
    # if a model and model directory are specified
    if (model is not None) and (model_dir is not None):
        # make a validation subdirectory in model directory
        validation_subdir = make_model_subdir(model_directory=model_dir,
                                              sub_dir_title='validation',
                                              sep='/')
        # evaluate model on validation data samples and labels
        score = model.evaluate(x_val, y_val, verbose=0)
        # get loss and accuracy from score-list
        val_loss = score[0]
        val_accuracy = score[1]
        # print model summary and evaluation metrics
        print()
        print('--- VALIDATION - MODEL ---')
        model.summary()
        print()
        print('Model Validation Loss:', val_loss)
        print()
        print('Model Validation Accuracy:', val_accuracy)
        print()
        # write evaluation metrics to .txt-file in validation subdirectory
        write_validation_metrics_to_file(validation_dir=validation_subdir,
                                         val_loss=val_loss,
                                         val_accuracy=val_accuracy)

    # if a validation-model and validation-model directory are specified
    if (validation_model is not None) and (val_model_dir is not None):
        # make a validation subdirectory in validation-model directory
        validation_subdir = make_model_subdir(model_directory=val_model_dir,
                                              sub_dir_title='validation',
                                              sep='/')
        # evaluate validation-model on validation data samples and labels
        validation_score = validation_model.evaluate(x_val, y_val, verbose=0)
        # get loss and accuracy from score-list
        validation_val_loss = validation_score[0]
        validation_val_accuracy = validation_score[1]
        # print validation-model summary and evaluation metrics
        print()
        print('--- VALIDATION - VALIDATION MODEL ---')
        validation_model.summary()
        print()
        print('Validation-Model Validation Loss:', validation_val_loss)
        print()
        print('Validation-Model Validation Accuracy:', validation_val_accuracy)
        print()
        # write evaluation metrics to .txt-file in validation subdirectory
        write_validation_metrics_to_file(validation_dir=validation_subdir,
                                         val_loss=validation_val_loss,
                                         val_accuracy=validation_val_accuracy)


def check_if_model_is_best(x_val: np.ndarray, y_val: np.ndarray,
                           model: keras.models.Sequential, model_dir: str,
                           history: pd.DataFrame, model_history_dir):
    """
    Compares the specified model (:code:`model`) to the best performing model. If specified
    model outperforms currently best performing model, the specified model overwrites the best model
    in the best model directory (:code:`best_model_dir`). All required fixed parameters (:code:`best_model_dir`) are
    specified in `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param x_val: Validation data samples
    :param y_val: Validation labels
    :param model: Model which is compared to currently best performing model
    :param model_dir: Directory of model
    :param history: Training history of specified model
    :param model_history_dir: Directory of training history
    """
    # if a model and model directory are specified
    if (model is not None) and (model_dir is not None):
        # load currently best performing model
        best_model = keras.models.load_model(best_model_dir)
        # try to evaluate currently best performing model on validation data samples and labels
        try:
            best_score = best_model.evaluate(x_val, y_val, verbose=0)
        # catch error if data architecture has changed and best model is deprecated (just for development purpose)
        except ValueError:
            # set the best loss to maximum and the best accuracy to minimum
            best_score = [100, 0]
        # evaluate specified model on validation data samples and labels
        score = model.evaluate(x_val, y_val, verbose=0)


        print()
        print('--- VALIDATION - BEST MODEL ---')
        # if specified model outperforms the best model
        if score[1] > best_score[1]:
            # print model summary and evaluation metrics of specified model
            print('The recently trained Model is the Best Model so far!')
            print()
            model.summary()
            print()
            print('Best-Model Validation Loss:', score[0])
            print()
            print('Best-Model Validation Accuracy:', score[1])
            print()
            # get information subdirectory of specified model directory
            model_info_dir = make_model_subdir(model_directory=model_dir,
                                               sub_dir_title='info',
                                               sep='/')
            # get information subdirectory of currently best performing model directory
            best_model_info_dir = make_model_subdir(model_directory=best_model_dir,
                                                    sub_dir_title='info',
                                                    sep='/')
            # replace previously best performing model with specified model including its subdirectories
            shutil.copy((model_info_dir + '/info.txt'), (best_model_info_dir + '/info.txt'))
            model.save(best_model_dir)
            shutil.copy(model_history_dir, best_history_dir)
            # plot learning history of specified model
            plot_learning_history(history=history, history_dir=(best_history_dir + 'history.log'), title='BEST MODEL')
        # if specified model cannot outperform the best model
        else:
            # print model summary and evaluation metrics of currently best performing model
            print('The recently trained Model is not better than the Best Model!')
            print()
            best_model.summary()
            print()
            print('Best-Model Validation Loss:', best_score[0])
            print()
            print('Best-Model Validation Accuracy:', best_score[1])
            print()


def test_model(x_test: np.ndarray, y_test: np.ndarray, model: keras.models.Sequential, model_dir: str, labels: list):
    """
    Evaluates the specified model (:code:`model`) on the testing data samples (:code:`x_test`) and testing labels
    (:code:`y_test`) regarding the *accuracy*, *loss*, *cohen kappa coefficient*, *matthew correlation coefficient*,
    *performance speed*, and *processing speed*. Moreover the *classification report* and the *confusion matrix* of the
    specified model are generated.

    The testing metrics are printed out and saved to a testing.txt-file in the testing subdirectory of the specified
    model directory (:code:`model_dir`).

    :param x_test: Testing data samples
    :param y_test: Testing labels
    :param model: Model to test
    :param model_dir: Directory of model
    :param labels: Label column names
    """
    # if a model and model directory are specified
    if (model is not None) and (model_dir is not None):
        # make testing subdirectory in specified model directory
        testing_dir = make_model_subdir(model_directory=model_dir, sub_dir_title="testing", sep='/')

        # evaluate model on testing data samples and labels
        score = model.evaluate(x_test, y_test, verbose=0)
        # get loss and accuracy from score-list
        test_loss = score[0]
        test_accuracy = score[1]

        # get the true and predicted class labels
        y_pred_prob = model.predict(x_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=-1)
        y_test = np.argmax(y_test, axis=-1)

        # measure performance time
        single_x_test = x_test[0][None]
        start_perf = time.perf_counter_ns()
        single_y_pred_prob = model.predict(single_x_test, verbose=0)
        single_y_pred = np.argmax(single_y_pred_prob, axis=-1)
        end_perf = time.perf_counter_ns()

        # measure processing time
        start_proc = time.process_time_ns()
        single_y_pred_prob = model.predict(single_x_test, verbose=0)
        single_y_pred = np.argmax(single_y_pred_prob, axis=-1)
        end_proc = time.process_time_ns()

        # print model summary and testing metrics of specified model
        print()
        print('--- TESTING - MODEL ---')
        model.summary()
        print()

        # print classification report, for more information:
        # https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
        print(classification_report(y_test, y_pred, target_names=labels))

        # print accuracy and loss evaluated on testing data samples and labels
        print('Model Test Loss:', test_loss)
        print()
        print('Model Test Accuracy:', test_accuracy)
        print()

        # calculate and print cohen kappa coefficient
        kappa_score = cohen_kappa_score(y_test, y_pred)
        print('Model Cohen Kappa Coefficient:', kappa_score)  # over 0.8 is excellent
        print()

        # calculate and print matthew correlation coefficient
        matthew = matthews_corrcoef(y_test, y_pred)
        print('Model Matthew Correlation Coefficient:', matthew)  # over 0.7 is good
        print()

        # calculate and print logarithmic loss (equals test_loss)
        print('Model Log Loss:', log_loss(y_test, y_pred_prob))  # Lower is better
        print()

        # print performance and processing speed to predict one sample
        print('Performance Time For one Sample:', end_perf - start_perf, ' ns')  # Lower is better
        print()
        print('Processing Time For one Sample:', end_proc-start_proc, ' ns')  # Lower is better
        print()

        # write testing metrics to .txt-file in testing subdirectory
        write_testing_metrics_to_file(testing_dir=testing_dir,
                                      test_loss=test_loss,
                                      test_accuracy=test_accuracy,
                                      kappa_score=kappa_score,
                                      matthew=matthew)

        # plot and save confusion matrix of specified model
        plot_confusion_matrix(y_test=y_test,
                              y_pred=y_pred,
                              path=testing_dir)

    # if no model or no model directory are specified
    else:
        print()
        print('--- TESTING - MODEL ---')
        print('No Model-Directory specified for Testing!')


def plot_confusion_matrix(y_test: np.ndarray, y_pred: np.ndarray, path: str):
    """
    Plots the confusion matrix of the provided true labels (:code:`y_test`) and the predicted labels (:code:`y_test`).

    The plot is saved to the specified directory (:code:`path`).

    :param y_test: Predicted labels
    :param y_pred: True labels
    :param path: Directory where confusion matrix figure is saved to
    """
    # set title if figure
    title = 'Confusion Matrix'

    # generate figure (size optimized for readability)
    fig, ax = plt.subplots(figsize=(16, 12))

    # define printed abbreviations for fault cases to enhance readability
    labels = ['Bias B115', 'Drift B115', 'Noise B115', 'Trend B115',
              'Bias B101', 'Drift B101', 'Noise B101', 'Trend B101',
              'Bias M001', 'Drift M001', 'Noise M001', 'Trend M001',
              'Bias FM', 'Drift FM', 'Noise FM', 'Trend FM',
              'Bias B116', 'Drift B116', 'Noise B116', 'Trend B116',
              'Bias PFT', 'Drift PFT', 'Noise PFT', 'Trend PFT',
              'Flow V112', 'LeakOut V113', 'Flow V118', 'Normal', 'LeakIn BT']

    # generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)
    conf_matrix_plot = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=labels, )

    # plot confusion matrix to figure
    conf_matrix_plot.plot(include_values=True, cmap='Blues',
                          xticks_rotation='vertical', ax=ax, colorbar=True)

    # save figure to directory specified in path
    plt.savefig(path + '/' + title + '.pdf')

    # show confusion matrix figure
    plt.show()
