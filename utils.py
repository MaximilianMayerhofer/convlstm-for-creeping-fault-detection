"""
This a submodule on the *Lower Level* that realises the functions called on the *Implementation Level*. It is accessible
from every other module or submodule.
"""

import os.path
from os import mkdir
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from global_variables import fault_type_column_name, column_names_of_sensors_to_manipulate


def create_model_dir(var1, var2, var3, var4, var5: str) -> tuple[str, str]:
    """
    Creates a model folder in the *Trained Models* directory. The folder name is based on the date, time, and five
    freely selectable variables (:code:`var1, var2, var3, var4, var5`). Moreover, model and history subdirectories are
    created within the created folder.

    :param var1: Freely selectable variable for directory name
    :param var2: Freely selectable variable for directory name
    :param var3: Freely selectable variable for directory name
    :param var4: Freely selectable variable for directory name
    :param var5: Freely selectable variable for directory name
    :return: Path of created model and history subdirectory
    """
    # get current datatime
    now = datetime.now
    # define temporary directory
    directory = r"Trained Models/{}".format(now().strftime("%Y%m%d_%H%M%S"))
    # create model folder, model subdirectory, and history subdirectory
    model_dir, history_dir = make_subdir_model_and_history(directory=directory,
                                                           var1=var1, var2=var2, var3=var3, var4=var4, var5=var5)
    return model_dir, history_dir


def create_validation_model_dir(var1, var2, var3, var4, var5: str) -> tuple[str, str]:
    """
    Creates a validation-model folder in the *Trained Validation Models* directory. The folder name is based on the
    date, time, and five freely selectable variables (:code:`var1, var2, var3, var4, var5`). Moreover, model and history
    subdirectories are created within the created folder.

    :param var1: Freely selectable variable for directory name
    :param var2: Freely selectable variable for directory name
    :param var3: Freely selectable variable for directory name
    :param var4: Freely selectable variable for directory name
    :param var5: Freely selectable variable for directory name
    :return: Path of created validation-model and history subdirectory
    """
    # get current datatime
    now = datetime.now
    # define temporary directory
    directory = r"Trained Validation Models/{}".format(now().strftime("%Y%m%d_%H%M%S"))
    # create validation-model folder, validation-model subdirectory, and history subdirectory
    model_dir, history_dir = make_subdir_model_and_history(directory=directory,
                                                           var1=var1, var2=var2, var3=var3, var4=var4, var5=var5)
    return model_dir, history_dir


def make_subdir_model_and_history(directory: str, var1, var2, var3, var4, var5: str) -> tuple[str, str]:
    """
    Makes the model folder, the model subdirectory, and history subdirectory in the specified directory
    (:code:`directory`). The folder name is based on the temporary directory name (:code:`directory`) and five freely
    selectable variables (:code:`var1, var2, var3, var4, var5`).

    :param directory: Temporary directory path and name
    :param var1: Freely selectable variable for directory name
    :param var2: Freely selectable variable for directory name
    :param var3: Freely selectable variable for directory name
    :param var4: Freely selectable variable for directory name
    :param var5: Freely selectable variable for directory name
    :return: Path of created model and history subdirectory
    """
    # define directory path and name
    directory = directory + "_" + str(var1) + "_" + var5 + "_" + str(var2) + "_" + str(var3) + "_" + str(var4)

    # create model folder
    mkdir(directory)

    # define model subdirectory within model folder
    model_dir = directory + r"/model"

    # define history subdirectory within model folder
    history_dir = directory + r"/history"

    # create subdirectories
    mkdir(model_dir)
    mkdir(history_dir)

    # append history file-extension to history directory path
    history_dir = history_dir + r"/history.log"
    return model_dir, history_dir


def make_model_subdir(model_directory: str, sub_dir_title: str, sep: str) -> str:
    """
    Makes a subdirectory in the model directory (:code:`model_directory`). The subdirectory name is based on the
    specified subdirectory title (:code:`sub_dir_title`).

    :param model_directory: Directory in which a subdirectory is created
    :param sub_dir_title: Name of the created subdirectory
    :param sep: Separation symbol in directory path
    :return: Path of created subdirectory
    """
    # split model directory path into segments using the specified separation symbol
    parts = model_directory.split(sep)

    # initialize model directory variable
    model_directory = ''

    # iterate over number of segments of directory path
    for idx in range(len(parts)):
        # if the current segment is not the last in the directory path
        if idx != (len(parts)-1):
            # append path segment to directory path
            model_directory = model_directory + parts[idx] + r"/"

    # append subdirectory name to model directory path
    sub_dir = model_directory + sub_dir_title

    # create the new subdirectory with generated path
    try:
        mkdir(sub_dir)

    # catch error of subdirectory already exists
    except FileExistsError:
        return sub_dir
    return sub_dir


def data_generator(x: np.ndarray, y: np.ndarray, batch_size: int = 1, epochs: int = 10):
    """
    Creates and returns data generator object that provides data samples in batches (:code:`batch_size`) during the
    training process. The generator provides the whole data set for a fixed number of iterations (:code:`epoch`).

    :param x: Data sample array
    :param y: Label array
    :param batch_size: Number of data samples provided to model at once
    :param epochs: Maximum number of training epochs
    :return: Generator object
    """
    # iterate over maximum number of epochs
    for epoch in range(epochs):
        # initialize temporary data sample and label array
        x_temp = []
        y_temp = []
        # set counter of batches to zero
        batch_counter = 0
        # get number of data samples in data sample array
        last_val = len(x)
        # create an index range from zero to the number of data samples
        index_range = list(range(last_val))
        # iterate over number of data samples
        for idx in index_range:
            # get data sample and label at current index position
            x_idx = x[idx]
            y_idx = y[idx]
            # append current data samples and label to temporary data sample and label array
            x_temp.append(x_idx)
            y_temp.append(y_idx)
            # increase batch counter by one
            batch_counter += 1
            # if the number of data samples in temporary data samples array equals the batch size
            # or the current data sample is the last one in the data sample array
            if batch_counter == batch_size or index_range[-1] == idx:
                # pass the temporary data sample and label array to the outside calling funtion
                yield np.array(x_temp), np.array(y_temp)
                # reset the temporary data sample and label array
                x_temp = []
                y_temp = []
                # reset the batch counter
                batch_counter = 0


def data_generator_for_conc(x: tuple[np.ndarray, np.ndarray], y: tuple[np.ndarray, np.ndarray],
                            batch_size: int = 1, epochs: int = 10):
    # TODO: deprecated data generator, not in use
    for epoch in range(epochs):
        x_ = []
        y_ = []
        for IDX in range(len(x)):
            x_temp = []
            y_temp = []
            batch_counter = 0
            last_val = len(x[IDX])
            index_range = list(range(last_val))
            for idx in index_range:
                x_idx = x[IDX][idx]
                y_idx = y[IDX][idx]
                x_temp.append(x_idx)
                y_temp.append(y_idx)
                batch_counter += 1
                if batch_counter == batch_size or index_range[-1] == idx:
                    x_.append(np.array(x_temp))
                    y_.append(np.array(y_temp))
                    x_temp = []
                    y_temp = []
                    batch_counter = 0
        yield x_, y_


def get_statistics(df: pd.DataFrame) -> tuple[list, list]:
    """
    Calculate mean and standard deviation of every column in the specified dataframe (:code:`df`) which is also listed
    in :code:`column_names_of_sensors_to_manipulate`. All required fixed parameters
    (:code:`column_names_of_sensors_to_manipulate`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe for which statistics are calculated
    :return: List of means and standard deviations
    """
    # initialize mean and standard deviation lists
    mean = []
    std = []
    # iterate over every column specified in column_names_of_sensors_to_manipulate
    for col in column_names_of_sensors_to_manipulate:
        # calculate mean of current column and append it to list of means
        mean.append(df[col].mean())
        # calculate standard deviation of current column and append it to list of standard deviations
        std.append(df[col].std())
    return mean, std


def plot_dataframe(df: pd.DataFrame, plot_path: str, title: str):
    """
    Plots the first full process cycle in the specified dataframe (:code:`df`). The plot figure is saved to the
    specified directory (:code:`plot_path`) with its corresponding name (:code:`title`).

    :param df: Dataframe for which the first process cycle is plotted
    :param plot_path: Directory where the figure is stored
    :param title: Title of the figure
    """
    # calculate the number of time steps of one cycle and the start and end time of first full process cycle
    number_of_time_steps_per_norm_cycle, cycle_time = get_cycle_time(df=df)

    # scale variable systemState up for better visualization
    df_plot = scale_variables(df=df,
                              col='systemState',
                              scale_fac=1000)

    # cut out the first full process cycle from data frame for plotting
    df_plot = df_plot.loc[cycle_time[0]:cycle_time[1]]

    # plot the first full process cycle (all numeric feature columns)
    df_plot.plot(grid=True)

    # set title of figure
    plt.title(title)

    # save figure to specified plot_path
    plt.savefig(plot_path + title + '.pdf')

    # close the plot figure
    plt.close()


def label_dataset(df: pd.DataFrame, fault_type: str) -> pd.DataFrame:
    """
    Creates label column in specified dataframe (:code:`df`) if it does not already exist. Fills the label column with
    fault labels (:code:`fault_type`). All required fixed parameters (:code:`fault_type_column_name`) are
    specified in `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe to be labelled
    :param fault_type: Label of dataframe
    :return: Dataframe with filled label column.
    """
    # if no label column exits in the specified dataframe
    if fault_type_column_name not in df.columns:
        # create label column and fill with NaN values
        df[fault_type_column_name] = np.nan

    # fill label column with specified fault label
    df.loc[:, fault_type_column_name] = fault_type
    return df


def get_datetime_from_model_dir(model_dir: str) -> tuple[str, str]:
    """
    Extracts the created date and time from the model directory path (:code:`model_dir`).

    :param model_dir: Model directory path
    :return: Creation date and time of directory path
    """
    # split model directory path in segments
    info = model_dir.split(sep='/')

    # split datetime of model directory path into date and time
    info = info[1].split(sep='_')
    date = info[0]
    dtime = info[1]
    return date, dtime


def write_model_information_to_file(info_text: dict, model_dir: str):
    """
    Writes model information (:code:`info_text`) to info.txt-file. Saves the file to the
    information subdirectory in the specified model directory (:code:`model_dir`).

    :param info_text: Model information to be saved in .txt-file
    :param model_dir: Directory of model
    """
    # make information directory/ get information subdirectory path
    model_info_dir = make_model_subdir(model_directory=model_dir,
                                       sub_dir_title='info',
                                       sep='/')

    # append info.txt-file extension to information subdirectory path
    model_info_dir = model_info_dir + "/info.txt"

    # create info.txt-file and write model information specified in info_text to info.txt-file
    with open(model_info_dir, 'w') as f:
        for key, value in info_text.items():
            f.write('%s: %s\n' % (key, value))


def write_testing_metrics_to_file(testing_dir: str, test_loss, test_accuracy, kappa_score, matthew):
    """
    Write testing metrics (:code:`test_loss, test_accuracy, kappa_score, matthew`) to metrics.txt-file. Save file to
    testing subdirectory (:code:`testing_dir`).

    :param testing_dir: Testing subdirectory path
    :param test_loss: Loss metric
    :param test_accuracy: Accuracy metric
    :param kappa_score: Cohen kappa coefficient
    :param matthew: Matthew correlation coefficient
    """
    # append metrics.txt-file extension to testing directory path
    testing_file_dir = testing_dir + '/metrics.txt'

    # create metrics.txt-file and write testing metrics to metrics.txt-file
    with open(testing_file_dir, 'w') as f:
        f.write('Model Test Loss: ' + str(test_loss) + '\n' +
                'Model Test Accuracy: ' + str(test_accuracy) + '\n' +
                'Model Cohen Kappa Coefficient: ' + str(kappa_score) + '\n' +
                'Model Matthew Correlation Coefficient: ' + str(matthew))


def write_validation_metrics_to_file(validation_dir: str, val_loss, val_accuracy):
    """
    Write validation metrics (:code:`val_loss, val_accuracy`) to metrics.txt-file. Save file to
    validation subdirectory (:code:`validation_dir`).

    :param validation_dir: Validation subdirectory path
    :param val_loss: Loss metric
    :param val_accuracy: Accuracy metric
    """
    # append metrics.txt-file extension to validation directory path
    validation_file_dir = validation_dir + '/metrics.txt'

    # create metrics.txt-file and write validation metrics to metrics.txt-file
    with open(validation_file_dir, 'w') as f:
        f.write('Validation Loss: ' + str(val_loss) + '\n' +
                'Validation Accuracy: ' + str(val_accuracy) + '\n')


def scale_variables(df: pd.DataFrame,
                    col: str,
                    scale_fac: int) -> pd.DataFrame:
    """
    Scale the values in the specified column (:code:`col`) of the specified dataframe (:code:`df`) with the scaling
    factor (:code:`scale_fac`).

    :param df: Dataframe for which scaling is performed
    :param col: Column name which is scaled
    :param scale_fac: Factor by which column values are multiplied
    :return: Scaled dataframe
    """
    # multiply all values of the specified column in the specified dataframe with the specified scaling factor
    df[col] = df[col].values * scale_fac
    return df


def get_cycle_time(df: pd.DataFrame) -> tuple[int, tuple]:
    """
    Detects the first full process cycle according to the :code:`systemState`-column in specified dataframe
    (:code:`df`). Calculates and returns the start and end time, and the number of time steps of first full process
    cycle.

    :param df: Dataframe in which full process cycle is detected
    :return: Start and end times, and number of time steps of first full process cycle
    """
    # extract systemState column from dataframe
    system_state = df['systemState']

    # initialize start and end variables
    start = 0
    stop = 0

    # iterate over all rows of systemState column until if condition (see below) is fulfilled
    for idx in range(0, len(system_state)):
        # if condition for the beginning of a process cycle is fulfilled
        if (system_state[idx] == 5 or system_state[idx] == 4 or system_state[idx] == 0) and system_state[idx + 1] == 1:
            # set start variable to index corresponding to begin of a process cycle and exit iteration loop
            start = idx + 1
            break
    # iterate over rows of systemState column from start-index until if condition (see below) is fulfilled
    for idx in range(start, len(system_state)):
        # if condition for the end of a full process cycle is fulfilled
        if (system_state[idx] == 5 or system_state[idx] == 4 or system_state[idx] == 0) and system_state[idx + 1] == 1:
            # set stop variable to index corresponding to end of a process cycle and exit iteration loop
            stop = idx
            break

    # get cycle start and end time form datetime index of dataframe
    cycle_start_time = df.index[start]
    cycle_stop_time = df.index[stop]
    cycle_time = (cycle_start_time, cycle_stop_time)

    # get number of time steps in detected full process cycle
    number_of_time_steps_per_cycle = stop - start

    return number_of_time_steps_per_cycle, cycle_time
