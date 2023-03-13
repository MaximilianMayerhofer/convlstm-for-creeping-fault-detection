"""
This a submodule on the *Implementation Level* that realises the functions called in the pipeline-function
(:code:`data_preprocessing_pipeline()`) on the *Main Level*.
"""

import pandas as pd
import numpy as np
from math import ceil
from sklearn.preprocessing import StandardScaler
from global_variables import cleaned_norm_data_dir, fault_labels, cleaned_fault_data_dir, \
    fault_type_column_name, manipulated_norm_data_dir, sensor_fault_labels, column_names_of_sensors_to_manipulate, \
    concatenated_data_dir


def one_hot_encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot-encodes the label columns of the specified dataframe (:code:`df`). All required fixed parameters
    (:code:`fault_type_column_name`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe for which columns are one-hot-encoded
    :return: One-hot-encoded dataframe
    """
    # replace label column by one-hot-encoded label columns
    df_ohe = pd.get_dummies(df, columns=[fault_type_column_name])
    return df_ohe


def separate_dataframe_into_x_and_y_ndarrays(df: pd.DataFrame,
                                             target_vars: list[str],
                                             input_vars: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Splits the specified dataframe (:code:`df`) into an array containing the feature columns' (:code:`input_vars`) values
    and an array containing the label columns' (:code:`target_vars`) values.

    :param df: Dataframe which is split into arrays
    :param target_vars: Label column names
    :param input_vars: Feature column names
    :return: Arrays of feature values and label values
    """

    # if feature columns are not specified
    if input_vars is None:
        # drop label columns from dataframe and write remaining values to feature array
        x = df.drop(columns=target_vars).values
        # write label column values to label array
        y = df[target_vars].values
    # if feature columns are specified
    else:
        # write feature column values to feature array
        x = df[input_vars].values
        #
        y = df[target_vars].values
    return x, y


def scale_x_and_y(x: np.ndarray,
                  y: np.ndarray,
                  scale_y: bool = False) -> tuple[np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    """
    Standardizes the feature array (:code:`x`) and the label array (:code:`y`) (if :code:`scale_y` is :code:`True`) by
    removing mean and scaling to unit variance.

    :param x: Array of feature values
    :param y: Array of label values
    :param scale_y: Boolean whether to standardize label array
    :return: Standardized arrays of feature values and label values and scaler-objects
    """
    # initialize scaler object for feature array
    scaler_x = StandardScaler()

    # standardize feature array
    scaler_x.fit(x)
    x_std = scaler_x.transform(x)

    # initialize scaler object for label array
    scaler_y = StandardScaler()

    # if scale_y is True
    if scale_y:
        # standardize label array
        scaler_y.fit(y)
        y_std = scaler_y.transform(y)
    # if scale_y is not True
    else:
        # do not standardize label array
        y_std = y
    return x_std, y_std, scaler_x, scaler_y


def restructure_x_and_y_to_timeseries_window_array(x: np.ndarray,
                                                   y: np.ndarray,
                                                   window: int,
                                                   same_fault_interval_length,
                                                   number_of_same_fault_interval) -> tuple[np.ndarray, np.ndarray]:
    """
    Restructures the feature value array (:code:`x`)  and the label value array (:code:`y`) to data sample arrays
    according to data sample length (:code:`window`).

    :param x: Array of feature values
    :param y: Array of label values
    :param window: Length of data samples
    :param same_fault_interval_length: Length of one fault case in feature and label arrays
    :param number_of_same_fault_interval: Number of repetitions of same fault case length
    :return: Lists of feature data sample arrays and list of label arrays
    """
    # initialize list of feature data sample arrays and list of feature arrays
    list_of_x_windows, list_of_y_windows = [], []

    # initialize feature data sample array and feature array
    x_ts_window, y_ts_window = [], []

    # if there is only one fault case, respectively the same_fault_interval_length is only a single value
    # TODO: if statement deprecated (was needed in development phase), only else case is in use
    if type(same_fault_interval_length) == int:
        # set iterable value to zero
        iterable = 0
        # while the iterable value is lower than the number_of_same_fault_interval value do:
        while iterable < number_of_same_fault_interval:
            # get the index range in the feature and label array corresponding to fault case
            idx_range = range((same_fault_interval_length * iterable),
                              (same_fault_interval_length * (iterable + 1) - window))
            # iterate over all indexes in the index range
            for idx in idx_range:
                # extract data samples with corresponding window length from the feature array at index position
                x_ts_window.append(x[idx:idx + window])
                # extract label of data sample from the label array at index position
                y_ts_window.append(y[idx])
            # increase iterable value by one
            iterable = iterable + 1
        x_ts_window = np.array(x_ts_window)
        y_ts_window = np.array(y_ts_window)

    # if there are several fault cases, respectively the same_fault_interval_length is a list of values
    elif type(same_fault_interval_length) == list:
        # set the last interval end index to zero
        last_interval_end = 0
        # iterate over all interval lengths in same_fault_interval_length
        for outerIter in range(0, len(same_fault_interval_length)):
            # set iterable value to zero
            iterable = 0
            # while the iterable value is lower than the current number_of_same_fault_interval value do:
            while iterable < number_of_same_fault_interval[outerIter]:
                # get the index range in the feature and label array corresponding to the current fault case
                idx_range = range((last_interval_end + (same_fault_interval_length[outerIter] * iterable)),
                                  (last_interval_end + (same_fault_interval_length[outerIter] * (iterable + 1))
                                   - window))
                # empty the feature data sample array and feature array
                x_ts_window, y_ts_window = [], []
                # iterate over all indexes in the current index range
                for idx in idx_range:
                    # extract data samples with corresponding window length from the feature array at index position
                    x_ts_window.append(x[idx:idx + window])
                    # extract label of data sample from the label array at index position
                    y_ts_window.append(y[idx])
                # append feature data sample array of current fault case to list of feature data sample arrays
                list_of_x_windows.append(np.array(x_ts_window))
                # append label array of current fault case to list of feature data sample arrays
                list_of_y_windows.append(np.array(y_ts_window))
                # increase iterable value by one
                iterable = iterable + 1
            # set last interval end index
            last_interval_end = last_interval_end + (same_fault_interval_length[outerIter] * iterable)
    return list_of_x_windows, list_of_y_windows


def split_list_of_datasets_into_train_val_test(list_of_x_datasets: np.ndarray,
                                               list_of_y_datasets: np.ndarray,
                                               val_test_split: float) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the feature data sample arrays (:code:`list_of_x_datasets`) and the label arrays(:code:`list_of_y_datasets`)
    into training, validation, and testing arrays. Every partition has the same percentage amount of data samples and
    labels from each fault case.

    :param list_of_x_datasets: List of arrays of feature data samples for each fault case
    :param list_of_y_datasets: List of arrays of label values for each fault case
    :param val_test_split: Split ratio of training data samples to validation and testing data samples
    :return: Arrays of features and labels for training, validation, and testing
    """
    # initialize training, validation, and testing arrays
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    x_test = None
    y_test = None
    # iterate over number of values in list_of_x_datasets
    for idx in range(len(list_of_x_datasets)):
        # get indexes of training section in current feature and label array according to split ratio
        train_idx_range = range(0,
                                ceil((1-2*val_test_split)*len(list_of_x_datasets[idx])))
        # get indexes of validation section in current feature and label array according to split ratio
        val_idx_range = range(ceil((1-2*val_test_split)*len(list_of_x_datasets[idx])),
                              ceil((1-val_test_split)*len(list_of_x_datasets[idx])))
        # get indexes of testing section in current feature and label array according to split ratio
        test_idx_range = range(ceil((1-val_test_split)*len(list_of_x_datasets[idx])),
                               len(list_of_x_datasets[idx]))
        # extract training, validation, and testing split from current feature and label array
        x_dataset_train = list_of_x_datasets[idx][train_idx_range]
        y_dataset_train = list_of_y_datasets[idx][train_idx_range]
        x_dataset_val = list_of_x_datasets[idx][val_idx_range]
        y_dataset_val = list_of_y_datasets[idx][val_idx_range]
        x_dataset_test = list_of_x_datasets[idx][test_idx_range]
        y_dataset_test = list_of_y_datasets[idx][test_idx_range]
        # append extracted training data samples and labels to training feature and label array
        if x_train is None and y_train is None:
            x_train = x_dataset_train
            y_train = y_dataset_train
        else:
            x_train = np.append(x_train, x_dataset_train, axis=0)
            y_train = np.append(y_train, y_dataset_train, axis=0)
        # append extracted validation data samples and labels to validation feature and label array
        if x_val is None and y_val is None:
            x_val = x_dataset_val
            y_val = y_dataset_val
        else:
            x_val = np.append(x_val, x_dataset_val, axis=0)
            y_val = np.append(y_val, y_dataset_val, axis=0)
        # append extracted testing data samples and labels to testing feature and label array
        if x_test is None and y_test is None:
            x_test = x_dataset_test
            y_test = y_dataset_test
        else:
            x_test = np.append(x_test, x_dataset_test, axis=0)
            y_test = np.append(y_test, y_dataset_test, axis=0)
    return x_train, y_train, x_val, y_val, x_test, y_test


def preprocess_dataframe_to_timeseries_window_array(df: pd.DataFrame,
                                                    input_vars: list[str],
                                                    target_var: list[str],
                                                    window_length: int,
                                                    dataset_interval_len,
                                                    number_of_datasets) -> tuple[np.ndarray, np.ndarray]:
    """
    Restructures the specified dataframe (:code:`df`) to standardized separate arrays of feature data samples and labels
    for each fault case in the dataframe.

    :param df: Dataframe for which restructuring is performed
    :param input_vars: Feature column names
    :param target_var: Label column names
    :param window_length: Length of data samples
    :param dataset_interval_len: Length of one fault case in dataframe
    :param number_of_datasets: Number of repetitions of same fault case length
    :return: List of feature data sample and label arrays
    """
    # split dataframe into feature value array and label value array
    x_raw, y_raw = separate_dataframe_into_x_and_y_ndarrays(df=df, target_vars=target_var, input_vars=input_vars)

    # standardize feature value arrays by removing mean and scaling to unit variance
    x_std, y_std, x_scaler, y_scaler = scale_x_and_y(x=x_raw, y=y_raw, scale_y=False)

    # restructure feature and label value array to list of data sample arrays for each fault case individually
    list_of_x_ts_windows, list_of_y_ts_windows = \
        restructure_x_and_y_to_timeseries_window_array(x=x_std, y=y_std, window=window_length,
                                                       same_fault_interval_length=dataset_interval_len,
                                                       number_of_same_fault_interval=number_of_datasets)
    return list_of_x_ts_windows, list_of_y_ts_windows


def concatenate_dataframes() -> tuple[pd.DataFrame, list, list]:
    """
    Loads dataframes of all fault cases and concatenates them into one dataframe. All required fixed parameters
    (:code:`cleaned_norm_data_dir, manipulated_norm_data_dir, cleaned_fault_data_dir, sensor_fault_labels,
    column_names_of_sensors_to_manipulate, fault_labels`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :return: Concatenated dataframe along with length and repetitions of fault cases in dataframe
    """
    # initialize list of length of fault cases in dataframe and list of repetitions of one fault case
    number_of_time_steps = []
    number_of_df_repetitions = []

    # load normal operation dataframe
    df_norm_labeled = pd.read_pickle(cleaned_norm_data_dir + 'norm.pkl')

    # get length of normal operation dataframe and write to number_of_time_steps
    number_of_time_steps.append(len(df_norm_labeled))

    # get number of repetitions of normal operation dataframe and write to number_of_df_repetitions
    number_of_df_repetitions.append(4 * len(column_names_of_sensors_to_manipulate) + 1)

    # initialize list of sensor fault dataframes
    list_of_manipulated_labeled_df = []

    # iterate over number of values in column_names_of_sensors_to_manipulate
    for idx in range(len(column_names_of_sensors_to_manipulate)):
        # load bias sensor fault dataframe and append it to list_of_manipulated_labeled_df
        list_of_manipulated_labeled_df.append(pd.read_pickle(manipulated_norm_data_dir +
                                                             column_names_of_sensors_to_manipulate[idx] + '_' +
                                                             sensor_fault_labels[1] + '.pkl'))
        # load trend sensor fault dataframe and append it to list_of_manipulated_labeled_df
        list_of_manipulated_labeled_df.append(pd.read_pickle(manipulated_norm_data_dir +
                                                             column_names_of_sensors_to_manipulate[idx] + '_' +
                                                             sensor_fault_labels[2] + '.pkl'))
        # load drift sensor fault dataframe and append it to list_of_manipulated_labeled_df
        list_of_manipulated_labeled_df.append(pd.read_pickle(manipulated_norm_data_dir +
                                                             column_names_of_sensors_to_manipulate[idx] + '_' +
                                                             sensor_fault_labels[3] + '.pkl'))
        # load noise sensor fault dataframe and append it to list_of_manipulated_labeled_df
        list_of_manipulated_labeled_df.append(pd.read_pickle(manipulated_norm_data_dir +
                                                             column_names_of_sensors_to_manipulate[idx] + '_' +
                                                             sensor_fault_labels[4] + '.pkl'))

    # initialize list of actor fault dataframes
    list_of_faulty_labeled_df_cleaned = []

    # iterate over number of values in fault_labels
    for idx in range(len(fault_labels)):
        # load actor fault dataframe and append it to list_of_faulty_labeled_df_cleaned
        df_fault_labeled = pd.read_pickle(cleaned_fault_data_dir + fault_labels[idx] + '_' + str(idx) + '.pkl')
        list_of_faulty_labeled_df_cleaned.append(df_fault_labeled)
        # get length of actor fault dataframe dataframe and write to number_of_time_steps
        number_of_time_steps.append(len(df_fault_labeled))
        # get number of repetitions of actor fault dataframe and write to number_of_df_repetitions
        number_of_df_repetitions.append(1)

    # concatenate all loaded dataframes
    df_concatenated = df_norm_labeled
    for df_manipulated in list_of_manipulated_labeled_df:
        df_concatenated = pd.concat([df_concatenated, df_manipulated], ignore_index=True)
    for df_faulty_cleaned in list_of_faulty_labeled_df_cleaned:
        df_concatenated = pd.concat([df_concatenated, df_faulty_cleaned], ignore_index=True)

    return df_concatenated, number_of_time_steps, number_of_df_repetitions


def get_variables_and_labels(df: pd.DataFrame = None) -> tuple[list, list]:
    """
    Extracts the feature column names and label column names of the specified dataframe (:code:`df`). If no dataframe
    (:code:`df`) is specified, the feature and label column names of the concatenated dataframe are extracted.
    All required fixed parameters (:code:`concatenated_data_dir, fault_type_column_name`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe from which feature and label column names are extracted
    :return: Extracted feature and label column names
    """
    # initialize list of feature and label column names
    variables = []
    labels = []

    # if no dataframe is specified
    if df is None:
        # load the concatenated dataframe from the concatenated_data_dir
        df = pd.read_pickle(concatenated_data_dir)

    # iterate over all columns in the dataframe
    for col in df:
        # if the fault type column name appears in the current column of the dataframe
        if fault_type_column_name in col:
            # the column name is appended to the label column name list
            labels.append(col)
        # if the fault type column name does not appear in the current column of the dataframe
        else:
            # the column name is appended to the feature column name list
            variables.append(col)
    return variables, labels
