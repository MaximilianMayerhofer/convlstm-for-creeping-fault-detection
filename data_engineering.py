"""
This a submodule on the *Implementation Level* that realises the functions called in the pipeline-function
(:code:`data_engineering_pipeline()`) on the *Main Level*.
"""

import numpy as np
import pandas as pd
import math

from global_variables import manipulated_norm_data_dir, sensor_fault_labels, random_state, \
    column_names_of_sensors_to_manipulate, std_dev_threshold
from utils import plot_dataframe, label_dataset


def add_bias_to_sensor_data(df: pd.DataFrame, target_column: str, bias_value: int):
    """
    Manipulates the specified column (:code:`target_column`) of the dataframe (:code:`df`) to
    the *bias sensor fault case* corresponding to the manipulation summand (:code:`bias_value`).

    :param df: Dataframe for which manipulation is performed
    :param target_column: Target column which is manipulated
    :param bias_value: Summand which is added to target column
    :return: Manipulated dataframe
    """
    # get length of dataframe
    length = df.shape[0]

    # generate list of constant summands with length of dataframe
    bias = np.ones(length)
    bias = bias * bias_value

    # copy dataframe
    df_bias = df.copy(deep=True)

    # add list of summands to target column
    df_bias[target_column] = df[target_column].values + bias
    return df_bias


def add_trend_to_sensor_data(df: pd.DataFrame, target_column: str, trend_end_value: int):
    """
    Manipulates the specified column (:code:`target_column`) of the dataframe (:code:`df`) to the
    *trend sensor fault case* corresponding to the manipulation factor (:code:`trend_end_value`).

    :param df: Dataframe for which manipulation is performed
    :param target_column: Target column which is manipulated
    :param trend_end_value: End value of manipulation-trend
    :return: Manipulated dataframe
    """
    # get length of dataframe
    length = df.shape[0]

    # generate list of linear increasing values from zero to trend_end_value with length of dataframe
    trend = np.linspace(start=0, stop=trend_end_value, num=length)

    # copy dataframe
    df_trend = df.copy(deep=True)

    # add list of linear increasing values to target column
    df_trend[target_column] = df[target_column].values + trend
    return df_trend


def add_drift_to_sensor_data(df: pd.DataFrame, target_column: str, exp: int, drift_end_value: int):
    """
    Manipulates the specified column (:code:`target_column`) of the dataframe (:code:`df`) to the
    *drift sensor fault case* corresponding to the manipulation factors (:code:`drift_end_value, exp`).

    :param df: Dataframe for which manipulation is performed
    :param target_column: Target column which is manipulated
    :param exp: Exponent of non-linear manipulation-drift
    :param drift_end_value: End value of manipulation-drift
    :return: Manipulated dataframe
    """
    # get length of dataframe
    length = df.shape[0]

    # generate list of non-linear increasing values from zero to drift_end_value with length of dataframe
    end_value = math.ceil(np.power(drift_end_value, exp))
    drift = np.power(np.linspace(start=0, stop=end_value, num=length), (1 / exp))

    # copy dataframe
    df_drift = df.copy(deep=True)

    # subtract list of non-linear increasing values from target column
    df_drift[target_column] = df[target_column].values - drift
    return df_drift


def add_noise_to_sensor_data(df: pd.DataFrame, target_column: str, mean: float, std_dev: float):
    """
    Manipulates the specified column (:code:`target_column`) of the dataframe (:code:`df`) to the
    *noise sensor fault case* corresponding to the manipulation factors (:code:`mean, std_dev`).

    :param df: Dataframe for which manipulation is performed
    :param target_column: Target column which is manipulated
    :param mean: Mean of target column (deprecated)
    :param std_dev: Standard deviation of manipulation-noise
    :return: Manipulated dataframe
    """
    # get length of dataframe
    length = df.shape[0]

    # generate list of random noise values corresponding to zero-mean and std_dev with length of dataframe
    rng = np.random.default_rng(random_state)
    noise = rng.normal(0.0, std_dev, length)

    # copy dataframe
    df_noise = df.copy(deep=True)

    # add list of random noise values to target column
    df_noise[target_column] = df[target_column].values + noise
    return df_noise


def get_manipulation_factors(means: list, stds: list, fraction: float):
    """
    Creates lists of manipulation factors for *bias, trend, drift, and noise sensor fault cases* based on the column
    statistics (:code:`means, stds`) of each column which is manipulated. All required fixed parameters
    (:code:`column_names_of_sensors_to_manipulate, std_dev_threshold`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param means: Means of all columns which are manipulated
    :param stds: Standard deviation of all columns which are manipulated
    :param fraction: Fraction value by which manipulation factors are divided
    :return: Lists of manipulation factors
    """
    # generate empty lists for all manipulation factors
    sensor_biases = []
    sensor_trends = []
    sensor_drift_exps = []
    sensor_drifts = []
    sensor_noise_means = []
    sensor_noise_stds = []

    # iterate over all columns that are manipulated specified in column_names_of_sensors_to_manipulate
    for idx in range(len(column_names_of_sensors_to_manipulate)):
        # set bias manipulation summand to half of the column's mean
        sensor_biases.append(means[idx] / (2 * fraction))
        # set trend manipulation value to the column's mean
        sensor_trends.append(means[idx] / fraction)
        # set drift exponent value to two
        sensor_drift_exps.append(2)
        # set drift manipulation value to the column's mean
        sensor_drifts.append(means[idx] / fraction)
        # set noise mean value to column's mean
        sensor_noise_means.append(means[idx] / fraction)
        # set noise standard deviation value to standard deviation of column if it is higher than threshold value
        if (stds[idx] / fraction) > std_dev_threshold:
            sensor_noise_stds.append(stds[idx] / fraction)
        # otherwise set noise standard deviation value threshold value
        else:
            sensor_noise_stds.append(std_dev_threshold)
    return sensor_biases, sensor_trends, sensor_drift_exps, sensor_drifts, sensor_noise_means, sensor_noise_stds


def manipulate_sensor_data(df: pd.DataFrame,
                           target_columns: list,
                           bias_value: list = None,
                           trend_end_value: list = None,
                           drift_exp: list = None, drift_end_value: list = None,
                           noise_mean: list = None, noise_std_dev: list = None):
    """
    Copies and manipulates the normal operation dataframe (:code:`df`) to *bias, trend, drift, and noise sensor fault*
    dataframes for each specified column (:code:`target_columns`) separately. Labels the generated
    *sensor fault* dataframes accordingly. All required fixed parameters
    (:code:`manipulated_norm_data_dir, sensor_fault_labels`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    Every generated *sensor fault* dataframe is saved separately to the sensor fault directory
    (:code:`manipulated_norm_data_dir`).

    :param df: Normal operation dataframe
    :param target_columns: Columns for which each sensor fault case dataset is generated
    :param bias_value: List of bias manipulation summands for each target column
    :param trend_end_value: List of trend manipulation factors for each target column
    :param drift_exp: List of drift exponent factors for each target column
    :param drift_end_value: List of drift manipulation factors for each target column
    :param noise_mean: List of noise manipulation means for each target column
    :param noise_std_dev: List of noise standard deviations for each target column
    :return: None
    """
    # iterate over all columns specified in target_columns
    for idx in range(len(target_columns)):
        if bias_value[idx] is not None:
            # copy normal operation dataframe and manipulate target column to bias sensor fault case
            df_bias = add_bias_to_sensor_data(df=df, target_column=target_columns[idx], bias_value=bias_value[idx])
            # label newly generated dataframe as bias sensor fault
            df_bias = label_dataset(df=df_bias, fault_type=(target_columns[idx] + '_' + sensor_fault_labels[1]))
            # save newly generated bias sensor fault dataframe to .pkl-file
            df_bias.to_pickle(manipulated_norm_data_dir + target_columns[idx] + '_' + sensor_fault_labels[1] + '.pkl')
            # plot dataframe and save plot
            plot_dataframe(df=df_bias, plot_path=manipulated_norm_data_dir,
                           title=(target_columns[idx] + '_' + sensor_fault_labels[1]))

        if trend_end_value[idx] is not None:
            # copy normal operation dataframe and manipulate target column to trend sensor fault case
            df_trend = add_trend_to_sensor_data(df=df, target_column=target_columns[idx],
                                                trend_end_value=trend_end_value[idx])
            # label newly generated dataframe as trend sensor fault
            df_trend = label_dataset(df=df_trend, fault_type=(target_columns[idx] + '_' + sensor_fault_labels[2]))
            # save newly generated trend sensor fault dataframe to .pkl-file
            df_trend.to_pickle(manipulated_norm_data_dir + target_columns[idx] + '_' + sensor_fault_labels[2] + '.pkl')
            # plot dataframe and save plot
            plot_dataframe(df=df_trend, plot_path=manipulated_norm_data_dir,
                           title=(target_columns[idx] + '_' + sensor_fault_labels[2]))

        if drift_exp[idx] is not None and drift_end_value[idx] is not None:
            # copy normal operation dataframe and manipulate target column to drift sensor fault case
            df_drift = add_drift_to_sensor_data(df=df, target_column=target_columns[idx],
                                                exp=drift_exp[idx], drift_end_value=drift_end_value[idx])
            # label newly generated dataframe as drift sensor fault
            df_drift = label_dataset(df=df_drift, fault_type=(target_columns[idx] + '_' + sensor_fault_labels[3]))
            # save newly generated drift sensor fault dataframe to .pkl-file
            df_drift.to_pickle(manipulated_norm_data_dir + target_columns[idx] + '_' + sensor_fault_labels[3] + '.pkl')
            # plot dataframe and save plot
            plot_dataframe(df=df_drift, plot_path=manipulated_norm_data_dir,
                           title=(target_columns[idx] + '_' + sensor_fault_labels[3]))

        if noise_mean[idx] is not None and noise_std_dev[idx] is not None:
            # copy normal operation dataframe and manipulate target column to noise sensor fault case
            df_noise = add_noise_to_sensor_data(df=df, target_column=target_columns[idx],
                                                mean=noise_mean[idx], std_dev=noise_std_dev[idx])
            # label newly generated dataframe as noise sensor fault
            df_noise = label_dataset(df=df_noise, fault_type=(target_columns[idx] + '_' + sensor_fault_labels[4]))
            # save newly generated noise sensor fault dataframe to .pkl-file
            df_noise.to_pickle(manipulated_norm_data_dir + target_columns[idx] + '_' + sensor_fault_labels[4] + '.pkl')
            # plot dataframe and save plot
            plot_dataframe(df=df_noise, plot_path=manipulated_norm_data_dir,
                           title=(target_columns[idx] + '_' + sensor_fault_labels[4]))
