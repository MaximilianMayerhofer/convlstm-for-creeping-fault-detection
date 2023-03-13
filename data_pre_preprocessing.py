"""
This a submodule on the *Implementation Level* that realises the functions called in the pipeline-functions
(:code:`norm_data_pre_preprocessing_pipeline(), fault_data_pre_preprocessing_pipeline()`) on the *Main Level*.
"""

import pandas as pd
from scipy.stats.mstats import winsorize
from global_variables import datetime_column_names, value_column_names, column_names, sep_symbol, recorded_sampling_freq


def load_dataframe_and_resort_values(data_path: str) -> pd.DataFrame:
    """
    Loads the .csv-file from the specified directory (:code:`data_path`) to a dataframe and maps all columns to the
    recorded sampling rate. All required fixed parameters (:code:`datetime_column_names, value_column_names, column_names`)
    are specified in `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param data_path: Directory of .csv-file where the data set is stored
    :return: Loaded and re-formatted dataframe.
    """
    # load data set from .csv-file specified in data_path to a dataframe
    df = load_dataframe(data_path=data_path)

    # re-formate individual datetime columns to datetime format
    df = reformat_datetime_columns(df=df)

    # get the universal timestamps corresponding to recorded sampling rate
    datetime_index_column = get_datetime_index_range(df=df)

    # map all columns to universal recording frequency and set universal timestamps as index
    df_resorted = append_dt_idx_and_resample(df=df,
                                             datetime_index_column=datetime_index_column)
    # drop individual datetime columns
    df_resorted.drop(datetime_column_names,
                     axis=1,
                     inplace=True)

    # rename feature columns
    for idx in range(len(column_names)):
        df_resorted.rename(columns={value_column_names[idx]: column_names[idx]},
                           inplace=True)

    return df_resorted


def load_dataframe(data_path: str) -> pd.DataFrame:
    """
    Reads the .csv-file from the specified directory (:code:`data_path`) to a dataframe and backward-fills emtpy
    entries. All required fixed parameters (:code:`sep_symbol`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param data_path: Directory of .csv-file where the data set is stored
    :return: Loaded dataframe
    """
    # read data set from .csv-file specified in data_path to a dataframe
    df = pd.read_csv(data_path,
                     sep=sep_symbol,
                     header=0)

    # fill missing entries using backward-fill method
    df.fillna(method='bfill',
              inplace=True)
    return df


def reformat_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-formats all specified columns (:code:`datetime_column_names`) to datetime format.
    All required fixed parameters (:code:`datetime_column_names`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe for which columns are re-formatted
    :return: Re-formatted dataframe
    """
    # iterate over all columns specified in datetime_column_names
    for dt_col in datetime_column_names:
        # re-formate column to datetime column in milliseconds
        df[dt_col] = pd.to_datetime(df[dt_col],
                                    unit='ms')
    return df


def get_datetime_index_range(df: pd.DataFrame) -> pd.DatetimeIndex:
    """
    Generates datetime index based on start and end time of specified dataframe according to the recorded sampling rate.
    All required fixed parameters (:code:`datetime_column_names`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe for which datetime index is generated
    :return: Datetime index
    """
    # get the earliest timestamp of each feature column in dataframe
    earliest_recorded_timestamp = df[datetime_column_names].min()

    # get the latest timestamp of each feature column in dataframe
    latest_recorded_timestamp = df[datetime_column_names].max()

    # get the earliest timestamp of dataframe
    earliest_recorded_timestamp = earliest_recorded_timestamp.min()

    # get the latest timestamp of dataframe
    latest_recorded_timestamp = latest_recorded_timestamp.max()

    # generate datetime index corresponding to the length of the dataframe
    datetime_index_column = pd.date_range(start=earliest_recorded_timestamp,
                                          end=latest_recorded_timestamp,
                                          periods=len(df))
    return datetime_index_column


def append_dt_idx_and_resample(df: pd.DataFrame,
                               datetime_index_column: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Sets index of specified dataframe (:code:`df`) to specified datetime index (:code:`datetime_index_column`) and maps
    all indexes and columns to universal recording sampling rate. All required fixed parameters are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    :param df: Dataframe for which index is replaced
    :param datetime_index_column: Datetime index which is used as new datatime index
    :return: Resampled dataframe
    """

    # set index of dataframe to datetime index  specified in datetime_index_column
    df.set_index(datetime_index_column,
                 inplace=True)

    # resample index and columns to recorded_sampling_frequ
    df_resampled = df.asfreq(freq=recorded_sampling_freq,
                             method='bfill')
    return df_resampled


def cut_off_initial_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cuts off the initial process state based on the value of the :coder:`BoilerWaterLevel` column of the specified
    dataframe (:code:`df`).

    :param df: Dataframe for which initial state is cut off
    :return: Cut dataframe
    """
    # drop all rows in which BoilerWaterLevel-value is equal to zero
    df.drop(df[df.BoilerWaterLevel == 0].index, inplace=True)
    return df


def outlier_detection_and_handling(df: pd.DataFrame,
                                   column: str) -> pd.DataFrame:
    """
    Performs outlier detection and handling for specified column (:code:`column`) in the dataframe (:code:`df`).

    :param df: Dataframe for which outlier detection is performed
    :param column: Column for which outlier detection is performed
    :return: Dataframe with edited column
    """
    # copy specified dataframe to df_out
    df_out = df.copy(deep=True)
    # perform outlier detection and handling with winsorize method
    ma_out_win = winsorize(df_out[column], limits=[0.005, 0.005])
    nda_out_win = ma_out_win.data
    df_out[column] = nda_out_win
    return df_out
