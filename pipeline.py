"""
This a module on the *Main Level* that realises the pipeline-functions called from the :code:`main` module on the
*Execution Level*.
"""

# import packages
import keras
import numpy as np
import pandas as pd

# import from modules
from global_variables import *
from data_pre_preprocessing import load_dataframe_and_resort_values, cut_off_initial_state, \
    outlier_detection_and_handling
from data_engineering import manipulate_sensor_data, get_manipulation_factors
from data_preprocessing import one_hot_encode_labels, preprocess_dataframe_to_timeseries_window_array, \
    concatenate_dataframes, get_variables_and_labels, split_list_of_datasets_into_train_val_test
from model_analysis import analyse_weights
from model_build import train_and_save_model_with_gen, build_conv_lstm_model, \
    build_simple_gru_model, build_simple_lstm_model
from model_evaluation import plot_learning_history, validate_model, check_if_model_is_best, test_model
from utils import get_statistics, create_validation_model_dir, create_model_dir, plot_dataframe, label_dataset, \
    get_cycle_time

# initialise global variables used during runtime
val_model_dir = ''
val_history_dir = ''
model_dir = ''
history_dir = ''


def norm_data_pre_preprocessing_pipeline(target_sampling_freq: str = None):
    """
    Loads the normal operation data set from .csv-file to dataframe and performs pre-preprocessing steps on the obtained
    dataframe. All required fixed parameters
    (:code:`raw_norm_data_dir, columns_to_be_dropped, columns_to_perform_outlier_detection, columns_to_remove_trend,
    columns_to_remove_cyclic_behavior, sensor_fault_labels, cleaned_norm_data_dir`)
    are specified in `global_variables <../global_variables.py>`_.

    The pre-preprocessed dataframe is saved to the normal operation data directory (:code:`cleaned_norm_data_dir`).

    :param target_sampling_freq: The target time step length the original dataframe index is resampled to
    :return: None
    """
    # load the normal operation data set from csv-file to dataframe
    # and map individual timestamps to universal recorded sampling rate
    df_norm = load_dataframe_and_resort_values(data_path=raw_norm_data_dir)

    # get cycle time of normal operation process
    number_of_time_steps_per_norm_cycle, cycle_time_norm = get_cycle_time(df=df_norm)

    # cleanup the norm operation dataframe
    df_norm_cleaned = cut_off_initial_state(df=df_norm)

    # drop columns specified in columns_to_be_dropped
    df_norm_cleaned.drop(columns_to_be_dropped,
                         axis=1,
                         inplace=True)

    # perform outlier detection for every column specified in columns_to_perform_outlier_detection separately
    for col in columns_to_perform_outlier_detection:
        df_norm_cleaned = outlier_detection_and_handling(df=df_norm_cleaned,
                                                         column=col)

    # remove trends for every column specified in columns_to_remove_trend separately
    for col in columns_to_remove_trend:
        df_norm_cleaned[col] = df_norm_cleaned[col].diff(periods=1)
    df_norm_cleaned.fillna(method='bfill')

    # remove cyclic behavior for every column specified in columns_to_remove_cyclic_behavior separately
    for col in columns_to_remove_cyclic_behavior:
        df_norm_cleaned[col] = df_norm_cleaned[col].diff(periods=number_of_time_steps_per_norm_cycle)
    df_norm_cleaned.fillna(method='bfill')

    # increase time step length in normal operation dataframe to target sampling rate
    df_norm_cleaned = df_norm_cleaned.asfreq(freq=target_sampling_freq,
                                             method='pad')

    # label normal operation dataframe
    df_norm_labeled = label_dataset(df=df_norm_cleaned,
                                    fault_type=sensor_fault_labels[0])

    # save pre-preprocessed normal operation dataframe to .pkl-file in cleaned_norm_data_dir
    df_norm_labeled.to_pickle(cleaned_norm_data_dir + 'norm.pkl')

    # plot normal operation dataframe and save plot to cleaned_norm_data_dir
    plot_dataframe(df=df_norm_labeled,
                   plot_path=cleaned_norm_data_dir,
                   title=sensor_fault_labels[0])


def fault_data_pre_preprocessing_pipeline(target_sampling_freq: str = None):
    """
    Loads all actor fault data sets from .csv-files to dataframes and performs pre-preprocessing steps on each
    obtained dataframe. All required fixed parameters
    (:code:`raw_fault_data_dir, columns_to_be_dropped, columns_to_perform_outlier_detection, columns_to_remove_trend,
    columns_to_remove_cyclic_behavior, fault_labels, cleaned_fault_data_dir`)
    are specified in `global_variables <../global_variables.py>`_.

    The pre-preprocessed dataframes are saved to the actor fault directory (:code:`cleaned_fault_data_dir`).

    :param target_sampling_freq: The target time step length the original dataframe index is resampled to
    :return: None
    """
    # load the actor fault data sets from csv-files to dataframes
    # and map individual timestamps to universal recorded sampling rate
    list_of_df_fault = []
    for single_raw_fault_data_dir in raw_fault_data_dir:
        df_fault = load_dataframe_and_resort_values(data_path=single_raw_fault_data_dir)
        list_of_df_fault.append(df_fault)

    # get cycle time of actor fault processes
    number_of_time_steps_per_faulty_cycle = []
    for df_fault in list_of_df_fault:
        steps, cycle_times = get_cycle_time(df=df_fault)
        number_of_time_steps_per_faulty_cycle.append(steps)

    # cleanup the actor fault dataframes
    list_of_df_fault_cleaned = []
    for df_fault in list_of_df_fault:
        df_fault_cleaned = cut_off_initial_state(df=df_fault)
        list_of_df_fault_cleaned.append(df_fault_cleaned)

    # drop columns specified in columns_to_be_dropped
    for df_fault_cleaned in list_of_df_fault_cleaned:
        df_fault_cleaned.drop(columns_to_be_dropped,
                              axis=1,
                              inplace=True)

    # perform outlier detection for every column specified in columns_to_perform_outlier_detection separately
    for idx in range(len(list_of_df_fault_cleaned)):
        for col in columns_to_perform_outlier_detection:
            list_of_df_fault_cleaned[idx] = outlier_detection_and_handling(df=list_of_df_fault_cleaned[idx],
                                                                           column=col)

    # remove trends for every column specified in columns_to_remove_trend separately
    for idx in range(len(list_of_df_fault_cleaned)):
        df_fault_cleaned = list_of_df_fault_cleaned[idx]
        for col in columns_to_remove_trend:
            df_fault_cleaned[col] = df_fault_cleaned[col].diff(periods=1)
        df_fault_cleaned.fillna(method='bfill')
        list_of_df_fault_cleaned[idx] = df_fault_cleaned

    # remove cyclic behavior for every column specified in columns_to_remove_cyclic_behavior separately
    for idx in range(len(list_of_df_fault_cleaned)):
        df_fault_cleaned = list_of_df_fault_cleaned[idx]
        for col in columns_to_remove_cyclic_behavior:
            df_fault_cleaned[col] = df_fault_cleaned[col].diff(periods=number_of_time_steps_per_faulty_cycle[idx])
        df_fault_cleaned.fillna(method='bfill')
        list_of_df_fault_cleaned[idx] = df_fault_cleaned

    # increase time step length in actor fault dataframes to target sampling rate
    for idx in range(len(list_of_df_fault_cleaned)):
        list_of_df_fault_cleaned[idx] = list_of_df_fault_cleaned[idx].asfreq(freq=target_sampling_freq,
                                                                             method='pad')

    # label each actor fault dataframes with corresponding fault label
    list_of_df_fault_labeled = []
    for idx in range(len(list_of_df_fault_cleaned)):
        list_of_df_fault_labeled.append(label_dataset(df=list_of_df_fault_cleaned[idx],
                                                      fault_type=fault_labels[idx]))

    # save pre-preprocessed actor fault dataframes to .pkl-files in cleaned_fault_data_dir
    # plot actor fault dataframes and save plots to cleaned_fault_data_dir
    for idx in range(len(list_of_df_fault_labeled)):
        list_of_df_fault_labeled[idx].to_pickle(cleaned_fault_data_dir + fault_labels[idx] + '_' + str(idx) + '.pkl')
        plot_dataframe(df=list_of_df_fault_labeled[idx],
                       plot_path=cleaned_fault_data_dir,
                       title=(fault_labels[idx] + '_' + str(idx)))


def data_engineering_pipeline(fraction: float):
    """
    Loads the normal operation dataframe and performs manipulation for each sensor fault case.
    All required fixed parameters (:code:`cleaned_norm_data_dir, column_names_of_sensors_to_manipulate`) are specified
    in `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    The manipulated sensor fault dataframes are saved to the sensor fault directory (:code:`manipulated_norm_data_dir`).

    :param fraction: Factor by which all manipulation factors are divided by
    :return: None
    """
    # load normal operation dataframe from cleaned_norm_data_dir
    df_norm_labeled = pd.read_pickle(cleaned_norm_data_dir + 'norm.pkl')

    # get mean and standard derivation from each manipulated feature column in normal operation dataframe
    mean_norm, std_norm = get_statistics(df=df_norm_labeled)

    # get manipulation factors for each sensor fault case dependent on mean and std of manipulated features columns
    sensor_biases, sensor_trends, sensor_drift_exps, sensor_drifts, sensor_noise_means, sensor_noise_stds = \
        get_manipulation_factors(means=mean_norm,
                                 stds=std_norm,
                                 fraction=fraction)

    # manipulate feature columns according to manipulation factors,
    # label data manipulated dataframes and save dataframes to .pkl-files in manipulated_norm_data_dir
    manipulate_sensor_data(df=df_norm_labeled,
                           target_columns=column_names_of_sensors_to_manipulate,
                           bias_value=sensor_biases,
                           trend_end_value=sensor_trends,
                           drift_exp=sensor_drift_exps,
                           drift_end_value=sensor_drifts,
                           noise_mean=sensor_noise_means,
                           noise_std_dev=sensor_noise_stds)


def data_preprocessing_pipeline(window_length: int,
                                val_and_test_size: float):
    """
    Loads the all dataframes, concatenates them and performs preprocessing to training, validation, and testing data
    samples arrays and their corresponding label arrays. All required fixed parameters
    (:code:`concatenated_data_dir, random_state, x_data_dir, y_data_dir`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    The concatenated dataframes are saved to the concatenated data directory (:code:`concatenated_data_dir`). The data
    sample and label arrays are saved to the training, validation, and testing data directories
    (:code:`x_data_dir, y_data_dir`).

    :param window_length: Number of time steps in each data sample
    :param val_and_test_size: Fraction of total amount of data samples to be reserved for validation and testing
    :return: None
    """
    # load all dataframes and concatenate them to one dataframe
    df_concatenated, number_of_time_steps, number_of_df_repetitions = concatenate_dataframes()

    # one hot encode the label column of concatenated dataframe
    df_ohe = one_hot_encode_labels(df=df_concatenated)

    # save concatenated, one-hot-encoded dataframe
    df_ohe.to_pickle(concatenated_data_dir)

    # get feature column names and fault labels
    variables, labels = get_variables_and_labels(df=df_ohe)

    # restructure concatenated dataframe to list of data samples and corresponding labels
    list_of_x_datasets, list_of_y_datasets = \
        preprocess_dataframe_to_timeseries_window_array(df=df_ohe,
                                                        input_vars=variables,
                                                        target_var=labels,
                                                        window_length=window_length,
                                                        dataset_interval_len=number_of_time_steps,
                                                        number_of_datasets=number_of_df_repetitions)

    # split lists of data samples and corresponding labels to training, validation, and testing data samples and labels
    x_train, y_train, x_val, y_val, x_test, y_test = \
        split_list_of_datasets_into_train_val_test(list_of_x_datasets=list_of_x_datasets,
                                                   list_of_y_datasets=list_of_y_datasets,
                                                   val_test_split=val_and_test_size)

    # shuffle training data samples and corresponding labels in the same order
    rng = np.random.default_rng(random_state)
    train_shuffler = rng.permutation(len(x_train), axis=0)
    x_train = x_train[train_shuffler]
    y_train = y_train[train_shuffler]

    # save training, validation, and testing data samples and labels
    np.save(training_x_data_dir, x_train)
    np.save(training_y_data_dir, y_train)
    np.save(validation_x_data_dir, x_val)
    np.save(validation_y_data_dir, y_val)
    np.save(testing_x_data_dir, x_test)
    np.save(testing_y_data_dir, y_test)


def model_build_and_training_pipeline(dropout: float,
                                      esp: int,
                                      epochs: int,
                                      batch_size: int,
                                      optimizer: str,
                                      window_length: int,
                                      val: bool = False):
    """
    Loads the training and validation data sample arrays and their corresponding label arrays. Builds and trains
    the validation model or the ConvLSTM model. All required fixed parameters (:code:`x_data_dir, y_data_dir`)
    are specified in `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    The trained model is saved to the model directory (:code:`model_dir, val_model_dir`).
    The training history is saved to the history directory (:code:`history_dir, val_history_dir`).

    :param dropout: Dropout rate of trained model
    :param esp: Early stopping Patience of training process
    :param epochs: Maximum number of training epochs
    :param batch_size: Size of data sample batches provided during training
    :param optimizer: Optimizer used for training
    :param window_length: Number of time steps in each data sample
    :param val: Boolean value for either building and training the validation model (True) or the ConvLSTM model (False)
    :return: None
    """
    # reference initialised global variables
    global model_dir, history_dir, val_model_dir, val_history_dir

    # get number of features and fault labels
    variables, labels = get_variables_and_labels()
    number_of_labels = len(labels)
    number_of_variables = len(variables)

    # load training and validation data samples and labels
    x_train = np.load(training_x_data_dir)
    y_train = np.load(training_y_data_dir)
    x_val = np.load(validation_x_data_dir)
    y_val = np.load(validation_y_data_dir)

    # if val is True, build and train validation model (GRU or LSTM)
    if val:
        # create validation model directory and validation history directory
        val_model_dir, val_history_dir = create_validation_model_dir(dropout, esp, epochs, batch_size, optimizer)
        print()
        print('--- TRAINING - VALIDATION MODEL ---')
        # build validation model (LSTM) (change function name for other validation model)
        val_model = build_simple_lstm_model(window_len=window_length, num_indicators=number_of_variables,
                                            num_labels=number_of_labels, dropout=dropout,
                                            optimizer=optimizer, model_dir=val_model_dir)
        # val_model = build_simple_gru_model(window_len=window_length, num_indicators=number_of_variables,
        #                                    num_labels=number_of_labels, dropout=dropout,
        #                                    optimizer=optimizer, model_dir=val_model_dir)

        # train validation model with generator
        train_and_save_model_with_gen(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, model=val_model,
                                      model_path=val_model_dir, history_path=val_history_dir,
                                      early_stopping_patience=esp,
                                      epochs=epochs, batch_size=batch_size)
    # if val is False, build and train ConvLSTM model
    else:
        # create ConvLSTM model directory and ConvLSTM history directory
        model_dir, history_dir = create_model_dir(dropout, esp, epochs, batch_size, optimizer)
        print()
        print('--- TRAINING - MODEL ---')
        # build ConvLSTM model
        model = build_conv_lstm_model(window_len=window_length, num_indicators=number_of_variables,
                                      num_labels=number_of_labels, dropout=dropout,
                                      optimizer=optimizer, model_dir=model_dir)
        # train ConvLSTM model with generator
        train_and_save_model_with_gen(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val,
                                      model=model, model_path=model_dir, history_path=history_dir,
                                      early_stopping_patience=esp, epochs=epochs, batch_size=batch_size)


def model_evaluation_pipeline(evaluation_model_dir: str = None,
                              evaluation_history_dir: str = None,
                              val: bool = True):
    """
    Loads model and history from specified directories (:code:`evaluation_model_dir, evaluation_history_dir`). Loads
    validation data samples and arrays. Evaluates the loaded model considering the validation
    data. Checks if evaluated model outperforms currently best performing model. All required fixed parameters
    (:code:`x_data_dir, y_data_dir`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    The evaluation metrics and plots are saved in the validation subdirectory of the model directory
    (:code:`evaluation_model_dir`).

    :param evaluation_model_dir: Directory of model to evaluate
    :param evaluation_history_dir: Directory of training history of model to evaluate
    :param val: Boolean value for either evaluating the validation model (True) or the ConvLSTM model (False)
    :return: None
    """
    # if val is True, evaluate validation model
    if val:
        # if no model directory is specified, take model directory from recently trained validation model
        if evaluation_model_dir is None:
            evaluation_model_dir = val_model_dir
        # if no history directory is specified, take history directory from recently trained validation model
        if evaluation_history_dir is None:
            evaluation_history_dir = val_history_dir
        # load model and training history
        try:
            val_model = keras.models.load_model(evaluation_model_dir)
            val_history = pd.read_csv(evaluation_history_dir, sep=',', engine='python')
        # catch error if no model directory was specified and no validation model was trained in same runtime
        except OSError:
            print('No Validation Model-/ History-Directory specified for Evaluation!')
            val_model = None
            val_history = None
        # plot learning progress and save to directory specified in evaluation_history_dir
        plot_learning_history(history=val_history, history_dir=evaluation_history_dir, title='VALIDATION MODEL')
        # load validation data samples and labels
        x_val = np.load(validation_x_data_dir)
        y_val = np.load(validation_y_data_dir)
        # evaluate model considering the validation data
        validate_model(x_val=x_val, y_val=y_val,
                       validation_model=val_model, val_model_dir=evaluation_model_dir)
    # if val is False, evaluate ConvLSTM model
    else:
        # if no model directory is specified, take model directory from recently trained ConvLSTM model
        if evaluation_model_dir is None:
            evaluation_model_dir = model_dir
        # if no history directory is specified, take history directory from recently trained ConvLSTM model
        if evaluation_history_dir is None:
            evaluation_history_dir = history_dir
        # load model and training history
        try:
            model = keras.models.load_model(evaluation_model_dir)
            history = pd.read_csv(evaluation_history_dir, sep=',', engine='python')
        # catch error if no model directory was specified and no ConvLSTM model was trained in same runtime
        except OSError:
            print('No Model-/ History-Directory specified for Evaluation!')
            model = None
            history = None
        # plot learning progress and save to directory specified in evaluation_history_dir
        plot_learning_history(history=history, history_dir=evaluation_history_dir, title='MODEL')
        # load validation data samples and labels
        x_val = np.load(validation_x_data_dir)
        y_val = np.load(validation_y_data_dir)
        # evaluate model considering the validation data
        validate_model(x_val=x_val, y_val=y_val,
                       model=model, model_dir=evaluation_model_dir)
        # check if evaluated model outperforms currently best performing model
        check_if_model_is_best(x_val=x_val, y_val=y_val, model=model, model_dir=evaluation_model_dir,
                               history=history, model_history_dir=evaluation_history_dir)


def model_testing_pipeline(best_model: bool = False, other_model_dir: str = None):
    """
    Loads either the best model (:code:`best_model`) or a model from the other model directory
    (:code:`other_model_dir`). Loads the testing data samples and labels. Tests the loaded model considering the testing
    data. All required fixed parameters (:code:`x_data_dir, y_data_dir, best_model_dir`) are specified in
    `global_variables </Users/Maxi/fdd-project-ma/global_variables.py>`_.

    The testing metrics and plots are saved in the testing subdirectory of the model directory
    (:code:`other_model_dir`).

    :param best_model: Boolean value for either testing the best performing model (True) or another model (False)
    :param other_model_dir: Directory of another model to test if best_model is False
    """
    # load testing data samples and labels
    x_test = np.load(testing_x_data_dir)
    y_test = np.load(testing_y_data_dir)

    # get feature column names and labels
    variables, labels = get_variables_and_labels()

    # if best_model is True, load model from best_model_dir
    if best_model:
        model = keras.models.load_model(best_model_dir)
        other_model_dir = best_model_dir
    # if best_model is False and another model directory is specified, load model form other_model_dir
    elif best_model is False and other_model_dir is not None:
        model = keras.models.load_model(other_model_dir)
    # if best_model is False and no other model directory is specified, load the recently trained model
    else:
        try:
            model = keras.models.load_model(model_dir)
            other_model_dir = model_dir
        # catch error if no other model directory was specified and no model was trained in same runtime
        except OSError:
            model = None
    # evaluate model considering the testing data
    test_model(x_test=x_test,
               y_test=y_test,
               model=model,
               model_dir=other_model_dir,
               labels=labels)


def model_analysis_pipeline(analysis_model_dir: str):
    """
    Loads model from specified model directory (:code:`analysis_model_dir`). Analyses the loaded model's learnable
    weights.

    The analysis plots are saved in the analysis subdirectory of the specified model directory
    (:code:`analysis_model_dir`).

    :param analysis_model_dir: Directory of model for which weights are analysed
    """
    # load model specified in analysis_model_dir
    model = keras.models.load_model(analysis_model_dir)

    # get feature column names and labels
    variables, labels = get_variables_and_labels()

    # analyse learnable weights of loaded model and plot them
    analyse_weights(model=model, model_dir=analysis_model_dir, indicators=variables, labels=labels)
