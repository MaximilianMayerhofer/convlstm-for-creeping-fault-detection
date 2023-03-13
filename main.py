"""
This is the main module on the *Execution Level* that is executed to perform data processing and model development.
It specifies all non-learnable parameters and calls the pipeline-functions implemented on the *Main Level*.
"""

from pipeline import *

""" VARIABLE DECLARATION """

# pre-preprocessing variables
sampling_freq = '1S'

# engineering variables
sensor_manipulation_fraction = 1.0

# preprocessing variables
window_length = 60
val_and_test_size = 0.1

# model-specific variables
dropout = 0.1
optimizer = 'rmsprop'
val_optimizer = 'rmsprop'

# training-specific variables
early_stopping_patience = 2
epochs = 100
batch_size = 32

""" DATA PRE-PREPROCESSING """

# pre-preprocess normal operation data set
norm_data_pre_preprocessing_pipeline(target_sampling_freq=sampling_freq)

# pre-preprocess actor fault data sets
fault_data_pre_preprocessing_pipeline(target_sampling_freq=sampling_freq)

""" DATA ENGINEERING """

# manipulated normal operation data set to obtain sensor fault data sets
data_engineering_pipeline(fraction=sensor_manipulation_fraction)

""" DATA PREPROCESSING """

# preprocess all data sets to obtain data samples
data_preprocessing_pipeline(window_length=window_length,
                            val_and_test_size=val_and_test_size)

""" MODEL BUILD AND TRAINING """

# build and train validation model (GRU or LSTM)
model_build_and_training_pipeline(dropout=dropout,
                                  esp=early_stopping_patience,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  optimizer=val_optimizer,
                                  window_length=window_length,
                                  val=True)

# build and train ConvLSTM model
model_build_and_training_pipeline(dropout=dropout,
                                  esp=early_stopping_patience,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  optimizer=optimizer,
                                  window_length=window_length,
                                  val=False)

""" VALIDATE MODEL """

# validate evaluation model
model_evaluation_pipeline(val=True)

# validate ConvLSTM model
model_evaluation_pipeline(val=False)

""" TEST MODEL"""

# test best performing ConvLSTM model
model_testing_pipeline(best_model=True)

# test specified model
model_testing_pipeline(best_model=False,
                       other_model_dir='Trained Validation Models/20230118_095707_0.1_rmsprop_2_100_32/model')

""" ANALYSE MODEL """

# analyse weights of specified model
model_analysis_pipeline(analysis_model_dir='Trained Models/Best Trained Model/model')
