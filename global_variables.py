"""
This a submodule that specifies all fixed parameters that are used throughout the whole data processing and
model development process. It is accessible from every other module or submodule on every *Level*.
"""

""" VARIABLE DECLARATION """

# data directories
raw_norm_data_dir = 'DataSet MA/Raw Data/Log_MM_Norm.csv'
raw_fault_data_dir = ['DataSet MA/Raw Data/Log_MM_Fault_V118.csv',
                      'DataSet MA/Raw Data/Log_MM_Fault_V112.csv',
                      'DataSet MA/Raw Data/Log_MM_Fault_V113.csv',
                      'DataSet MA/Raw Data/Log_MM_Fault_waterIn.csv']

cleaned_norm_data_dir = 'DataSet MA/Engineered Data/Normal Data/'
cleaned_fault_data_dir = 'DataSet MA/Engineered Data/Faulty Data/'
manipulated_norm_data_dir = 'DataSet MA/Engineered Data/Manipulated Sensor Data/'
concatenated_data_dir = 'DataSet MA/Engineered Data/Concatenated Data/df_concatenated.pkl'

training_x_data_dir = 'DataSet MA/Data/Train/x_train.npy'
training_y_data_dir = 'DataSet MA/Data/Train/y_train.npy'
validation_x_data_dir = 'DataSet MA/Data/Val/x_val.npy'
validation_y_data_dir = 'DataSet MA/Data/Val/y_val.npy'
testing_x_data_dir = 'DataSet MA/Data/Test/x_test.npy'
testing_y_data_dir = 'DataSet MA/Data/Test/y_test.npy'

# data pre-preprocessing variables
sep_symbol = ';'
recorded_sampling_freq = '50ms'
datetime_column_names = ['time_log_systemTime', 'time_trigger', 'time_log_Mode', 'time_mesCounter',
                         'time_systemState',
                         'time_BoilerWaterLevel', 'time_BoilerWaterLevelLow', 'time_BoilerWaterLevelHigh',
                         'time_BoilerTemperatur', 'time_GassingTankWaterLevelLow', 'time_GassingTankCompletelyFilled',
                         'time_PressureTankFront', 'time_PressurePumpB', 'time_Flowmeter', 'time_CurrentPumpB',
                         'time_BoilerHeaterOn', 'time_ValveBoilerToPump', 'time_ValvePumpToGassingTank',
                         'time_PumpIntern',
                         'time_ValveGassingTankToBoiler', 'time_PressAirInTank']
value_column_names = ['value_log_systemTime', 'value_trigger', 'value_log_Mode', 'value_mesCounter',
                      'value_systemState',
                      'value_BoilerWaterLevel', 'value_BoilerWaterLevelLow', 'value_BoilerWaterLevelHigh',
                      'value_BoilerTemperatur', 'value_GassingTankWaterLevelLow', 'value_GassingTankCompletelyFilled',
                      'value_PressureTankFront', 'value_PressurePumpB', 'value_Flowmeter', 'value_CurrentPumpB',
                      'value_BoilerHeaterOn', 'value_ValveBoilerToPump', 'value_ValvePumpToGassingTank',
                      'value_PumpIntern',
                      'value_ValveGassingTankToBoiler', 'value_PressAirInTank']
column_names = ['log_systemTime', 'trigger', 'log_Mode', 'mesCounter', 'systemState', 'BoilerWaterLevel',
                'BoilerWaterLevelLow', 'BoilerWaterLevelHigh',
                'BoilerTemperatur', 'GassingTankWaterLevelLow', 'GassingTankCompletelyFilled', 'PressureTankFront',
                'PressurePumpB', 'Flowmeter', 'CurrentPumpB', 'BoilerHeaterOn', 'ValveBoilerToPump',
                'ValvePumpToGassingTank', 'PumpIntern', 'ValveGassingTankToBoiler', 'PressAirInTank']
columns_to_be_dropped = ['trigger', 'log_Mode', 'mesCounter', 'log_systemTime']
indicator_columns = ['systemState', 'BoilerWaterLevel',
                     'BoilerWaterLevelLow', 'BoilerWaterLevelHigh',
                     'BoilerTemperatur', 'GassingTankWaterLevelLow', 'GassingTankCompletelyFilled', 'PressureTankFront',
                     'PressurePumpB', 'Flowmeter', 'CurrentPumpB', 'BoilerHeaterOn', 'ValveBoilerToPump',
                     'ValvePumpToGassingTank', 'PumpIntern', 'ValveGassingTankToBoiler', 'PressAirInTank']
columns_to_remove_cyclic_behavior = []
columns_to_remove_trend = []
columns_to_perform_outlier_detection = ['PressureTankFront', 'PressurePumpB']

# data engineering variables
column_names_of_sensors_to_manipulate = ['BoilerWaterLevel', 'BoilerTemperatur', 'PressureTankFront', 'PressurePumpB',
                                         'Flowmeter', 'CurrentPumpB']
sensor_fault_labels = ['norm', 'bias', 'trend', 'drift', 'noise']
fault_labels = ['V118', 'V112', 'V113', 'waterIn']
random_state = 42
std_dev_threshold = 300

# data preprocessing variables
fault_type_column_name = 'Fault Type'

# model directories
best_model_dir = 'Trained Models/Best Trained Model/model'
best_history_dir = 'Trained Models/Best Trained Model/history/'
