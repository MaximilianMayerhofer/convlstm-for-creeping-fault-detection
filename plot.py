"""
This is the plot module on the *Execution Level* that is executed to plot the dataframes generated during the data
processing. It is solely used to generate plots for the purpose of the associated master thesis.
"""

import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

from global_variables import column_names_of_sensors_to_manipulate, fault_labels, sensor_fault_labels


def plot_df(df: pd.DataFrame, title: str, column: str = None, begin: int = None, end: int = None, ylim: list = None):
    df_plot = df
    if (begin is not None) and (end is not None):
        df_plot = df_plot.iloc[begin:end]
    if column is not None:
        df_plot = df_plot[column]
        title = title + ' - ' + column
    fig, ax = plt.subplots(figsize=(7, 5))
    if ylim is not None:
        df_plot.plot(grid=True, xlabel='Time [h]', ylabel='Deviation from Mean [%]', ax=ax, ylim=ylim)
    else:
        df_plot.plot(grid=True, ax=ax)
    # plt.title(title)
    plt.gca().get_lines()[0].set_color('#0A2D57')
    plt.gca().get_lines()[1].set_color('#5E94D4')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    plt.legend(loc='upper right')
    plt.savefig('DataSet MA/PlotsForThesis/' + title + '.pdf')
    plt.show()
    plt.close()


def percentage_deviation_from_mean(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        mean = df[col].mean()
        df[col] = df[col].sub(mean)
        df[col] = df[col].div(mean)
    return df


def get_y_axis_limits(list_of_df: list, column: str = None, max_y_lim: int = 16000):
    y_lim = [max_y_lim, 0]
    if column is None:
        for df in list_of_df:
            min_values = df[column_names_of_sensors_to_manipulate].min()
            min_val = min_values.min()
            if y_lim[0] > min_val:
                y_lim[0] = min_val
            max_values = df[column_names_of_sensors_to_manipulate].max()
            max_val = max_values.max()
            if y_lim[1] < max_val < max_y_lim:
                y_lim[1] = max_val
    else:
        for df in list_of_df:
            min_val = df[column].min()
            if y_lim[0] > min_val:
                y_lim[0] = min_val
            max_val = df[column].max()
            if y_lim[1] < max_val < max_y_lim:
                y_lim[1] = max_val
    return y_lim


# set data directories
cleaned_norm_data_dir = 'DataSet MA/Engineered Data/Normal Data/norm.pkl'
cleaned_fault_data_dirs = ['DataSet MA/Engineered Data/Faulty Data/V118_0.pkl',
                           'DataSet MA/Engineered Data/Faulty Data/V112_1.pkl',
                           'DataSet MA/Engineered Data/Faulty Data/V113_2.pkl',
                           'DataSet MA/Engineered Data/Faulty Data/waterIn_3.pkl']
cleaned_manipulated_data_dirs = ['DataSet MA/Engineered Data/Manipulated Sensor Data/BoilerTemperatur_bias.pkl',
                                 'DataSet MA/Engineered Data/Manipulated Sensor Data/BoilerTemperatur_trend.pkl',
                                 'DataSet MA/Engineered Data/Manipulated Sensor Data/BoilerTemperatur_drift.pkl',
                                 'DataSet MA/Engineered Data/Manipulated Sensor Data/BoilerTemperatur_noise.pkl']
col_to_be_dropped = ['systemState', 'BoilerWaterLevelLow', 'BoilerWaterLevelHigh', 'PressureTankFront',
                     'PressurePumpB', 'Flowmeter', 'CurrentPumpB',
                     'GassingTankWaterLevelLow', 'GassingTankCompletelyFilled', 'BoilerHeaterOn', 'ValveBoilerToPump',
                     'ValvePumpToGassingTank', 'PumpIntern', 'ValveGassingTankToBoiler', 'PressAirInTank', 'Fault Type']

# get all dfs
list_of_all_dfs = []
df_norm = pd.read_pickle(cleaned_norm_data_dir)
df_norm.drop(col_to_be_dropped, axis=1, inplace=True)
df_norm = percentage_deviation_from_mean(df=df_norm)
# df_norm = scale_variables(df=df_norm,
#                          col='systemState',
#                          scale_fac=1000)
list_of_all_dfs.append(df_norm)
list_of_df_faulty = []
for idx in range(len(cleaned_fault_data_dirs)):
    df_faulty = pd.read_pickle(cleaned_fault_data_dirs[idx])
    df_faulty.drop(col_to_be_dropped, axis=1, inplace=True)
    df_faulty = percentage_deviation_from_mean(df=df_faulty)
    # df_faulty = scale_variables(df=df_faulty,
    #                             col='systemState',
    #                             scale_fac=1000)
    list_of_df_faulty.append(df_faulty)
    list_of_all_dfs.append(df_faulty)
list_of_df_manipulated = []
for idx in range(len(cleaned_manipulated_data_dirs)):
    df_manipulated = pd.read_pickle(cleaned_manipulated_data_dirs[idx])
    df_manipulated.drop(col_to_be_dropped, axis=1, inplace=True)
    df_manipulated = percentage_deviation_from_mean(df=df_manipulated)
    # df_manipulated = scale_variables(df=df_manipulated,
    #                                  col='systemState',
    #                                  scale_fac=1000)
    list_of_df_manipulated.append(df_manipulated)
    list_of_all_dfs.append(df_manipulated)

# set start and stop idxs
start = 4975
stop = 5155

# # plot all dfs, from beginning till end with all columns at once
# y_lim = get_y_axis_limits(list_of_df=list_of_all_dfs)
# plot_df(df=df_norm, title='Normal Data', ylim=y_lim)
# for idx in range(len(list_of_df_faulty)):
#     plot_df(df=list_of_df_faulty[idx], title='Faulty Data ' + fault_labels[idx], ylim=y_lim)

# # plot all dfs, from beginning till end for all numeric columns separately
# for col in column_names_of_sensors_to_manipulate:
#     y_lim = get_y_axis_limits(list_of_df=list_of_all_dfs, column=col)
#     plot_df(df=df_norm, title='Normal Data', column=col, ylim=y_lim)
#     for idx in range(len(list_of_df_faulty)):
#         plot_df(df=list_of_df_faulty[idx], title='Faulty Data ' + fault_labels[idx], column=col, ylim=y_lim)

# plot all dfs, for one cycle with all columns at once
plot_df(df=df_norm, title='Normal Operating Condition', begin=start, end=stop, ylim=[-1.0, 1.0])
for idx in range(len(list_of_df_faulty)):
    plot_df(df=list_of_df_faulty[idx], title='Actor Fault ' + fault_labels[idx],
            begin=start, end=stop, ylim=[-1.0, 1.0])
for idx in range(len(list_of_df_manipulated)):
    plot_df(df=list_of_df_manipulated[idx], title='Sensor Fault BoilerTemperature ' + sensor_fault_labels[idx + 1],
            begin=start, end=stop, ylim=[-1.0, 1.0])

# # plot all dfs, for one cycle for all numeric columns separately
# for col in column_names_of_sensors_to_manipulate:
#     y_lim = get_y_axis_limits(list_of_df=list_of_all_dfs, column=col)
#     plot_df(df=df_norm, title='Normal Data', column=col, begin=start, end=stop, ylim=y_lim)
#     for idx in range(len(list_of_df_faulty)):
#         plot_df(df=list_of_df_faulty[idx], title='Faulty Data ' + fault_labels[idx], column=col, begin=start, end=stop, ylim=y_lim)
