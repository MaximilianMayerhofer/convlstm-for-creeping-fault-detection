### [ConvLSTM for Creeping Fault Detection](https://gitlab.lrz.de/maximilianmayerhofer/convlstm-for-creeping-fault-detection)
***
# Implementation of Creeping Fault Detection Method for Procedural Applications

## Description
This project is developed in the course of a research project which tackles the problem of creeping fault identification and prediction in cyclic and continuous processes of procedural applications.
An exemplary procedural application is utilised to evaluate the developed approach and provide proof of concept.
The required process data is obtained from the exemplary demonstrator plant.

This project implements the processing and data-engineering steps to prepare the acquired process data for the fault identification method. 
Also, a convolutional long-short-term memory network (ConvLSTM) is built, trained, validated, and tested regarding the acquired data. 
This particular artificial neuronal network serves as a creeping fault identification and prediction method.
Along with the ConvLSTM, other validation models are built, trained, validated, and tested for comparison only.

## Structure
```
- DataSet MA
	- Data
		- Train
		- Val
		- Test
	- Engineered Data
		- Concatenated Data
		- Faulty Data
		- Manipulated Sensor Data
		- Normal Data
	- Raw Data
- Trained Models
	- 20230118_093033_0.1_rmsprop_2_100_32
	- Best Trained Model
		- history
		- model
		- indo
		- validation
		- testing
		- weights
- Trained Validation Models
	- 20230118_095707_0.1_rmsprop_2_100_32
	- 20230118_102133_0.1_rmsprop_2_100_32
- README.md
- requirements.txt
- main.py
- global_variables.py
- pipeline.py
- data_pre_preporcessing.py
- data_engineering.py
- data_processing.py
- model_build.py
- model_evaluation.py
- model_analyis.py
- utils.py
- plot.py
```

## Requirements
This project was build using ```Python 3.10```.

All requirements needed for executing the code are listed in ```requirements.txt``` with their required version. Nevertheless, the most important packages are:
- ```Pandas 1.5.1```
- ```NumPy 1.23.4```
- ```Keras 2.10.0```
- ```Scikit-Learn 1.1.3```

## Installation
To install the project, [this project's repository](https://gitlab.lrz.de/maximilianmayerhofer/convlstm-for-creeping-fault-detection) must be cloned and saved to a local machine. 
When the repository is locally stored, it is highly recommended to [create a virtual environment](https://docs.python.org/3/library/venv.html). 
Now, with an active virtual environment [the required packages can be installed](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/). 
Note that a competent IDE like PyCharm does this job automatically as soon as the project repository is opened.

When the virtual environment is set up and all requirements are installed, the project is ready to use.

## Usage
To use the main functionalities of this implemented fault detection concept, the ```main.py``` file must be executed. 
Therein, all major ```pipeline``` functions responsible for pre-preprocessing, data-engineering, preprocessing, model build, training, validation, testing and analysis are called. 
Every processing step can be executed individually by commenting out the remaining ```pipeline```functions. 
For a detailed description of every ```pipeline``` function and sub-function thereof, please refer to the module and function descriptions, as well as the inline code-comments.

When executing the ```main.py``` file, without commenting any ```pipeline``` functions, the data preparation and model development processes are performed. 
This might take a while, depending on the computing power of the local machine.
The command window will show updates on the model's training process, as well as the validation and testing results. 

If new process data is provided, keep in mind to replace the data sets in the ```Raw Data``` directory with new data sets. 
Another option is to change the ```raw_data_dir``` variables in the ```global_variables.py``` module.
The code is particularly written for the process data structure of the exemplary procedural applications. 
Therefore, several adjustments must be made if the new data structure differs from this specific structure. 
A good place to start is the ```global_variables.py``` module, where all data set-specific variables are declared.

## Support
Please report any bug or suggestions for improvement using the [issue tracker](https://gitlab.lrz.de/maximilianmayerhofer/convlstm-for-creeping-fault-detection/-/issues).

## Contributing
Any collaboration is highly appreciated. 
This especially refers to testing, improving, or adapting the code for other creeping fault detection applications. 
Feel free to get in touch.

# Contact
- [LinkedIn](https://www.linkedin.com/in/maximilian-mayerhofer-41804917b/)
- [eMail](mailto:mayerhofermaximilian@gmail.com)

## Authors
I, Maximilian Mayerhofer, am the sole author of this implementation project.

## License
This project is licensed under the terms of the MIT License (c.f. ```LICENSE.txt```).
