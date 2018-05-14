# Predict Saturated Thickness using TensorBoard Visualization
Water plays a critical role in our living and manufacturing activities. The continuously growing exploitation of water over the aquifer poses a risk for over-extraction and pollution, leading to many negative effects on land irrigation. Therefore, predicting aquifer water level accurately is urgently important, which can help us prepare water demands ahead of time. In this study, we employ the Long-Short Term Memory (LSTM) model to predict the saturated thickness of an aquifer in the Southern High Plains Aquifer System in Texas, and exploit TensorBoard as a guide for model configurations. The Root Mean Squared Error of this study shows that the LSTM model can provide a good prediction capability using multiple data sources, and provides a good visualization tool to help us understand and evaluate the model configuration. 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites
In this documentation, we demonstrate how to run TensorFlow and TensorBoard on MacOS, but it supposes to work on both Windows and Ubuntu.
* **[python3](https://www.python.org/download/releases/3.0/ "Python's Homepage") and [pip3](https://pip.pypa.io/en/stable/installing/ "Pip's homepage")** 

First, install TensorFlow from Terminal
```
$pip3 install tensorflow
```
Second, to make sure that TensorFlow and TensorBoard are successfully installed on your local machine. Test them with the following codes in the Terminal command.
```
$python3
>>>import tensorflow as tf
>>>a = tf.constant(3)
>>>b = tf.constant(6)
>>>sum = tf.add(a,b)
>>>with tf.Session() as sess:
...   writerlog = tf.summary.FileWriter('./logpath', sess.graph)
...   print(sess.run(sum))
>>>
```
After you hit Enter, operations are written to the event file in the logpath folder. Now we can initialize TensorBoard by typing the followig command
```
$tensorboard --logdir logpath
```
It gives the result: TensorBoard 1.8.0 at http://Vinhs-MacBook-Air.local:6006 (Press CTRL+C to quit).....pay attention to the last four digits only: the port number 6006.

Then open any web browser and type: localhost:6006 on the address bar. If you find this graph below then congratulations !

![vinh.nguyen@ttu.edu](/figures/First_demo.png)

The approach above is straightforward but the main drawback is that it's difficult to maintain and debug the code. Our suggestion is to use Jupyter notebook in [Anaconda](https://anaconda.org/). Please download and install the version for python 3.x

Open Terminal to create tensorflow for Anaconda environment first

```
$ conda create -n tensorflow pip python=3.6.4 //Check your python version by typing $python3 -v
$ source activate tensorflow //Activate the conda environment 
$ pip3 install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl // install TensorFlow inside your conda environment:
```
Open Anaconda Navigator -> Environment to check if everything is installed correctly

![vinh.nguyen@ttu.edu](/figures/Anaconda_Environment.png)

Then start the Jupyter notebook. Remember to select the "Applications on" as tensorflow rather than base (root)

![vinh.nguyen@ttu.edu](/figures/Run_Jupyter.png)

Start python 3

![vinh.nguyen@ttu.edu](/figures/Jupyter_Python3.png)

Validate our example again
![vinh.nguyen@ttu.edu](/figures/Run_first_example.png)

Running TensorBoard is similar to the previous step. Open Terminal and issue the following command:
```
$tensorboard --logdir logpath
```
### Install dependencies for Jupyter notebook
Please follow the instructions below to install all required libraries in our demonstration. Open the Terminal from Anaconda Environment >> Open with Terminal
![vinh.nguyen@ttu.edu](/figures/Dependency.png)

and type the following commands:

```
(tensorflow) bash-3.2$ conda install panda
(tensorflow) bash-3.2$ conda install scikit-learn
(tensorflow) bash-3.2$ conda install keras
(tensorflow) bash-3.2$ conda install matplotlib 
```
## Application
### Import required libraries
```
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.callbacks import TensorBoard
from time import time
from keras import backend as K
K.clear_session()
```
### Define function
```
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
```
### Processing dataset
```
# load dataset
dataset = read_csv('Processed_%s.csv'%datasetname, index_col=0)
raw_values = dataset.values 
# ensure all data is float
raw_values = raw_values.astype('float32')
diff_values = np.diff(raw_values, axis=0)
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(raw_values)
# frame as supervised learning

reframed = series_to_supervised(scaled, 1, 1)
reframed.drop(reframed.columns[[4,5]], axis=1, inplace=True)
values = reframed.values
train_ratio =0.85
train_index  = int(len(values)*train_ratio)
train_X, train_y = values[:train_index,:-1], values[:train_index,-1]
test_X, test_y = values[train_index:,:-1], values[train_index:,-1]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
```

### Define LSTM model
```
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
```

### Create log event file for TensorBoard Visualization
```
tensorboard = TensorBoard(log_dir="logs/%s"%datasetname)
```
### Training the model
```
history = model.fit(train_X, train_y, epochs=300, batch_size=72, validation_data=(test_X, test_y), verbose=0, shuffle=False, callbacks=[tensorboard])
```
### Plotting data
```
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.title('%s Model'%datasetname)
pyplot.legend()
pyplot.show()
```
### Calculate RMSE and plot the result
```
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
plot_predicted, = pyplot.plot(inv_yhat, label='predicted')
plot_test, = pyplot.plot(inv_y, label='test')
pyplot.legend(handles=[plot_predicted, plot_test])
pyplot.title('%s Prediction, RMSE =%.4f'%(datasetname,rmse))
pyplot.xlabel('Month')
pyplot.ylabel('Saturated Thickness')
pyplot.show()
```

## Authors

* **Vinh The Nguyen** - *PhD Student* - Computer Science Department, Texas Tech University
* **Fang Jin** - *Faculty* - Computer Science Department,Texas Tech University
* **Tommy Dang** - *Faculty - Advisor* - Computer Science Department,Texas Tech University
## Acknowledgments
This work was partially supported by the U.S. National Science Foundation under Grant CNS-1737634
