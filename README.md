# Predict Saturated Thickness using TensorBoard Visualization
Water plays a critical role in our living and manufacturing activities. The continuously growing exploitation of water over the aquifer poses a risk for over-extraction and pollution, leading to many negative effects on land irrigation. Therefore, predicting aquifer water level accurately is urgently important, which can help us prepare water demands ahead of time. In this study, we employ the Long-Short Term Memory (LSTM) model to predict the saturated thickness of an aquifer in the Southern High Plains Aquifer System in Texas, and exploit TensorBoard as a guide for model configurations. The Root Mean Squared Error of this study shows that the LSTM model can provide a good prediction capability using multiple data sources, and provides a good visualization tool to help us understand and evaluate the model configuration. 
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Prerequisites
In this documentation, we demonstrate how to run TensorFlow and TensorBoard on MacOS, but it supposes to work on both Windows and Ubuntu.
* **python3 and pip3** 

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
...   writerlog = tf.summary.FileWrite('./logpath', sess.graph)
...   print(sess.run(sum))
>>>
```
After you hit Enter, operations are written to the event file in the logpath folder. Now we can initialize TensorBoard by typing the followig command
```
$tensorboard --logdir logpath
```
It gives the result: TensorBoard 1.8.0 at http://Vinhs-MacBook-Air.local:6006 (Press CTRL+C to quit).....pay attention to the last four digits only: the port number 6006.

Then open any web browser and type: localhost:6006 on the address bar.

![GitHub Logo](/figures/First_demo.png)
## Authors

* **Vinh The Nguyen** - *PhD Student* - Computer Science Department, Texas Tech University
* **Fang Jin** - *Faculty* - Computer Science Department,Texas Tech University
* **Tommy Dang** - *Faculty - Advisor* - Computer Science Department,Texas Tech University
## Acknowledgments
This work was partially supported by the U.S. National Science Foundation under Grant CNS-1737634
