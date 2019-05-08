import fix_yahoo_finance as yf
from utils import preprocess
import tensorflow as tf

# Download data
training_data = yf.download('^GSPC', '2012-01-01', '2015-12-31')
val_data = yf.download('^GSPC', '2016-01-01', '2017-01-01')
x_train, y_train = preprocess(training_data)
x_val, y_val = preprocess(val_data)

# Data changes n stuff..
n_train, d = x_train.shape
n_val, _ = x_val.shape
y_train = y_train.reshape(n_train, 1)
y_val = y_val.reshape(n_val, 1)

n_neurons_1 = 1024
n_neurons_2 = 512
n_target = 1

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_train, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_2]))

# Output Layer: Variables for output layer and biases
W_output = tf.Variable(weight_initializer( [ n_neurons_2, n_target   ]   ))
bias_output = tf.Variable(bias_output([n_target]))

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(x_train, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
output =  tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
