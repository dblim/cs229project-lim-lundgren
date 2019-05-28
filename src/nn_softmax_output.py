from alpha_vantage.timeseries import TimeSeries
from utils import preprocess, lookback_kernel, quadratic_kernel, y_numeric_to_vector
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# methods
keras: bool = False
t_flow: bool = True
plot_roc: bool = True
reg: bool = True

# Download data
SP500 = '^GSPC'
GOOGLE = 'GOOGL'

ts = TimeSeries(key='YOUR_API_KEY', output_format='pandas')
data, _ = ts.get_intraday(SP500, interval='5min', outputsize='full')

# Preprocess data
lookback_period = 10
bins: bool = False
zero_one: bool = True
x_train: np.ndarray
x_val: np.ndarray
y_train: np.ndarray
y_val: np.ndarray
K: int
if bins is True:
    partitions = [0]
    K = len(partitions) + 1
    X, Y = preprocess(data, incremental_data=True, output_variable='multinomial', partitions=partitions)
    training_width = int(0.85*len(X))
    X, Y = lookback_kernel(X, Y, periods=lookback_period)
    Y = y_numeric_to_vector(Y, K)
    X = quadratic_kernel(X)
    x_train = X[0:training_width]
    y_train = Y[0:training_width]
    x_val = X[training_width:X.shape[0]]
    y_val = Y[training_width:Y.shape[0]]
    if K == 2:
        y_mean_train = np.mean(y_train[:, 1])
        y_mean_val = np.mean(y_val[:, 1])
        print('training set mean:', y_mean_train)
        print('test set mean:', y_mean_val)
elif zero_one is True:
    X, Y = preprocess(data, incremental_data=True)
    K = 1
    training_width = int(0.85 * len(X))
    X, Y = lookback_kernel(X, Y, periods=lookback_period)
    #X = quadratic_kernel(X)
    x_train = X[0:training_width]
    y_train = Y[0:training_width]
    x_val = X[training_width:X.shape[0]]
    y_val = Y[training_width:Y.shape[0]]
    y_train = y_train.reshape(y_train.shape[0], 1)
    y_val = y_val.reshape(y_val.shape[0], 1)
    print('training set mean:', np.mean(y_train))
    print('test set mean:', np.mean(y_val))

# Data changes n stuff..
n_train, d = x_train.shape
n_val, _ = x_val.shape
print(n_train, d)

# Hyperparameters
learning_rate = 0.001
training_epochs = 5000

# Regularization parameters
beta = 0.1

# Arrays:
W1_weights: np.ndarray
b1_weights: np.ndarray
W2_weights: np.ndarray
b2_weights: np.ndarray
W3_weights: np.ndarray
b3_weights: np.ndarray

# Tensorflow implementation

if t_flow is True:
    # tf input
    x = tf.placeholder(tf.float32, [None, d], name='x_place')
    y = tf.placeholder(tf.float32, [None, K], name='y_place')

    # dimensions
    layer_1_size = 20
    layer_2_size = 10

    # Variables
    W1 = tf.Variable(tf.random_normal([d, layer_1_size]), name='W1')
    b1 = tf.Variable(tf.random_normal([layer_1_size]), name='b1')

    W2 = tf.Variable(tf.random_normal([layer_1_size, layer_2_size]), name='W2')
    b2 = tf.Variable(tf.random_normal([layer_2_size]), name='b2')

    W3 = tf.Variable(tf.random_normal([layer_2_size, K]), name='W3')
    b3 = tf.Variable(tf.random_normal([K]), name='b3')

    # Pipeline
    z1 = tf.add(tf.matmul(x, W1), b1)
    a1 = tf.nn.sigmoid(z1)

    z2 = tf.add(tf.matmul(a1, W2), b2)
    a2 = tf.nn.relu(z2)

    z3 = tf.add(tf.matmul(a2, W3), b3)
    y_hat = tf.nn.sigmoid(z3)


    # Loss function
    y_clipped = tf.clip_by_value(y_hat, 1e-10, 0.9999999)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_clipped))

    if reg is True:
        regularizer = beta * (tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W1))
        cost = tf.reduce_mean(cost + beta * regularizer)

    prediction = tf.round(y_hat)
    correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
    accuracy = tf.reduce_mean(correct)
    #cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))
    #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_hat, 1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(training_epochs):
            sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
            temp_train_acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
            temp_test_acc = sess.run(accuracy, feed_dict={x: x_val, y: y_val})

            if (epoch + 1) % 100 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "train accuracy =",
                      max(temp_train_acc, 1-temp_train_acc), "test accuracy =", max(temp_test_acc, 1-temp_test_acc))

        # Get weights
        if plot_roc is True:
            divider = 1
            W1_weights = W1.eval(sess)/divider
            b1_weights = b1.eval(sess).reshape(b1.eval(sess).shape[0], 1)/divider
            W2_weights = W2.eval(sess)/divider
            b2_weights = b2.eval(sess).reshape(b2.eval(sess).shape[0], 1)/divider
            W3_weights = W3.eval(sess)/divider
            b3_weights = b3.eval(sess).reshape(b3.eval(sess).shape[0], 1)/divider

if plot_roc is True:
    z1_numeric = np.add(np.matmul(x_val, W1_weights), b1_weights.T)
    a1_numeric = 1/(1+np.exp(-z1_numeric))

    z2_numeric = np.add(np.matmul(a1_numeric, W2_weights), b2_weights.T)
    #a2_numeric = 1/(1+np.exp(-z2_numeric))
    a2_numeric = z2_numeric
    a2_numeric[a2_numeric < 0] = 0

    z3_numeric = np.add(np.matmul(a2_numeric, W3_weights), b3_weights.T)

    TPR_list = []
    FPR_list = []
    step_size = 20
    for i in range((int(min(z3_numeric))-1)*step_size, (int(max(z3_numeric))+1)*step_size):
        threshold = i/step_size
        y_hat_numeric = np.zeros(z3_numeric.shape)
        y_hat_numeric[z3_numeric > threshold] = 1

        TP = np.sum(np.logical_and(y_hat_numeric == 1, y_val == 1))
        TN = np.sum(np.logical_and(y_hat_numeric == 0, y_val == 0))
        P = np.sum(y_val)
        N = y_val.shape[0] - P
        TPR = TP/P
        FPR = 1 - TN/N
        TPR_list.append(TPR)
        FPR_list.append(FPR)
    plt.scatter(FPR_list, TPR_list)
    plt.plot(np.linspace(0, 1), np.linspace(0, 1), color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

if keras is True:
    from keras.models import Sequential
    from keras.layers import Dense

    def build_model():
        model = Sequential()
        model.add(Dense(20, input_dim=20, activation='relu'))
        model.add(Dense(40, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model