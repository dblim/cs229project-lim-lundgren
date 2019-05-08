import fix_yahoo_finance as yf
from utils import preprocess, quadratic_kernel
import tensorflow as tf

# Variables
compare_to_sklearn: bool = False
# Download data
SP500 = '^GSPC'
GOOGLE = 'GOOGL'
training_data = yf.download(SP500, '2012-01-01', '2015-12-31')

# Preprocess data
val_data = yf.download(SP500, '2016-01-01', '2017-01-01')
x_train, y_train = preprocess(training_data, incremental_data=True)
x_train = quadratic_kernel(x_train)
x_val, y_val = preprocess(val_data, incremental_data=True)
x_val = quadratic_kernel(x_val)

# Data changes n stuff..
n_train, d = x_train.shape
n_val, _ = x_val.shape
y_train = y_train.reshape(n_train, 1)
y_val = y_val.reshape(n_val, 1)

# Hyper Parameters
learning_rate = 0.01
training_epochs = 5000

# tf input
x = tf.placeholder(tf.float32, [None, d], name='x_place')
y = tf.placeholder(tf.float32, [None, 1], name='y_place')

# Weight and bias
w = tf.Variable(tf.random_normal([d, 1]))
b = tf.Variable(tf.random_normal([1]))

# Prediction function and accuracy measure
prediction = tf.nn.sigmoid(tf.add(tf.matmul(x, w), b))
correct = tf.cast(tf.equal(tf.round(prediction), y), dtype=tf.float32)
accuracy = tf.reduce_mean(correct)

# Loss function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=tf.add(tf.matmul(x, w), b)))

# Define optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize
init = tf.global_variables_initializer()

# Train
with tf.Session() as sess:

    sess.run(init)

    for epoch in range(training_epochs):
        sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
        temp_train_acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
        temp_test_acc = sess.run(accuracy, feed_dict={x: x_val, y: y_val})

        if (epoch + 1) % 100 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train accuracy =", temp_train_acc, "test accuracy =", temp_test_acc)

# Sklearn

if compare_to_sklearn is True:
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    y_train = y_train.reshape(n_train,)
    y_val = y_val.reshape(n_val,)
    clf = LogisticRegression(fit_intercept=True, random_state=0, solver='lbfgs')
    clf.fit(x_train, y_train)
    print('Sklearn Accuracy:', np.mean(np.abs(1 - clf.predict(x_val) - y_val)))

