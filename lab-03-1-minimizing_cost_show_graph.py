# Lab 3 Minimizing Cost
import tensorflow.compat.v1 as tf  # modified from 'import tensorflow as tf'
import matplotlib.pyplot as plt

tf.disable_v2_behavior() # added due to code only available with TF version 1

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Variables for plotting cost function
W_history = []
cost_history = []

# Launch the graph in a session.
with tf.Session() as sess:
    for i in range(-30, 50):
        curr_W = i * 0.1
        curr_cost = sess.run(cost, feed_dict={W: curr_W})

        W_history.append(curr_W)
        cost_history.append(curr_cost)

# Show the cost function
plt.plot(W_history, cost_history)
plt.show()
