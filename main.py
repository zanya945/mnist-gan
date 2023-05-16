import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import Preprocessing
from keras.models import Sequential, load_model
from Model_self import Model
from Preprocessing import Proprecessing
from Train import Train
from Gan import Gan

gan = Gan()

# input = xtrain.shape[1] #784
# output = ytrain.shape[1] #10
# tf.compat.v1.disable_eager_execution()
# X = tf.compat.v1.placeholder(tf.float32, shape=[None, input])
# Y = tf.compat.v1.placeholder(tf.float32, shape=[None, output])
#
# yhat = Model.logistic(X, input, output)
# pred = tf.nn.softmax(yhat)
# lossop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=Y))
# optimizer = tf.compat.v1.train.GradientDescentOptimizer(lr)
# train_op = optimizer.minimize(lossop)
# correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
# init = tf.compat.v1.global_variables_initializer()
#
# with tf.compat.v1.Session() as sess:
#     sess.run(init)
#     for epoch in range(epoch):
#         xbatch, ybatch = Preprocessing.Proprecessing.batch_size(batchsize, xtrain, ytrain)
#         sess.run(train_op, feed_dict={X: xbatch, Y: ybatch})
#         loss, acc = sess.run([lossop, accuracy], feed_dict={X: xbatch, Y: ybatch})
#     acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
#     print('test acc=' + '{:.3f}'.format(acc))
#     sess.close()


