import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import Preprocessing
from keras.models import Sequential, load_model
from Model import Model
from Preprocessing import Proprecessing


xtrain, ytrain, x_test, y_test = Preprocessing.Proprecessing.load_data_cnn()
# lr = 0.1
# batchsize1 = 128
# epoch1 = 10
#
# model = Sequential()
# Model.cnn(model)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(xtrain, ytrain, epochs=epoch1, batch_size=batchsize1, verbose=1)
#
# loss, acc = model.evaluate(x_test, y_test)
# print(loss, acc)
# model.save('./CNN_Mnist.h5')
model = load_model('./CNN_Mnist.h5')
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

def showimg(n):
    plt.imshow(x_test[n], cmap="gray")
    plt.show()

def one_img_predict(model, n):
    predict = model.predict(x_test)
    classes_x = np.argmax(predict, axis=1)
    print('Prediction:', classes_x[n])
    print('Answer:', y_test[n])
    showimg(n)

one_img_predict(model, 990)
