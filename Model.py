import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

def layer(input, wshape, bshape):
    w = tf.compat.v1.get_variable('w', wshape, initializer=tf.random_normal_initializer(stddev=(2.0 / wshape[0]) ** 0.5))
    b = tf.compat.v1.get_variable('b', bshape, initializer=tf.constant_initializer(value=0))
    y = tf.add(tf.matmul(input, w), b)
    return tf.nn.relu(y)


class Model:
    def logistic(x, input, output):
        n0 = input
        n1 = 256
        n2 = 256
        n3 = output
        with tf.compat.v1.variable_scope('L1'):
            Y1 = layer(x, [n0, n1], [n1])
        with tf.compat.v1.variable_scope('L2'):
            Y2 = layer(Y1, [n1, n2], [n2])
        with tf.compat.v1.variable_scope('output'):
            yhat = layer(Y2, [n2, n3], n3)
        return yhat
    def cnn(model):
        model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same', input_shape=(28, 28, 1), activation=tf.nn.relu))
        model.add(MaxPool2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid', activation=tf.nn.relu))
        model.add(MaxPool2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(filters=120, kernel_size=(5,5), strides=1, padding='valid', activation=tf.nn.relu))
        model.add(Dropout(0))
        model.add(Flatten())
        model.add(Dense(84, activation= tf.nn.relu))
        model.add(Dense(10, activation= tf.nn.softmax))
        return model

