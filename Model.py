import tensorflow as tf


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
