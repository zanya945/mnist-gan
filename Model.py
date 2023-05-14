import tensorflow as tf

class Model:
    def logistic(X, input, output):
        n0 = input
        n1 = 256
        n2 = output
        w1 = tf.Variable(tf.random.normal([n0, n1]))
        b1 = tf.Variable(tf.random.normal([n1]))
        w2 = tf.Variable(tf.random.normal([n1, n2]))
        b2 = tf.Variable(tf.random.normal([n2]))

        Y1 = tf.add(tf.matmul(X, w1), b1)
        yhat = tf.add(tf.matmul(Y1, w2), b2)
        return yhat