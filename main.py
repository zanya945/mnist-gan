import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist


def load_data(numclass=10):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    xtrain = np.reshape(x_train,  (x_train.shape[0], -1))
    xtest = np.reshape(x_test, (x_test.shape[0], -1))

    xtrain = xtrain.astype('float32')
    xtest = xtest.astype('float32')

    xtrain /= 255
    xtest /= 255
    ytrain = np.eye(numclass)[y_train]
    ytest = np.eye(numclass)[y_test]
    return xtrain, ytrain, xtest, ytest

def batchsize(batchsize, label, data):
    idx = np.arange(0, len(data))
    idx = np.random.shuffle(idx)
    idx = idx[:batchsize]
    data_shu = [data[i] for i in idx]
    label_shu = [label[i] for i in idx]
    return np.asarray(data_shu), np.asarray(label_shu)

def build_model(X, input, output):
    n0 = input
    n1 = 256
    n2 = output
    w1 = tf.Variable(tf.random.normal([n0,n1]))
    b1 = tf.Variable(tf.random.normal([n1]))
    w2 = tf.Variable(tf.random.normal([n1, n2]))
    b2 = tf.Variable(tf.random.normal([n2]))

    Y1 = tf.add(tf.matmul(X, w1), b1)
    yhat = tf.add(tf.matmul(Y1, w2), b2)
    return yhat

xtrain, ytrain, x_test, y_test = load_data()
lr = 0.1
batchsize = 128
epoch = 5000

input = xtrain.shape[1]
output = ytrain.shape[1]
tf.compat.v1.disable_eager_execution()
X = tf.compat.v1.placeholder(tf.float32, input)
Y = tf.compat.v1.placeholder(tf.float32, output)

yhat = build_model(X, input, output)
pred = tf.nn.softmax(yhat)

lossop = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_op = optimizer.minimize(lossop)
correct = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.compat.v1.glorot_normal_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epoch):
        xbatch, ybatch = batchsize(batchsize, xtrain, ytrain)
        sess.run(train_op, feed_dict={X: xbatch, Y: ybatch})
        loss, acc = sess.run([lossop, accuracy], feed_dict={X: xbatch, Y: ybatch})
        print("epoch " + str(epoch) + ", loss= " + "{:.4f}".format(loss) + ", acc= " + "{:.3f}".format(acc))

def showimg(ndarr):
    img1 = ndarr.copy()
    img1.shape = (28, 28)
    plt.imshow(img1, cmap="gray")
    plt.show()



