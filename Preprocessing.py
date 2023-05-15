import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist


class Proprecessing:
    def load_data(numclass=10):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        xtrain = np.reshape(x_train, (x_train.shape[0], -1))
        xtest = np.reshape(x_test, (x_test.shape[0], -1))

        xtrain = xtrain.astype('float32')
        xtest = xtest.astype('float32')

        xtrain /= 255
        xtest /= 255
        ytrain = np.eye(numclass)[y_train]
        ytest = np.eye(numclass)[y_test]
        return xtrain, ytrain, xtest, ytest
    def load_data_cnn(numclass = 10):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        xtrain = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        xtest = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        ytrain = np.eye(numclass)[y_train]
        ytest = np.eye(numclass)[y_test]
        return xtrain, ytrain, xtest, ytest

    def batch_size(batchsize, data, label):
        idx = np.arange(0, len(data))
        np.random.shuffle(idx)
        idx = idx[:batchsize]
        data_shu = [data[i] for i in idx]
        label_shu = [label[i] for i in idx]
        return np.asarray(data_shu), np.asarray(label_shu)
