import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Input, Reshape
from keras.layers.activation import LeakyReLU, Softmax
from keras.optimizers import Adam
from keras.datasets.mnist import load_data
class Gan():
    def __init__(self):
        self.imgsize = 28
        self.channel = 1
        self.imgshape = (28, 28, 1)
        self.dim = 100
        optimizer = Adam(0.002, 0.5)
        self.discriminator = self.cnn_discriminator()
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        self.img_generactor = self.img_generactor()
        owo = Input(shape=(self.dim,))
        img = self.img_generactor(owo)
        validity = self.discriminator(img)
        self.discriminator.trainable = False
        self.combined = Model(owo, validity)
        self.combined.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])



    def cnn_discriminator(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(5, 5), strides=1, padding='same', input_shape=(28, 28, 1), activation=tf.nn.relu))
        model.add(MaxPool2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=1, padding='valid', activation=tf.nn.relu))
        model.add(MaxPool2D(pool_size=(2,2), strides=2))
        model.add(Conv2D(filters=120, kernel_size=(5,5), strides=1, padding='valid', activation=tf.nn.relu))
        model.add(Flatten())
        model.add(Dense(84, activation= tf.nn.relu))
        model.add(Dense(10))
        model.add(Dense(1, activation= tf.nn.sigmoid))

        img = Input(shape=self.imgshape)
        va = model(img)
        print(model.summary())
        return Model(img, va)
    def img_generactor(self):
        model = Sequential()
        model.add(Dense(256, input_dim=100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(784, activation='tanh'))
        model.add(Reshape(self.imgshape))

        noise = Input(shape=(100,))
        img = model(noise)
        print(model.summary())
        return Model(noise, img)
    def train_gan(self, epochs, batch_size=128, sample_interval=50):
        (x_train, _),(_, _) = load_data()

        x_train = x_train/127.5 -1
        x_train = np.expand_dims(x_train, axis=3)

        vaild = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        for i in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            img = x_train[idx]

            noise = np.random.normal(0, 1,(batch_size, self.dim))
            gan_img = self.img_generactor.predict(noise)
            d_loss_real = self.discriminator.train_on_batch(img, vaild)
            d_loss_fake = self.discriminator.train_on_batch(gan_img, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1,(batch_size, self.dim))
            g_loss = self.combined.train_on_batch(noise, vaild)
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100 * d_loss[1], g_loss[0]))
            if i % sample_interval == 0:
                self.sample_images(i)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.dim))
        gen_imgs = self.img_generactor.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()





