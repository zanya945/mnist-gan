import Preprocessing
from keras.models import Sequential
from Model_self import Model

class Train:
    def cnn_train(self):
        xtrain, ytrain, x_test, y_test = Preprocessing.Proprecessing.load_data_cnn()
        lr = 0.1
        batchsize1 = 128
        epoch1 = 10

        model = Model.cnn_discriminator()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xtrain, ytrain, epochs=epoch1, batch_size=batchsize1, verbose=1)

        loss, acc = model.evaluate(x_test, y_test)
        print(loss, acc)
        model.save('./CNN_Mnist_di.h5')