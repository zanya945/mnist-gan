import matplotlib as plt
import numpy  as np
import Preprocessing

xtrain, ytrain, x_test, y_test = Preprocessing.Proprecessing.load_data_cnn()

class Showimg:
    def showimg(n):
        plt.imshow(x_test[n], cmap="gray")
        plt.show()

    def one_img_predict(model, n):
        predict = model.predict(x_test)
        classes_x = np.argmax(predict, axis=1)
        print('Prediction:', classes_x[n])
        print('Answer:', y_test[n])
        showimg(n)
