import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
import matplotlib.image as pltimg
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.models import load_model

from PyQt5.QtWidgets import *
from f74076108_Q5 import *
import f74076108_Q5 as ui

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.showSum)
        self.pushButton_2.clicked.connect(self.showTensorBoard)
        self.pushButton_3.clicked.connect(self.modelTest)
        self.pushButton_4.clicked.connect(self.model_acc)

        self.labels = ['cats', 'dogs']

    def showSum(self):
        model = load_model('ResNet50_cat_dog.h5')
        print(model.summary())

    def showTensorBoard(self):
        image = cv2.imread('tensorBoard.png')
        cv2.imshow("TensorBoard", image)

    def modelTest(self):
        msg = self.lineEdit.text()
        model = load_model('ResNet50_cat_dog.h5')
        files = 'kagglecatsanddogs_3367a\PetImages\\'
        if (int(msg) % 2 == 1):
            num = int(msg) % 2000
            img = image.load_img(files + 'Cat\\' + str(num) + '.jpg', target_size=(224, 224))
            img_show = pltimg.imread(files + 'Cat\\' + str(num) + '.jpg')
        else :
            num = int(msg) % 2000
            img = image.load_img(files + 'Dog\\' + str(num) + '.jpg', target_size=(224, 224))
            img_show = pltimg.imread(files + 'Dog\\' + str(num) + '.jpg')
            
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)[0]
        title = self.labels[pred.argmax()]
        plt.imshow(img_show)
        plt.title(title)
        plt.show()
        

    def model_acc(self):
        model = ['Before Random-Erasing', 'After Random-Erasing']
        acc = [99.50, 99.07]
        plt.bar(model, acc)
        plt.ylabel('Accuracy')
        for index,data in enumerate(acc):
            plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=20))
        plt.show()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainwindow = Main()
    mainwindow.show()
    sys.exit(app.exec_())