import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import *
from f74076108_hw2 import *
import f74076108_hw2 as ui

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.cornerDetect)
        self.pushButton_2.clicked.connect(self.IntrinsicMatrix)
        self.pushButton_3.clicked.connect(self.ExtrinsicMatrix)
        self.pushButton_4.clicked.connect(self.Distortion)
        self.pushButton_5.clicked.connect(self.Undistortion)

    def cornerDetect(self):
        for i in range(15):
            img = cv2.imread("Q2_image" + "\\" + str(i+1) + ".bmp")
            # img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
            ret, corners = cv2.findChessboardCorners(img, (11, 8), None)
            if ret == True:
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
                cv2.imshow("Corner Detection", img)
                cv2.waitKey(500)
    
    def IntrinsicMatrix(self):
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        threedpoints = []
        twodpoints = []
        
        objectp3d = np.zeros((1, 11*8, 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        for i in range(15):
            img = cv2.imread("Q2_image" + "\\" + str(i+1) + ".bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                threedpoints.append(objectp3d)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                twodpoints.append(corners2)
        
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None, None)
        print("Intrinsic:")
        print(matrix)

    def ExtrinsicMatrix(self):
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        threedpoints = []
        twodpoints = []
        objectp3d = np.zeros((1, 11*8, 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        msg = self.textEdit.toPlainText()
        img = cv2.imread("Q2_image" + "\\" + msg + ".bmp")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret == True:
            threedpoints.append(objectp3d)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            twodpoints.append(corners2)
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None, None)
        r_vecs_arr = np.array(r_vecs)
        r_vecs_arr = np.array([r_vecs_arr[0][0], r_vecs_arr[0][1], r_vecs_arr[0][2]])
        r = cv2.Rodrigues(r_vecs_arr)
        t_vecs_arr = np.array(t_vecs)
        t_vecs_arr = np.array([[t_vecs_arr[0][0][0], t_vecs_arr[0][1][0], t_vecs_arr[0][2][0]]])
        extrinsic = r[0]
        extrinsic = np.c_[extrinsic, t_vecs_arr.T]
        print("Extrinsic:")
        print(extrinsic)

    def Distortion(self):
        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        threedpoints = []
        twodpoints = []
        
        objectp3d = np.zeros((1, 11*8, 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

        for i in range(15):
            img = cv2.imread("Q2_image" + "\\" + str(i+1) + ".bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                threedpoints.append(objectp3d)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                twodpoints.append(corners2)
        
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None, None)
        print("Distortion:")
        print(distortion)

    def Undistortion(self):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objectp3d = np.zeros((1, 11*8, 3), np.float32)
        objectp3d[0, :, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        threedpoints = []
        twodpoints = []
        for i in range(15):
            img = cv2.imread("Q2_image" + "\\" + str(i+1) + ".bmp")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret == True:
                threedpoints.append(objectp3d)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                twodpoints.append(corners2)
                
        
        ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(threedpoints, twodpoints, gray.shape[::-1], None, None)
        
        for i in range(15):
            img = cv2.imread("Q2_image" + "\\" + str(i+1) + ".bmp")
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
            dst = cv2.undistort(img, matrix, distortion, None, newcameramtx)
            # dst = cv2.resize(dst, (512, 512), cv2.INTER_AREA)
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]
            dst = cv2.resize(dst, (512, 512), cv2.INTER_AREA)
            img = cv2.resize(img, (512, 512), cv2.INTER_AREA)
            cv2.imshow("Distorted", dst)
            cv2.imshow("Undistorted", img)
            cv2.waitKey(500)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWindow = Main()
    mainWindow.show()
    sys.exit(app.exec_())