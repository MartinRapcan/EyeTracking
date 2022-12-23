import argparse
import sys
import random
import os
import cv2
import numpy as np
import re

from PySide6.QtCore import Qt
from pyqt_frameless_window import FramelessMainWindow
from tkinter import *
from ctypes import windll
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QDialog, QHBoxLayout, QLabel, QTextEdit, QPushButton, QWidget, QVBoxLayout
from os.path import isfile, join
from pupil_detectors import Detector2D
import pyautogui
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
# generate ui pyside6-uic ./ui/main.ui > ui_mainwindow.py

def main(path):
    # create 2D detector
    detector_2d = Detector2D()
    # create pye3D detector
    camera = CameraModel(focal_length=561.5, resolution=[400, 400])
    detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
    
    eye_coordinates = ()
    eye_video = cv2.VideoCapture(path)
    # read each frame of video and run pupil detectors
    while eye_video.isOpened():
        frame_number = eye_video.get(cv2.CAP_PROP_POS_FRAMES)
        fps = eye_video.get(cv2.CAP_PROP_FPS)
        ret, eye_frame = eye_video.read()
    
        if ret:
            # dobre pre iris ...
            #(thresh, blackAndWhiteImage) = cv2.threshold(eye_frame, 50, 255, cv2.THRESH_TOZERO_INV)
            # (thresh, blackAndWhiteImage) = cv2.threshold(eye_frame, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            # cv2.GaussianBlur(blackAndWhiteImage, (17, 17), cv2.BORDER_DEFAULT)
            # cv2.medianBlur(blackAndWhiteImage, 21)

            # edges = cv2.Canny(blackAndWhiteImage, 100, 200)
            # cv2.imshow('frame', blackAndWhiteImage)
            # cv2.imshow('edges', edges)

            # read video frame as numpy array
            grayscale_array = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            # run 2D detector on video frame
            result_2d = detector_2d.detect(grayscale_array)

            result_2d["timestamp"] = frame_number / fps
            # pass 2D detection result to 3D detector
            result_3d = detector_3d.update_and_detect(result_2d, grayscale_array)

            ellipse_3d = result_3d["ellipse"]
            
            #print(ellipse_3d)
            # draw 3D detection result on eye frame
            cv2.ellipse(
                eye_frame,
                tuple(int(v) for v in ellipse_3d["center"]),
                tuple(int(v / 2) for v in ellipse_3d["axes"]),
                ellipse_3d["angle"],
                0,
                360,  # start/end angle for drawing
                (0, 255, 0),  # color (BGR): red
            )
            cv2.circle(
                eye_frame,
                tuple(int(v) for v in ellipse_3d["center"]),
                2,
                (0, 0, 255),  # color (BGR): blue
                thickness=-2,
            )
            # show frame
            cv2.imshow("eye_frame", eye_frame)
            # press esc to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            break
    eye_video.release()
    cv2.destroyAllWindows()


def create_video(path, name, resolution):
    if resolution and path and name:
        images = [f for f in os.listdir(path) if isfile(join(path, f))]
        video = cv2.VideoWriter(f'dataset/{name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)
        for image in images:
            img = cv2.imread(path + image)
            video.write(img)
        video.release()

class MainWindow(FramelessMainWindow):
    detector_2d = Detector2D()
    camera = CameraModel(focal_length=561.5, resolution=[400, 400])
    detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
    images = {}
    imageB = None
    angle = 0
    topLeft = 80
    topRight = 70
    bottomLeft = 130
    bottomRight = 120

    def __init__(self):
        super().__init__()
        self.__testWidget = QWidget()
        loader = QUiLoader()
        ui = QFile("ui/test.ui")
        ui.open(QFile.ReadOnly)
        self.__testWidget = loader.load(ui)
        ui.close()
        self.setWindowTitle('Eye Tracking')
        self.setWindowIcon('public/light.png')
        self.mainWidget = self.centralWidget()
        lay = self.mainWidget.layout()
        lay.addWidget(self.__testWidget)
        self.mainWidget.setLayout(lay)
        self.setCentralWidget(self.mainWidget)
        self.setGeometry(200, 50, 1100, 735)
        self.setFixedSize(1100, 730)
        titleBar = self.getTitleBar()
        titleBar.setFixedHeight(35)
        #print(dir(titleBar))
        #print(titleBar.children())
        # use findchildren to find QPushButtons
        for button in titleBar.findChildren(QPushButton):
            button.setStyleSheet("QPushButton {background-color: #90cdf4; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:hover {background-color: #3182ce; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:pressed {background-color: #3182ce; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px}")
        titleBar.findChildren(QLabel)[1].setStyleSheet("QLabel {font-size: 15px; color: #F7FAFC; font-weight: bold; margin-left: 15px}")
        titleBar.findChildren(QLabel)[0].setStyleSheet("QLabel {margin-left: 5px}")
        self.__testWidget.slide.hide()
        self.image = None
        self.screenShot = None
        self.__testWidget.startButton.clicked.connect(self.startWebcam)
        self.__testWidget.stopButton.clicked.connect(self.stopWebcam)
        self.__testWidget.loadImage.clicked.connect(self.loadImage)
        self.__testWidget.removeImage.clicked.connect(self.removeImage)
        self.__testWidget.slideShow.clicked.connect(self.slideShow)
        # hide widget for now
        self.__testWidget.slide.hide()
        #self.window.previous.hide()
        #self.window.previous.clicked.connect(self.previousImage)
        self.__testWidget.next.hide()
        self.__testWidget.next.clicked.connect(self.nextImage)
        self.__testWidget.end.hide()
        self.__testWidget.end.clicked.connect(self.endSlideShow)
        # self.position = self.centralWidget().frameGeometry()
        # self.imagePosition = self.__testWidget.image.frameGeometry()
        # get window position
        self.position = self.frameGeometry()

    # def __init__(self):
    #     super().__init__()
    #     self.loader = QUiLoader()
    #     self.ui = QFile("ui/main.ui")
    #     self.ui.open(QFile.ReadOnly)
    #     self.window = self.loader.load(self.ui)
    #     self.ui.close()
    #     self.window.setGeometry(200, 50, 1100, 700)
    #     self.window.setFixedSize(1100, 700)
    #     self.window.setWindowTitle("Eye Tracking Platform")
    #     self.window.setWindowIcon(QtGui.QIcon("public/icon.png"))
    #     self.window.show()
    #     self.image = None
    #     self.screenShot = None
    #     self.window.startButton.clicked.connect(self.startWebcam)
    #     self.window.stopButton.clicked.connect(self.stopWebcam)
    #     self.window.loadImage.clicked.connect(self.loadImage)
    #     self.window.removeImage.clicked.connect(self.removeImage)
    #     self.window.slideShow.clicked.connect(self.slideShow)
    #     # hide widget for now
    #     self.window.slide.hide()
    #     #self.window.previous.hide()
    #     #self.window.previous.clicked.connect(self.previousImage)
    #     self.window.next.hide()
    #     self.window.next.clicked.connect(self.nextImage)
    #     self.window.end.hide()
    #     self.window.end.clicked.connect(self.endSlideShow)
    #     self.position = self.window.frameGeometry()
    #     self.imagePosition = self.window.image.frameGeometry()

    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.gif *.png *.jpeg)")
        if fname[0] != "":
            imageName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
            if not self.images.get(imageName):
                self.images[imageName] = QtGui.QImage(fname[0])
                self.__testWidget.listImages.addItem(imageName)

    def removeImage(self):
        if self.__testWidget.listImages.currentItem():
            self.images.pop(self.__testWidget.listImages.currentItem().text())
            self.__testWidget.listImages.takeItem(self.__testWidget.listImages.currentRow())

    # TODO: hide main window and show slide show window 
    def slideShow(self):
        if self.__testWidget.listImages.count() > 0:
            self.__testWidget.slide.show()
            self.imageB = self.__testWidget.listImages.item(0).text()
            self.__testWidget.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]))
            if self.__testWidget.listImages.count() > 1:
                self.__testWidget.next.show()
            else:
                self.__testWidget.end.show()

    def endSlideShow(self):
        self.__testWidget.slide.hide()
        self.__testWidget.next.hide()
        self.__testWidget.end.hide()


    def previousImage(self):
        pass
            
    def nextImage(self):
        if self.imageB != list(self.images.keys())[-1]:
            self.imageB = list(self.images.keys())[list(self.images.keys()).index(self.imageB) + 1]
            self.__testWidget.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]))
            if self.imageB == list(self.images.keys())[-1]:
                self.__testWidget.next.hide()
                self.__testWidget.end.show()
        else:
            self.__testWidget.next.hide()
            self.__testWidget.end.show()

    def startWebcam(self):
        #self.startScreen()
        self.captureWeb = cv2.VideoCapture("data/train_img.mp4")
        self.captureWeb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.captureWeb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.timerWeb = QtCore.QTimer()
        self.timerWeb.timeout.connect(self.update_frame)
        self.timerWeb.start(5)

    def update_frame(self):
        self.position = self.frameGeometry()
        ret, self.image = self.captureWeb.read()
        frame_number = self.captureWeb.get(cv2.CAP_PROP_POS_FRAMES)
        fps = self.captureWeb.get(cv2.CAP_PROP_FPS)

        if ret:
            grayscale_array = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            result_2d = self.detector_2d.detect(grayscale_array)
            result_2d["timestamp"] = frame_number / fps
            result_3d = self.detector_3d.update_and_detect(result_2d, grayscale_array)
            ellipse_3d = result_3d["ellipse"]

            self.angle = round(float(ellipse_3d["angle"]), 2)

            cv2.ellipse(
                self.image,
                tuple(int(v) for v in ellipse_3d["center"]),
                tuple(int(v / 2) for v in ellipse_3d["axes"]),
                ellipse_3d["angle"],
                0,
                360,  # start/end angle for drawing
                (0, 255, 0),  # color (BGR): red
            )
            cv2.circle(
                self.image,
                tuple(int(v) for v in ellipse_3d["center"]),
                2,
                (0, 0, 255),  # color (BGR): blue
                thickness=-2,
            )

            # print text on image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image, f'Angle: {self.angle}', (5, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            # cv2.putText(self.image, f"Position: {self.position.getCoords()}", (5, 50), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
        x = self.position.getCoords()[0]
        y = self.position.getCoords()[1]
        newCoords = (self.__testWidget.image.geometry().getCoords()[0] + x, self.__testWidget.image.geometry().getCoords()[1] + y, self.__testWidget.image.geometry().getCoords()[2] + x, self.__testWidget.image.geometry().getCoords()[3] + y)
        #print(f"Position: {self.position.getCoords()}")   # 30 pixelov na title bar       
        print(f"Position: {newCoords}")
        self.displayImage(self.image, 1)

    def stopWebcam(self):
        if self.timerWeb.isActive(): #and self.timerScreen.isActive():
            self.timerWeb.stop()
            #self.timerScreen.stop()

    def displayImage(self, img, window=1):
        qformat = QtGui.QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QtGui.QImage.Format_RGBA8888
            else:
                qformat = QtGui.QImage.Format_RGB888

        outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        if window == 1:
            self.__testWidget.imgLabel.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.__testWidget.imgLabel.setScaledContents(True)

    # def startScreen(self):
    #     # self.captureScreen = cv2.VideoCapture("data/train_img.mp4")
    #     # self.captureScreen.set(cv2.CAP_PROP_FRAME_WIDTH, 370)
    #     # self.captureScreen.set(cv2.CAP_PROP_FRAME_HEIGHT, 260)
    #     self.timerScreen = QtCore.QTimer()
    #     self.timerScreen.timeout.connect(self.updateScreen)
    #     self.timerScreen.start(5)

    # def displayScreen(self, img, window=1):

    #     if len(img.shape) == 3:
    #         if img.shape[2] == 4:
    #             qformat = QtGui.QImage.Format_RGBA8888
    #         else:
    #             qformat = QtGui.QImage.Format_RGB888

    #     outImage = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
    #     outImage = outImage.rgbSwapped()
    #     if window == 2:
    #         self.window.screenLabel.setPixmap(QtGui.QPixmap.fromImage(outImage))
    #         self.window.screenLabel.setScaledContents(True)

    # def updateScreen(self):
    #     self.screenShot = pyautogui.screenshot()
    #     frame = np.array(self.screenShot)
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     cv2.circle(
    #         frame,
    #         (frame.shape[1] // 2, frame.shape[0] // 2),
    #         15,
    #         (0, 0, 255),  # color (BGR): blue
    #         thickness=2,
    #     )

    #     self.displayScreen(frame, 2)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet("QMainWindow {background: '#171923';}")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

    #main("data/train_img.mp4")
    #create_video("dataset/data_png/", "another_png", resolution=(1280, 720))


# využiť angle .. ktorý je v 3D detektori
# skusiť vytvoriť vektor z 3D detekcie a zistiť jeho smer