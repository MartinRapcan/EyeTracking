import argparse
import sys
import random
import os
import cv2
import numpy as np
import re
import time

from PySide6.QtCore import Qt
from pyqt_frameless_window import FramelessMainWindow
from tkinter import *
from ctypes import windll
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, QPalette, QBrush, QIcon, QTransform, QColor
from PySide6.QtCore import QFile, QObject, QThread, Signal, Slot, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QDialog, QHBoxLayout, QLabel, QTextEdit, QPushButton, QWidget, QVBoxLayout, QSplashScreen, QProgressBar, QStyleFactory
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
    loader = QUiLoader()
    videoPath = None
    isPaused = False
    timerWeb = None

    def __init__(self):
        super().__init__()
        self.__testWidget = QWidget()
        ui = QFile("ui/test.ui")
        ui.open(QFile.ReadOnly)
        self.__testWidget = self.loader.load(ui)
        ui.close()
        self.setWindowTitle('Eye Tracking')
        self.setWindowIcon('public/light.png')
        self.mainWidget = self.centralWidget()
        lay = self.mainWidget.layout()
        lay.addWidget(self.__testWidget)
        self.mainWidget.setLayout(lay)
        self.setCentralWidget(self.mainWidget)
        self.setGeometry(200, 50, 1100, 735)
        self.setFixedSize(1100, 735)
        titleBar = self.getTitleBar()
        titleBar.setFixedHeight(35)
        #print(dir(titleBar))
        #print(titleBar.children())
        # use findchildren to find QPushButtons
        for button in titleBar.findChildren(QPushButton):
            button.setStyleSheet("QPushButton {background-color: #FFE81F; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:hover {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:pressed {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px}")
        titleBar.findChildren(QLabel)[1].setStyleSheet("QLabel {font-size: 15px; color: #F7FAFC; font-weight: bold; margin-left: 10px}")
        titleBar.findChildren(QLabel)[0].setStyleSheet("QLabel {margin-left: 10px}")
        self.image = None
        self.__testWidget.startButton.clicked.connect(self.startWebcam)
        self.__testWidget.stopButton.clicked.connect(self.stopWebcam)
        self.__testWidget.loadImage.clicked.connect(self.loadImage)
        self.__testWidget.removeImage.clicked.connect(self.removeImage)
        self.__testWidget.slideShow.clicked.connect(self.slideShow)
        self.__testWidget.videoPicker.clicked.connect(self.videoPicker)
        self.__testWidget.startCalibrationButton.clicked.connect(self.openCalibration)
        self.slideShowOverlay = None
        self.calibrationOverlay = None

    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.gif *.png *.jpeg)")
        if fname[0] != "":
            imageName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
            if not self.images.get(imageName):
                self.images[imageName] = QtGui.QImage(fname[0])
                self.__testWidget.listImages.addItem(imageName)

    def videoPicker(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Video files (*.mp4 *.avi *.mov *.mkv)")
        if fname[0] != "":
            videoName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
            self.videoPath = fname[0]
            self.__testWidget.videoPath.setText(videoName)
        else:
            self.videoPath = None
            self.__testWidget.videoPath.setText("")

    def removeImage(self):
        if self.__testWidget.listImages.currentItem():
            self.images.pop(self.__testWidget.listImages.currentItem().text())
            self.__testWidget.listImages.takeItem(self.__testWidget.listImages.currentRow())

    def slideShow(self):
        if self.timerWeb is not None and self.timerWeb.isActive() and self.__testWidget.listImages.count() > 0:
            if not self.slideShowOverlay:
                ui = QFile("ui/fullScreenSlideShowOverlay.ui")
                ui.open(QFile.ReadOnly)
                self.slideShowOverlay = self.loader.load(ui)
                ui.close()
            if self.__testWidget.listImages.count() > 1:
                self.slideShowOverlay.next.show()
                self.slideShowOverlay.end.hide()
            else:
                self.slideShowOverlay.next.hide()
                self.slideShowOverlay.end.show()
            self.slideShowOverlay.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            self.slideShowOverlay.showFullScreen()
            self.slideShowOverlay.marker.setPixmap(QtGui.QPixmap("public/marker.png").scaled(80, 80, QtCore.Qt.KeepAspectRatio))
            self.slideShowOverlay.next.clicked.connect(self.nextImage)
            self.slideShowOverlay.end.clicked.connect(self.endSlideShow)

            self.imageB = self.__testWidget.listImages.item(0).text()
            #self.overlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB])
            #.scaled(self.overlay.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

            # if the width and height of the image is greater than the width and height of the label
            if self.images[self.imageB].width() > self.slideShowOverlay.image.width() or self.images[self.imageB].height() > self.slideShowOverlay.image.height():
                self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]).scaled(self.slideShowOverlay.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            else:
                self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]))
            self.slideShowOverlay.image.setAlignment(QtCore.Qt.AlignCenter)

    def endSlideShow(self):
        self.slideShowOverlay.close()

    def previousImage(self):
        pass
            
    def nextImage(self):
        if self.imageB != list(self.images.keys())[-1]:
            self.imageB = list(self.images.keys())[list(self.images.keys()).index(self.imageB) + 1]
            if self.images[self.imageB].width() > self.slideShowOverlay.image.width() or self.images[self.imageB].height() > self.slideShowOverlay.image.height():
                self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]).scaled(self.slideShowOverlay.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
            else:
                self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]))
            self.slideShowOverlay.image.setAlignment(QtCore.Qt.AlignCenter)
            if self.imageB == list(self.images.keys())[-1]:
                self.slideShowOverlay.next.hide()
                self.slideShowOverlay.end.show()
        else:
            self.slideShowOverlay.next.hide()
            self.slideShowOverlay.end.show()

    def startWebcam(self):
        if self.videoPath:
            self.captureWeb = cv2.VideoCapture(self.videoPath)
        else:
            self.captureWeb = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        self.captureWeb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.captureWeb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.captureWeb.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.captureWeb.set(cv2.CAP_PROP_FPS, 60)
        self.captureWeb.set(cv2.CAP_PROP_POS_MSEC, 0)

        self.timerWeb = QtCore.QTimer()
        self.timerWeb.timeout.connect(self.update_frame)
        self.timerWeb.start(5)

    def update_frame(self):
        ret, self.image = self.captureWeb.read()
        frame_number = self.captureWeb.get(cv2.CAP_PROP_POS_FRAMES)
        fps = self.captureWeb.get(cv2.CAP_PROP_FPS)

        if ret and not self.isPaused:
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

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.image, f'Angle: {self.angle}', (5, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            self.displayImage(self.image, 1)
            cv2.waitKey(15)
        else:
            self.stopWebcam()

    def stopWebcam(self):
        if self.timerWeb is not None and self.timerWeb.isActive():
            self.timerWeb.stop()
            self.captureWeb.release()
            self.__testWidget.imgLabel.clear()

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

        
    def closeEvent(self, event):
        if self.slideShowOverlay:
            self.slideShowOverlay.close()

        if self.calibrationOverlay:
            self.calibrationOverlay.close()
        event.accept()

    # def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
    #     print(self.size())
    #     return super().resizeEvent(event)    

    def openCalibration(self):
        if self.timerWeb is not None and self.timerWeb.isActive():
            if not self.calibrationOverlay:
                ui = QFile("ui/fullScreenCalibrationOverlay.ui")
                ui.open(QFile.ReadOnly)
                self.calibrationOverlay = self.loader.load(ui)
                ui.close()
                self.calibrationOverlay.setWindowFlags(QtCore.Qt.FramelessWindowHint)
                self.calibrationOverlay.showFullScreen()
                self.calibrationOverlay.marker.setPixmap(QtGui.QPixmap("public/marker.png").scaled(80, 80, QtCore.Qt.KeepAspectRatio))
                self.calibrationOverlay.startCalibration.clicked.connect(self.startCalibration)
                self.calibrationOverlay.endCalibration.clicked.connect(self.endCalibration)
            else:
                self.calibrationOverlay.show()

    def startCalibration(self):
        # TODO: add FIFO queue for calibration points and reset style
        self.calibrationOverlay.bottomLeftWidget.setStyleSheet("QWidget {border-radius: 40px; border: 2px solid red; }")
        self.calibrationOverlay.bottomLeft.setPixmap(QtGui.QPixmap("public/calibration_point.png").scaled(80, 80, QtCore.Qt.KeepAspectRatio))

    def endCalibration(self):
        self.calibrationOverlay.close()


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

# TODO: scan path podobne ako heatmap .. čiarky a body
# TODO: kalibracia a validacia
# TODO: spraviť nejake opatrenie ked je otvoreny overlay aby nenastala nejaka šarapata keby sa vymazalo nieco z obrazku