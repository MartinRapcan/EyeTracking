import sys
import os
import glob
import cv2
import re
import json

from PySide6.QtCore import Qt
from pyqt_frameless_window import FramelessMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, QPalette, QBrush, QIcon, QTransform, QColor, QRegularExpressionValidator
from PySide6.QtCore import QFile, QObject, QThread, Signal, Slot, QTimer, QRegularExpression
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem, QDialog, QHBoxLayout, QLabel, QTextEdit, QPushButton, QWidget, QVBoxLayout, QSplashScreen, QProgressBar, QStyleFactory
from os.path import isfile, join
from pupil_detectors import Detector2D
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
        # i have one image in my folder make it like 1000 images
        images = images * 1000
        video = cv2.VideoWriter(f'dataset/{name}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)
        for image in images:
            img = cv2.imread(path + image)
            video.write(img)
        video.release()

class MainWindow(FramelessMainWindow):
    detector_2d = Detector2D()
    camera = None
    detector_3d = None
    images = {}
    imageB = None
    angle = 0
    loader = QUiLoader()
    imagePath = None
    imageName = None
    folderPath = None
    imageAmount = None
    isPaused = False
    timerWeb = None
    detector_2d_config = {}
    detectionRound = 0
    fillImageList = 0
    imagesPaths = {}
    rawDataFromDetection = {}
    clickedItem = None

    def __init__(self):
        super().__init__()
        self.__testWidget = QWidget()
        ui = QFile("ui/main.ui")
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

        with open('config/config.json') as json_file:
            self.config = json.load(json_file)
            self.detector_2d_config = self.config["detector_2d"]
            self.detector_2d_config["coarse_detection"] = bool(self.detector_2d_config["coarse_detection"])

        # Camera config values
        self.__testWidget.focalLength.setText(str(self.config['focal_length']))

        # Detector 2D config values
        self.__testWidget.coarseDetection.setText(str(bool(self.config["detector_2d"]['coarse_detection'])))
        self.__testWidget.coarseFilterMin.setText(str(self.config["detector_2d"]['coarse_filter_min']))
        self.__testWidget.coarseFilterMax.setText(str(self.config["detector_2d"]['coarse_filter_max'])) 
        self.__testWidget.intensityRange.setText(str(self.config["detector_2d"]['intensity_range']))
        self.__testWidget.blurSize.setText(str(self.config["detector_2d"]['blur_size']))
        self.__testWidget.cannyTreshold.setText(str(self.config["detector_2d"]['canny_threshold']))
        self.__testWidget.cannyRation.setText(str(self.config["detector_2d"]['canny_ration']))
        self.__testWidget.cannyAperture.setText(str(self.config["detector_2d"]['canny_aperture']))
        self.__testWidget.pupilSizeMax.setText(str(self.config["detector_2d"]['pupil_size_max']))
        self.__testWidget.pupilSizeMin.setText(str(self.config["detector_2d"]['pupil_size_min']))
        self.__testWidget.strongPerimeterMin.setText(str(self.config["detector_2d"]['strong_perimeter_ratio_range_min']))
        self.__testWidget.strongPerimeterMax.setText(str(self.config["detector_2d"]['strong_perimeter_ratio_range_max']))
        self.__testWidget.strongAreaMin.setText(str(self.config["detector_2d"]['strong_area_ratio_range_min']))
        self.__testWidget.strongAreaMax.setText(str(self.config["detector_2d"]['strong_area_ratio_range_max']))
        self.__testWidget.contourSizeMin.setText(str(self.config["detector_2d"]['contour_size_min']))
        self.__testWidget.ellipseRoudnessRatio.setText(str(self.config["detector_2d"]['ellipse_roundness_ratio']))
        self.__testWidget.initialEllipseTreshhold.setText(str(self.config["detector_2d"]['initial_ellipse_fit_threshhold']))
        self.__testWidget.finalPerimeterMin.setText(str(self.config["detector_2d"]['final_perimeter_ratio_range_min'])) 
        self.__testWidget.finalPerimeterMax.setText(str(self.config["detector_2d"]['final_perimeter_ratio_range_max']))
        self.__testWidget.ellipseSupportMinDist.setText(str(self.config["detector_2d"]['ellipse_true_support_min_dist']))
        self.__testWidget.supportPixelRatio.setText(str(self.config["detector_2d"]['support_pixel_ratio_exponent']))

        # Validators
        floatingRegex = QRegularExpression("^(0|[1-9]\\d*)(\\.\\d+)?$")
        integerRegex = QRegularExpression("^0|[1-9]\\d*$")
        boolRegex = QRegularExpression("^True|False$")

        # Camera validation
        self.__testWidget.focalLength.setValidator(QRegularExpressionValidator(floatingRegex))
        
        # Detector 2D validation
        self.__testWidget.coarseDetection.setValidator(QRegularExpressionValidator(boolRegex))
        self.__testWidget.coarseFilterMin.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.coarseFilterMax.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.intensityRange.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.blurSize.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.cannyTreshold.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.cannyRation.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.cannyAperture.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.pupilSizeMax.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.pupilSizeMin.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.strongPerimeterMin.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.strongPerimeterMax.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.strongAreaMin.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.strongAreaMax.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.contourSizeMin.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.ellipseRoudnessRatio.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.initialEllipseTreshhold.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.finalPerimeterMin.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.finalPerimeterMax.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.ellipseSupportMinDist.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.supportPixelRatio.setValidator(QRegularExpressionValidator(floatingRegex))

        # Camera event listeners
        self.__testWidget.focalLength.textChanged.connect(self.configChanged)

        # Detector 2D event listeners
        self.__testWidget.coarseDetection.textChanged.connect(self.configChanged)
        self.__testWidget.coarseFilterMin.textChanged.connect(self.configChanged)
        self.__testWidget.coarseFilterMax.textChanged.connect(self.configChanged)
        self.__testWidget.intensityRange.textChanged.connect(self.configChanged)
        self.__testWidget.blurSize.textChanged.connect(self.configChanged)
        self.__testWidget.cannyTreshold.textChanged.connect(self.configChanged)
        self.__testWidget.cannyRation.textChanged.connect(self.configChanged)
        self.__testWidget.cannyAperture.textChanged.connect(self.configChanged)
        self.__testWidget.pupilSizeMax.textChanged.connect(self.configChanged)
        self.__testWidget.pupilSizeMin.textChanged.connect(self.configChanged)
        self.__testWidget.strongPerimeterMin.textChanged.connect(self.configChanged)
        self.__testWidget.strongPerimeterMax.textChanged.connect(self.configChanged)
        self.__testWidget.strongAreaMin.textChanged.connect(self.configChanged)
        self.__testWidget.strongAreaMax.textChanged.connect(self.configChanged)
        self.__testWidget.contourSizeMin.textChanged.connect(self.configChanged)
        self.__testWidget.ellipseRoudnessRatio.textChanged.connect(self.configChanged)
        self.__testWidget.initialEllipseTreshhold.textChanged.connect(self.configChanged)
        self.__testWidget.finalPerimeterMin.textChanged.connect(self.configChanged)
        self.__testWidget.finalPerimeterMax.textChanged.connect(self.configChanged)
        self.__testWidget.ellipseSupportMinDist.textChanged.connect(self.configChanged)
        self.__testWidget.supportPixelRatio.textChanged.connect(self.configChanged)



        # Manipulate config
        # TODO: add reset to default button 
        self.__testWidget.saveParameters.setEnabled(False)
        self.__testWidget.saveParameters.clicked.connect(self.saveParameters)


        self.detector_2d.update_properties(self.detector_2d_config)
        self.camera = CameraModel(focal_length=self.config['focal_length'], resolution=[640, 480])
        self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)


        #print(dir(titleBar))
        #print(titleBar.children())
        # use findchildren to find QPushButtons
        for button in titleBar.findChildren(QPushButton):
            button.setStyleSheet("QPushButton {background-color: #FFE81F; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:hover {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:pressed {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px}")
        titleBar.findChildren(QLabel)[1].setStyleSheet("QLabel {font-size: 15px; color: #F7FAFC; font-weight: bold; margin-left: 10px}")
        titleBar.findChildren(QLabel)[0].setStyleSheet("QLabel {margin-left: 10px}")
        self.image = None
        self.__testWidget.listImages.itemClicked.connect(self.imageClicked)
        self.__testWidget.startButton.setEnabled(False)
        self.__testWidget.startButton.clicked.connect(self.startDetection)
        self.__testWidget.stopButton.clicked.connect(self.stopDetection)
        self.__testWidget.loadImage.clicked.connect(self.loadImage)
        self.__testWidget.imagePath.setText("No image selected")
        self.__testWidget.imagePath.setText(self.__testWidget.imagePath.fontMetrics().elidedText(self.__testWidget.imagePath.text(), Qt.ElideRight, self.__testWidget.imagePath.width()))
        #self.__testWidget.removeImage.clicked.connect(self.removeImage)
        #self.__testWidget.slideShow.clicked.connect(self.slideShow)
        #self.__testWidget.videoPicker.clicked.connect(self.videoPicker)
        #self.__testWidget.startCalibrationButton.clicked.connect(self.openCalibration)
        #self.slideShowOverlay = None
        #self.calibrationOverlay = None

    def configChanged(self):
        pupil_size_min = int(self.__testWidget.pupilSizeMin.text()) if self.__testWidget.pupilSizeMin.text() != "" else None
        pupil_size_max = int(self.__testWidget.pupilSizeMax.text()) if self.__testWidget.pupilSizeMax.text() != "" else None
        strong_perimeter_min = float(self.__testWidget.strongPerimeterMin.text()) if self.__testWidget.strongPerimeterMin.text() != "" else None
        strong_perimeter_max = float(self.__testWidget.strongPerimeterMax.text()) if self.__testWidget.strongPerimeterMax.text() != "" else None
        strong_area_min = float(self.__testWidget.strongAreaMin.text()) if self.__testWidget.strongAreaMin.text() != "" else None
        strong_area_max = float(self.__testWidget.strongAreaMax.text()) if self.__testWidget.strongAreaMax.text() != "" else None
        final_perimeter_min = float(self.__testWidget.finalPerimeterMin.text()) if self.__testWidget.finalPerimeterMin.text() != "" else None
        final_perimeter_max = float(self.__testWidget.finalPerimeterMax.text()) if self.__testWidget.finalPerimeterMax.text() != "" else None

        if self.__testWidget.focalLength.text() != "" and self.__testWidget.focalLength.text()[-1] != "." \
            and self.__testWidget.intensityRange.text() != "" \
            and self.__testWidget.pupilSizeMax.text() != "" \
            and self.__testWidget.pupilSizeMin.text() != "" \
            and self.__testWidget.blurSize.text() != "" \
            and self.__testWidget.cannyTreshold.text() != "" \
            and self.__testWidget.cannyRation.text() != "" \
            and self.__testWidget.cannyAperture.text() != "" \
            and self.__testWidget.coarseFilterMin.text() != "" \
            and self.__testWidget.coarseFilterMax.text() != "" \
            and self.__testWidget.coarseDetection.text() == "True" or self.__testWidget.coarseDetection.text() == "False" \
            and self.__testWidget.contourSizeMin.text() != "" \
            and self.__testWidget.strongPerimeterMin.text() != "" and self.__testWidget.strongPerimeterMin.text()[-1] != "." \
            and self.__testWidget.strongPerimeterMax.text() != "" and self.__testWidget.strongPerimeterMax.text()[-1] != "." \
            and self.__testWidget.strongAreaMin.text() != "" and self.__testWidget.strongAreaMin.text()[-1] != "." \
            and self.__testWidget.strongAreaMax.text() != "" and self.__testWidget.strongAreaMax.text()[-1] != "." \
            and self.__testWidget.ellipseRoudnessRatio.text != "" \
            and self.__testWidget.initialEllipseTreshhold.text() != "" \
            and self.__testWidget.finalPerimeterMin.text() != "" and self.__testWidget.finalPerimeterMin.text()[-1] != "." \
            and self.__testWidget.finalPerimeterMax.text() != "" and self.__testWidget.finalPerimeterMax.text()[-1] != "." \
            and self.__testWidget.ellipseSupportMinDist.text() != "" \
            and self.__testWidget.supportPixelRatio.text() != "":
            if pupil_size_min is not None and pupil_size_max is not None and pupil_size_max > pupil_size_min \
                and strong_perimeter_min is not None and strong_perimeter_max is not None and strong_perimeter_max > strong_perimeter_min \
                    and strong_area_min is not None and strong_area_max is not None and strong_area_max > strong_area_min \
                        and final_perimeter_min is not None and final_perimeter_max is not None and final_perimeter_max > final_perimeter_min:
                self.__testWidget.saveParameters.setEnabled(True)
            else:
                self.__testWidget.saveParameters.setEnabled(False)

        else:
            self.__testWidget.saveParameters.setEnabled(False)
    
    def saveParameters(self):
        self.config['focal_length'] = float(self.__testWidget.focalLength.text())
        self.config["detector_2d"]['intensity_range'] = int(self.__testWidget.intensityRange.text())
        self.config["detector_2d"]['pupil_size_max'] = int(self.__testWidget.pupilSizeMax.text())
        self.config["detector_2d"]['pupil_size_min'] = int(self.__testWidget.pupilSizeMin.text())
        self.config["detector_2d"]['blur_size'] = int(self.__testWidget.blurSize.text())
        self.config["detector_2d"]['canny_threshold'] = int(self.__testWidget.cannyTreshold.text())
        self.config["detector_2d"]['canny_ration'] = int(self.__testWidget.cannyRation.text())
        self.config["detector_2d"]['canny_aperture'] = int(self.__testWidget.cannyAperture.text())
        self.config["detector_2d"]['coarse_filter_min'] = int(self.__testWidget.coarseFilterMin.text())
        self.config["detector_2d"]['coarse_filter_max'] = int(self.__testWidget.coarseFilterMax.text())
        self.config["detector_2d"]['coarse_detection'] = int(self.__testWidget.coarseDetection.text() == "True")
        self.config["detector_2d"]['contour_size_min'] = int(self.__testWidget.contourSizeMin.text())
        self.config["detector_2d"]['strong_perimeter_ratio_range_min'] = float(self.__testWidget.strongPerimeterMin.text())
        self.config["detector_2d"]['strong_perimeter_ratio_range_max'] = float(self.__testWidget.strongPerimeterMax.text())
        self.config["detector_2d"]['strong_area_ratio_range_min'] = float(self.__testWidget.strongAreaMin.text())
        self.config["detector_2d"]['strong_area_ratio_range_max'] = float(self.__testWidget.strongAreaMax.text())
        self.config["detector_2d"]['ellipse_roundness_ratio'] = float(self.__testWidget.ellipseRoudnessRatio.text())
        self.config["detector_2d"]['initial_ellipse_fit_threshhold'] = float(self.__testWidget.initialEllipseTreshhold.text())
        self.config["detector_2d"]['final_perimeter_ratio_range_min'] = float(self.__testWidget.finalPerimeterMin.text())
        self.config["detector_2d"]['final_perimeter_ratio_range_max'] = float(self.__testWidget.finalPerimeterMax.text())
        self.config["detector_2d"]['ellipse_true_support_min_dist'] = float(self.__testWidget.ellipseSupportMinDist.text())
        self.config["detector_2d"]['support_pixel_ratio_exponent'] = float(self.__testWidget.supportPixelRatio.text())
        with open('config/config.json', 'w') as outfile:
            json.dump(self.config, outfile)
        self.detector_2d_config = self.config["detector_2d"]
        self.detector_2d_config["coarse_detection"] = bool(self.detector_2d_config["coarse_detection"])
        self.detector_2d.update_properties(self.detector_2d_config)
        self.camera = CameraModel(focal_length=self.config['focal_length'], resolution=[640, 480])
        self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
        self.detectionRound = 0

        if self.clickedItem:
            self.__testWidget.imageLabel.clear()
            self.imageClicked(self.clickedItem)
        self.rawDataFromDetection = {}
        self.__testWidget.saveParameters.setEnabled(False)

    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if fname[0] != "":
            self.imageName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
            self.imagePath = fname[0]
            self.rawDataFromDetection = {}
            self.fillImageList = 0
            self.detectionRound = 0
            self.imagesPaths = {}
            self.clickedItem = None 
            self.folderPath = os.path.dirname(fname[0])
            file_list = glob.glob(os.path.join(self.folderPath, "*"))
            self.imageAmount = len(file_list)
            self.__testWidget.startButton.setEnabled(True)
            self.__testWidget.imageLabel.clear()
            self.__testWidget.listImages.clear()
            self.__testWidget.imagePath.setText(self.imageName)
            self.__testWidget.imagePath.setText(self.__testWidget.imagePath.fontMetrics().elidedText(self.__testWidget.imagePath.text(), Qt.ElideRight, self.__testWidget.imagePath.width()))
        else:
            self.imagePath = None
            self.folderPath = None
            self.imageAmount = 0
            self.fillImageList = 0
            self.clickedItem = None
            self.detectionRound = 0
            self.imagesPaths = {}
            self.rawDataFromDetection = {}
            self.__testWidget.imagePath.setText("No image selected")
            self.__testWidget.startButton.setEnabled(False)
            self.__testWidget.listImages.clear()
            self.__testWidget.imageLabel.clear()
                #self.images[imageName] = QtGui.QImage(fname[0])
                #self.__testWidget.listImages.addItem(imageName)

    # def videoPicker(self):
    #     fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Video files (*.mp4 *.avi *.mov *.mkv)")
    #     if fname[0] != "":
    #         videoName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
    #         self.videoPath = fname[0]
    #         self.__testWidget.videoPath.setText(videoName)
    #     else:
    #         self.videoPath = None
    #         self.__testWidget.videoPath.setText("")

    # def removeImage(self):
    #     if self.__testWidget.listImages.currentItem():
    #         self.images.pop(self.__testWidget.listImages.currentItem().text())
    #         self.__testWidget.listImages.takeItem(self.__testWidget.listImages.currentRow())

    # def slideShow(self):
    #     if self.timerWeb is not None and self.timerWeb.isActive() and self.__testWidget.listImages.count() > 0:
    #         if not self.slideShowOverlay:
    #             ui = QFile("ui/fullScreenSlideShowOverlay.ui")
    #             ui.open(QFile.ReadOnly)
    #             self.slideShowOverlay = self.loader.load(ui)
    #             ui.close()
    #         if self.__testWidget.listImages.count() > 1:
    #             self.slideShowOverlay.next.show()
    #             self.slideShowOverlay.end.hide()
    #         else:
    #             self.slideShowOverlay.next.hide()
    #             self.slideShowOverlay.end.show()
    #         self.slideShowOverlay.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    #         self.slideShowOverlay.showFullScreen()
    #         self.slideShowOverlay.marker.setPixmap(QtGui.QPixmap("public/marker.png").scaled(80, 80, QtCore.Qt.KeepAspectRatio))
    #         self.slideShowOverlay.next.clicked.connect(self.nextImage)
    #         self.slideShowOverlay.end.clicked.connect(self.endSlideShow)

    #         self.imageB = self.__testWidget.listImages.item(0).text()
    #         #self.overlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB])
    #         #.scaled(self.overlay.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))

    #         # if the width and height of the image is greater than the width and height of the label
    #         if self.images[self.imageB].width() > self.slideShowOverlay.image.width() or self.images[self.imageB].height() > self.slideShowOverlay.image.height():
    #             self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]).scaled(self.slideShowOverlay.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    #         else:
    #             self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]))
    #         self.slideShowOverlay.image.setAlignment(QtCore.Qt.AlignCenter)

    # def endSlideShow(self):
    #     self.slideShowOverlay.close()

    # def previousImage(self):
    #     pass
            
    # def nextImage(self):
    #     if self.imageB != list(self.images.keys())[-1]:
    #         self.imageB = list(self.images.keys())[list(self.images.keys()).index(self.imageB) + 1]
    #         if self.images[self.imageB].width() > self.slideShowOverlay.image.width() or self.images[self.imageB].height() > self.slideShowOverlay.image.height():
    #             self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]).scaled(self.slideShowOverlay.image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
    #         else:
    #             self.slideShowOverlay.image.setPixmap(QtGui.QPixmap.fromImage(self.images[self.imageB]))
    #         self.slideShowOverlay.image.setAlignment(QtCore.Qt.AlignCenter)
    #         if self.imageB == list(self.images.keys())[-1]:
    #             self.slideShowOverlay.next.hide()
    #             self.slideShowOverlay.end.show()
    #     else:
    #         self.slideShowOverlay.next.hide()
    #         self.slideShowOverlay.end.show()

    def startDetection(self):
        if self.imagePath:
            if self.detectionRound == 0:
                self.renderImage()
                self.fillImageList = 1
                self.detectionRound = 1
                self.renderImage()
            else:
                self.renderImage()

    def renderImage(self):
        fileName = self.imageName.split("_")
        fileNumber = int(fileName[1].split(".")[0])
        fileFormat = self.imageName.split(".")[-1]
        for i in range(fileNumber, self.imageAmount):
            newPath = self.folderPath + "/" + fileName[0] + "_" + str(i) + "." + fileFormat
            if self.fillImageList == 0:
                listImageName = fileName[0] + "_" + str(i) + "." + fileFormat
                self.__testWidget.listImages.addItem(listImageName)
                self.imagesPaths[listImageName] = newPath

            image = cv2.imread(newPath)
            # read video frame as numpy array
            grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # run 2D detector on video frame
            result_2d = self.detector_2d.detect(grayscale_array)
            result_2d["timestamp"] = i
            # pass 2D detection result to 3D detector
            result_3d = self.detector_3d.update_and_detect(result_2d, grayscale_array, apply_refraction_correction=False)

            if self.detectionRound == 1:
                self.rawDataFromDetection[i] = result_3d
            ellipse_3d = result_3d["ellipse"]
            # draw 3D detection result on eye frame
            cv2.ellipse(
                image,
                tuple(int(v) for v in ellipse_3d["center"]),
                tuple(int(v / 2) for v in ellipse_3d["axes"]),
                ellipse_3d["angle"],
                0,
                360,  # start/end angle for drawing
                (0, 255, 0),  # color (BGR): red
            )

            self.displayImage(image, 1)
            cv2.waitKey(10)

    def imageClicked(self, item):
        self.clickedItem = item

        image = cv2.imread(self.imagesPaths[item.text()])
        # read video frame as numpy array
        grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # run 2D detector on video frame
        result_2d = self.detector_2d.detect(grayscale_array)

        result_2d["timestamp"] = list(self.imagesPaths.keys()).index(item.text())
        # pass 2D detection result to 3D detector
        result_3d = self.detector_3d.update_and_detect(result_2d, grayscale_array, apply_refraction_correction=False)

        ellipse_3d = result_3d["ellipse"]
        # draw 3D detection result on eye frame
        cv2.ellipse(
            image,
            tuple(int(v) for v in ellipse_3d["center"]),
            tuple(int(v / 2) for v in ellipse_3d["axes"]),
            ellipse_3d["angle"],
            0,
            360,  # start/end angle for drawing
            (0, 255, 0),  # color (BGR): red
        )

        self.displayImage(image, 1)


        # if self.imagePath:
        #     self.captureWeb = cv2.VideoCapture(self.imagePath)
        # else:
        #     self.captureWeb = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # self.captureWeb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.captureWeb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.captureWeb.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # self.captureWeb.set(cv2.CAP_PROP_FPS, 60)
        # self.captureWeb.set(cv2.CAP_PROP_POS_MSEC, 0)

        # self.timerWeb = QtCore.QTimer()
        # self.timerWeb.timeout.connect(self.update_frame)
        # self.timerWeb.start(5)

    # def update_frame(self):
    #     ret, self.image = self.captureWeb.read()
    #     frame_number = self.captureWeb.get(cv2.CAP_PROP_POS_FRAMES)
    #     fps = self.captureWeb.get(cv2.CAP_PROP_FPS)

    #     if ret and not self.isPaused:
    #         grayscale_array = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    #         result_2d = self.detector_2d.detect(grayscale_array)
    #         result_2d["timestamp"] = frame_number / fps
    #         result_3d = self.detector_3d.update_and_detect(result_2d, grayscale_array)
    #         ellipse_3d = result_3d["ellipse"]
    #         self.angle = round(float(ellipse_3d["angle"]), 2)

    #         cv2.ellipse(
    #             self.image,
    #             tuple(int(v) for v in ellipse_3d["center"]),
    #             tuple(int(v / 2) for v in ellipse_3d["axes"]),
    #             ellipse_3d["angle"],
    #             0,
    #             360,  # start/end angle for drawing
    #             (0, 255, 0),  # color (BGR): red
    #         )
    #         cv2.circle(
    #             self.image,
    #             tuple(int(v) for v in ellipse_3d["center"]),
    #             2,
    #             (0, 0, 255),  # color (BGR): blue
    #             thickness=-2,
    #         )

    #         font = cv2.FONT_HERSHEY_SIMPLEX
    #         cv2.putText(self.image, f'Angle: {self.angle}', (5, 30), font, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    #         self.displayImage(self.image, 1)
    #         cv2.waitKey(15)
    #     else:
    #         self.stopWebcam()

    def stopDetection(self):
        pass
        # if self.timerWeb is not None and self.timerWeb.isActive():
        #     self.timerWeb.stop()
        #     self.captureWeb.release()
        #     self.__testWidget.imgLabel.clear()

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
            self.__testWidget.imageLabel.setPixmap(QtGui.QPixmap.fromImage(outImage))
            self.__testWidget.imageLabel.setScaledContents(True)
        # else:
        #     self.__testWidget.imagePreview.setPixmap(QtGui.QPixmap.fromImage(outImage))
        #     self.__testWidget.imagePreview.setScaledContents(True)

        
    # def closeEvent(self, event):
    #     if self.slideShowOverlay:
    #         self.slideShowOverlay.close()

    #     if self.calibrationOverlay:
    #         self.calibrationOverlay.close()
    #     event.accept()

    # def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
    #     print(self.size())
    #     return super().resizeEvent(event)    

    # def openCalibration(self):
    #     if self.timerWeb is not None and self.timerWeb.isActive():
    #         if not self.calibrationOverlay:
    #             ui = QFile("ui/fullScreenCalibrationOverlay.ui")
    #             ui.open(QFile.ReadOnly)
    #             self.calibrationOverlay = self.loader.load(ui)
    #             ui.close()
    #             self.calibrationOverlay.setWindowFlags(QtCore.Qt.FramelessWindowHint)
    #             self.calibrationOverlay.showFullScreen()
    #             self.calibrationOverlay.marker.setPixmap(QtGui.QPixmap("public/marker.png").scaled(80, 80, QtCore.Qt.KeepAspectRatio))
    #             self.calibrationOverlay.startCalibration.clicked.connect(self.startCalibration)
    #             self.calibrationOverlay.endCalibration.clicked.connect(self.endCalibration)
    #         else:
    #             self.calibrationOverlay.show()

    # def startCalibration(self):
    #     # TODO: add FIFO queue for calibration points and reset style
    #     self.calibrationOverlay.bottomLeftWidget.setStyleSheet("QWidget {border-radius: 40px; border: 2px solid red; }")
    #     self.calibrationOverlay.bottomLeft.setPixmap(QtGui.QPixmap("public/calibration_point.png").scaled(80, 80, QtCore.Qt.KeepAspectRatio))

    # def endCalibration(self):
    #     self.calibrationOverlay.close()


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
# TODO: transforms lib knižnica
# TODO: prevod medzi lokal a global coord systemom
# TODO: neskor pridať dlib na detekciu zrenice .. funguje na zaklade machine learningu
# TODO: pre kameru pridať velkosť obrazku do configu
# TODO: filter requirements