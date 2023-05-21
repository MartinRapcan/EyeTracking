import sys
import os
import glob
import cv2
import re
import json
import numpy as np
import csv

from math import sqrt, atan2, cos, sin

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, QRegularExpressionValidator
from PySide6.QtCore import QFile, QRegularExpression, Qt, QCoreApplication, QObject
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QWidget, QButtonGroup, QColorDialog, QVBoxLayout
from pyqt_frameless_window import FramelessMainWindow

from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

from scipy.spatial.transform import Rotation

from matplotlib import pyplot, use
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Text3D
use('Agg')


class GlobalSharedClass():
    # TODO: Singleton
    _instance = None

    def __init__(self):
        # UI LoadercameraPos
        self.loader = QUiLoader()

        # Constants
        self.planeNormal = np.array([0, 1, 0])
        self.planeCenter = np.array([0, -500, 0])
        self.planeRot = np.array([0, 0, 180])
        self.cameraPos = np.array([20, -50, -10])
        self.cameraRotMat = np.array([
            [0.884918212890625, -0.105633445084095, -0.4536091983318329],
            [0.4657464325428009, 0.20070354640483856, 0.8618574738502502],
            [0.0, -0.973940372467041, 0.22680459916591644]
        ])
        self.displaySize = (250, 250) #width, height
        self.displayPos = np.array([0, -500, 0])
        self.displayRot = np.array([0, 0, 180])
        self.displayRotMat = self.eulerToRot(self.displayRot)
        self.displayNormalLocal = np.array([0, -1, 0])
        self.displayNormalWorld = self.normalize(self.rotate(self.displayNormalLocal, self.displayRotMat))
        self.cameraDirsWorld = (
            self.rotate(np.array((1, 0, 0)), self.cameraRotMat),
            self.rotate(np.array((0, 1, 0)), self.cameraRotMat),
            self.rotate(np.array((0, 0, 1)), self.cameraRotMat)
        )

        # Validators
        self.radiusRegex = QRegularExpression("^[1-9][0-9]?$|^100$")
        self.floatingRegex = QRegularExpression("^(0|[1-9]\\d*)(\\.\\d+)?$")
        self.integerRegex = QRegularExpression("^0|[1-9]\\d*$")
        self.boolRegex = QRegularExpression("^True|False$")
        self.detectorModeRegex = QRegularExpression("^blocking|asynchronous$")
        self.graphParamRegex = QRegularExpression("^([0-9]|[1-9][0-9]|1[0-9][0-9]|2[0-9][0-9]|3[0-5][0-9]|360)$")
        self.thresholdRegex = QRegularExpression("^(1?\d{1,2}|2[0-4]\d|25[0-5])$")

    def distance(self, p1, p2):
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def dir_vector(self, vec1, vec2):
        return [vec2[0] - vec1[0], vec2[1] - vec1[1], vec2[2] - vec1[2]]

    def lookAt(self, camera, target, up):
        forward = target - camera
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)

        new_up = np.cross(right, forward)

        result = np.identity(4)
        result[0][0] = right[0]
        result[0][1] = right[1]
        result[0][2] = right[2]

        result[1][0] = new_up[0]
        result[1][1] = new_up[1]
        result[1][2] = new_up[2]

        result[2][0] = -forward[0]
        result[2][1] = -forward[1]
        result[2][2] = -forward[2]

        translation = np.identity(4)
        translation[0][3] = -camera[0]
        translation[1][3] = -camera[1]
        translation[2][3] = -camera[2]

        lookAt_matrix = np.matmul(result, translation)
        # Extract the rotation submatrix from the look-at matrix
        rot_matrix = lookAt_matrix[:3, :3]
        
        # Convert the rotation to Euler angles using the zxy convention
        theta_z = np.arctan2(-rot_matrix[0, 1], rot_matrix[0, 0])
        theta_x = np.arctan2(-rot_matrix[1, 2], rot_matrix[2, 2])
        theta_y = np.arcsin(rot_matrix[0, 2])

        # Convert the angles to degrees and print the result
        euler_angles = np.array([theta_x, theta_y, theta_z]) * 180 / np.pi

        return euler_angles


    def transform(self, p, position, rotMat):
        return self.rotate(p, rotMat) + position
    
    def inverseTransform(self, p, position, rotMat):
        return (p - position) @ rotMat #inverse rotation
        
    def rotate(self, p, rotMat):
        return p @ rotMat.T

    def transfer_vector(self, vec, position, rotation):
        return vec @ self.eulerToRot(rotation) + position

    def eulerToRot(self, theta, degrees=True) :
        r = Rotation.from_euler("zxy", (theta[2], theta[0], theta[1]), degrees)
        return r.as_matrix()

    def intersectPlane(self, n, p0, l0, l):
        denom = self.matmul(-n, l)
        if (denom > sys.float_info.min):
            p0l0 = p0 - l0
            t = self.matmul(p0l0, -n) / denom
            return t
        return -1.0
            
    def matmul(self, v1, v2, pad=False, padBy=1.0):
        if(pad is True):
            return np.matmul(v1, np.append(v2, padBy))[:-1]
        return np.matmul(v1, v2)
        
    def getPoint(self, ray, distance):
        return ray[0] + ray[1] * distance

    def normalize(self, v):
        return v / self.magnitude(v)
        
    def magnitude(self, v):
        return np.sqrt(self.sqrMagnitude(v))
            
    def sqrMagnitude(self, v):
        return self.matmul(v, v)
    
    def lerp(self, a, b, t):
        return (1 - t) * a + t * b

    def convert_uv_to_px(self, uv_data, width, height):
        return (int(uv_data[0] * width), int(uv_data[1] * height))

    def convert_to_uv(self, vec, size_x=250, size_y=250, flip_y=True, includeOutliers=False):
        x = (vec[0] + size_x / 2) / size_x
        y = (vec[2] + size_y / 2) / size_y
        if flip_y:
            y = 1 - y

        if not includeOutliers:
            if x < 0 or x > 1 or y < 0 or y > 1:
                return None
        return (x, y)
    
    def setupTitleBar(self, outerClass):
        outerClass.getTitleBar().setFixedHeight(35)
        for button in outerClass.getTitleBar().findChildren(QPushButton):
            button.setStyleSheet("QPushButton {background-color: #FFE81F; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:hover {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:pressed {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px}")
        outerClass.getTitleBar().findChildren(QLabel)[1].setStyleSheet("QLabel {font-size: 15px; color: #F7FAFC; font-weight: bold; margin-left: 10px}")
        outerClass.getTitleBar().findChildren(QLabel)[0].setStyleSheet("QLabel {margin-left: 10px}")

class MainWindow(FramelessMainWindow, GlobalSharedClass):
    def __init__(self):
        GlobalSharedClass.__init__(self)
        super().__init__()

        self.detector_2d = Detector2D()
        self.previewDetector2d = Detector2D()
        self.camera = None
        self.detector_3d = None
        self.images = {}
        self.imageB = None
        self.angle = 0
        self.imagePath = None
        self.imageName = None
        self.folderPath = None
        self.imageAmount = None
        self.isPaused = False
        self.timerWeb = None
        self.detector_2d_config = {}
        self.detectionRound = 0
        self.fillImageList = 0
        self.imagesPaths = {}
        self.rawDataFromDetection = {}
        self.clickedItem = None
        self.image = None
        self.imageFlag = 'Simple'
        self.lastDetectionImage = None
        self.isRunning = False
        self.openedWindows = []
        self.pointsOnDisplay = []

        self.__mainWidget = QWidget()
        ui = QFile("ui/main.ui")
        ui.open(QFile.ReadOnly)
        self.__mainWidget = self.loader.load(ui)
        ui.close()
        self.setWindowTitle('Eye Tracking')
        self.setWindowIcon('public/light.png')
        self.mainWidget = self.centralWidget()
        lay = self.mainWidget.layout()
        lay.addWidget(self.__mainWidget)
        self.mainWidget.setLayout(lay)
        self.setCentralWidget(self.mainWidget)
        self.setGeometry(200, 50, 1100, 735)
        self.setFixedSize(1100, 735)
        
        with open('config/config.json') as json_file:
            self.config = json.load(json_file)
            json_file.seek(0)
            self.original_config = json.load(json_file)
            self.detector_2d_config = self.config["detector_2d"].copy()
            self.detector_2d_config["coarse_detection"] = bool(self.detector_2d_config["coarse_detection"])

            self.detector_3d_config = self.config["detector_3d"].copy()
            self.detector_3d_config["long_term_mode"] = DetectorMode.blocking if int(self.detector_3d_config["long_term_mode"]) == 0 else DetectorMode.asynchronous
            self.detector_3d_config["calculate_rms_residual"] = bool(self.detector_3d_config["calculate_rms_residual"])
       
        with open('config/default.json') as json_file:
            self.default_config = json.load(json_file)

        # Graph config values
        self.__mainWidget.elev.setText(str(self.config['elev']))
        self.__mainWidget.azim.setText(str(self.config['azim']))
        self.__mainWidget.scaleFactor.setText(str(self.config['scaleFactor']))

        # Camera config values
        self.__mainWidget.focalLength.setText(str(self.config['focal_length']))

        # Detector 2D config values
        self.__mainWidget.coarseDetection.setText(str(bool(self.config["detector_2d"]['coarse_detection'])))
        self.__mainWidget.coarseFilterMin.setText(str(self.config["detector_2d"]['coarse_filter_min']))
        self.__mainWidget.coarseFilterMax.setText(str(self.config["detector_2d"]['coarse_filter_max'])) 
        self.__mainWidget.intensityRange.setText(str(self.config["detector_2d"]['intensity_range']))
        self.__mainWidget.blurSize.setText(str(self.config["detector_2d"]['blur_size']))
        self.__mainWidget.cannyTreshold.setText(str(self.config["detector_2d"]['canny_threshold']))
        self.__mainWidget.cannyRation.setText(str(self.config["detector_2d"]['canny_ration']))
        self.__mainWidget.cannyAperture.setText(str(self.config["detector_2d"]['canny_aperture']))
        self.__mainWidget.pupilSizeMax.setText(str(self.config["detector_2d"]['pupil_size_max']))
        self.__mainWidget.pupilSizeMin.setText(str(self.config["detector_2d"]['pupil_size_min']))
        self.__mainWidget.strongPerimeterMin.setText(str(self.config["detector_2d"]['strong_perimeter_ratio_range_min']))
        self.__mainWidget.strongPerimeterMax.setText(str(self.config["detector_2d"]['strong_perimeter_ratio_range_max']))
        self.__mainWidget.strongAreaMin.setText(str(self.config["detector_2d"]['strong_area_ratio_range_min']))
        self.__mainWidget.strongAreaMax.setText(str(self.config["detector_2d"]['strong_area_ratio_range_max']))
        self.__mainWidget.contourSizeMin.setText(str(self.config["detector_2d"]['contour_size_min']))
        self.__mainWidget.ellipseRoudnessRatio.setText(str(self.config["detector_2d"]['ellipse_roundness_ratio']))
        self.__mainWidget.initialEllipseTreshhold.setText(str(self.config["detector_2d"]['initial_ellipse_fit_threshhold']))
        self.__mainWidget.finalPerimeterMin.setText(str(self.config["detector_2d"]['final_perimeter_ratio_range_min'])) 
        self.__mainWidget.finalPerimeterMax.setText(str(self.config["detector_2d"]['final_perimeter_ratio_range_max']))
        self.__mainWidget.ellipseSupportMinDist.setText(str(self.config["detector_2d"]['ellipse_true_support_min_dist']))
        self.__mainWidget.supportPixelRatio.setText(str(self.config["detector_2d"]['support_pixel_ratio_exponent']))

        # Detector 3D config values
        self.__mainWidget.thresholdSwirski.setText(str(self.config["detector_3d"]['threshold_swirski']))
        self.__mainWidget.thresholdKalman.setText(str(self.config["detector_3d"]['threshold_kalman']))
        self.__mainWidget.thresholdShortTerm.setText(str(self.config["detector_3d"]['threshold_short_term']))
        self.__mainWidget.thresholdLongTerm.setText(str(self.config["detector_3d"]['threshold_long_term']))
        self.__mainWidget.longTermBufferSize.setText(str(self.config["detector_3d"]['long_term_buffer_size']))
        self.__mainWidget.longTermForgetTime.setText(str(self.config["detector_3d"]['long_term_forget_time']))
        self.__mainWidget.longTermForgetObservations.setText(str(self.config["detector_3d"]['long_term_forget_observations']))
        self.__mainWidget.longTermMode.setText("asynchronous" if str(self.config["detector_3d"]['long_term_mode']) == "1" else "blocking")
        self.__mainWidget.modelUpdateIntervalLongTerm.setText(str(self.config["detector_3d"]['model_update_interval_long_term']))
        self.__mainWidget.modelUpdateIntervalUltLongTerm.setText(str(self.config["detector_3d"]['model_update_interval_ult_long_term']))
        self.__mainWidget.modelWarmupDuration.setText(str(self.config["detector_3d"]['model_warmup_duration']))
        self.__mainWidget.calculateRmsResidual.setText(str(bool(self.config["detector_3d"]['calculate_rms_residual'])))

        # Graph validation
        self.__mainWidget.elev.setValidator(QRegularExpressionValidator(self.graphParamRegex))
        self.__mainWidget.azim.setValidator(QRegularExpressionValidator(self.graphParamRegex))
        self.__mainWidget.scaleFactor.setValidator(QRegularExpressionValidator(self.floatingRegex))

        # Camera validation
        self.__mainWidget.focalLength.setValidator(QRegularExpressionValidator(self.floatingRegex))
        
        # Detector 2D validation
        self.__mainWidget.coarseDetection.setValidator(QRegularExpressionValidator(self.boolRegex))
        self.__mainWidget.coarseFilterMin.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.coarseFilterMax.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.intensityRange.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.blurSize.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.cannyTreshold.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.cannyRation.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.cannyAperture.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.pupilSizeMax.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.pupilSizeMin.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.strongPerimeterMin.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.strongPerimeterMax.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.strongAreaMin.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.strongAreaMax.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.contourSizeMin.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.ellipseRoudnessRatio.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.initialEllipseTreshhold.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.finalPerimeterMin.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.finalPerimeterMax.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.ellipseSupportMinDist.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.supportPixelRatio.setValidator(QRegularExpressionValidator(self.floatingRegex))

        # Detector 3D validation
        self.__mainWidget.thresholdSwirski.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.thresholdKalman.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.thresholdShortTerm.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.thresholdLongTerm.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.longTermBufferSize.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.longTermForgetTime.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.longTermForgetObservations.setValidator(QRegularExpressionValidator(self.integerRegex))
        self.__mainWidget.longTermMode.setValidator(QRegularExpressionValidator(self.detectorModeRegex))
        self.__mainWidget.modelUpdateIntervalLongTerm.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.modelUpdateIntervalUltLongTerm.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.modelWarmupDuration.setValidator(QRegularExpressionValidator(self.floatingRegex))
        self.__mainWidget.calculateRmsResidual.setValidator(QRegularExpressionValidator(self.boolRegex))

        # Graph event listeners
        self.__mainWidget.elev.textChanged.connect(self.configChanged)
        self.__mainWidget.azim.textChanged.connect(self.configChanged)
        self.__mainWidget.scaleFactor.textChanged.connect(self.configChanged)

        # Camera event listeners
        self.__mainWidget.focalLength.textChanged.connect(self.configChanged)

        # Detector 2D event listeners
        self.__mainWidget.coarseDetection.textChanged.connect(self.configChanged)
        self.__mainWidget.coarseFilterMin.textChanged.connect(self.configChanged)
        self.__mainWidget.coarseFilterMax.textChanged.connect(self.configChanged)
        self.__mainWidget.intensityRange.textChanged.connect(self.configChanged)
        self.__mainWidget.blurSize.textChanged.connect(self.configChanged)
        self.__mainWidget.cannyTreshold.textChanged.connect(self.configChanged)
        self.__mainWidget.cannyRation.textChanged.connect(self.configChanged)
        self.__mainWidget.cannyAperture.textChanged.connect(self.configChanged)
        self.__mainWidget.pupilSizeMax.textChanged.connect(self.configChanged)
        self.__mainWidget.pupilSizeMin.textChanged.connect(self.configChanged)
        self.__mainWidget.strongPerimeterMin.textChanged.connect(self.configChanged)
        self.__mainWidget.strongPerimeterMax.textChanged.connect(self.configChanged)
        self.__mainWidget.strongAreaMin.textChanged.connect(self.configChanged)
        self.__mainWidget.strongAreaMax.textChanged.connect(self.configChanged)
        self.__mainWidget.contourSizeMin.textChanged.connect(self.configChanged)
        self.__mainWidget.ellipseRoudnessRatio.textChanged.connect(self.configChanged)
        self.__mainWidget.initialEllipseTreshhold.textChanged.connect(self.configChanged)
        self.__mainWidget.finalPerimeterMin.textChanged.connect(self.configChanged)
        self.__mainWidget.finalPerimeterMax.textChanged.connect(self.configChanged)
        self.__mainWidget.ellipseSupportMinDist.textChanged.connect(self.configChanged)
        self.__mainWidget.supportPixelRatio.textChanged.connect(self.configChanged)

        # Detector 3D event listeners
        self.__mainWidget.thresholdSwirski.textChanged.connect(self.configChanged)
        self.__mainWidget.thresholdKalman.textChanged.connect(self.configChanged)
        self.__mainWidget.thresholdShortTerm.textChanged.connect(self.configChanged)
        self.__mainWidget.thresholdLongTerm.textChanged.connect(self.configChanged)
        self.__mainWidget.longTermBufferSize.textChanged.connect(self.configChanged)
        self.__mainWidget.longTermForgetTime.textChanged.connect(self.configChanged)
        self.__mainWidget.longTermForgetObservations.textChanged.connect(self.configChanged)
        self.__mainWidget.longTermMode.textChanged.connect(self.configChanged)
        self.__mainWidget.modelUpdateIntervalLongTerm.textChanged.connect(self.configChanged)
        self.__mainWidget.modelUpdateIntervalUltLongTerm.textChanged.connect(self.configChanged)
        self.__mainWidget.modelWarmupDuration.textChanged.connect(self.configChanged)
        self.__mainWidget.calculateRmsResidual.textChanged.connect(self.configChanged)

        # Manipulate config
        self.__mainWidget.saveParameters.setEnabled(False)
        self.__mainWidget.saveParameters.clicked.connect(self.saveParameters)
        self.__mainWidget.resetParameters.clicked.connect(self.resetParameters)
        self.__mainWidget.uploadSaved.clicked.connect(self.uploadSavedParameters)

        # Radio buttons
        self.radioButtons = QButtonGroup()
        self.radioButtons.addButton(self.__mainWidget.rawRadio)
        self.radioButtons.addButton(self.__mainWidget.ellipseRadio)
        self.radioButtons.addButton(self.__mainWidget.debugRadio)
        self.radioButtons.addButton(self.__mainWidget.rayRadio)
        self.__mainWidget.rayRadio.setEnabled(False)
        self.__mainWidget.ellipseRadio.setChecked(True)
        self.radioButtons.buttonClicked.connect(self.radioClicked)

        # Setup scripts
        self.detector_2d.update_properties(self.detector_2d_config)
        self.previewDetector2d.update_properties(self.detector_2d_config)
        self.camera = CameraModel(focal_length=self.config['focal_length'], resolution=[640, 480])
        self.detector_3d = Detector3D(camera=self.camera)
        self.detector_3d.update_properties(self.detector_3d_config)

        # Title bar design
        self.setupTitleBar(self)

        # Main window events
        self.__mainWidget.listImages.itemClicked.connect(self.imageClicked)
        self.__mainWidget.startButton.setEnabled(False)
        self.__mainWidget.startButton.clicked.connect(self.startDetection)
        #self.__mainWidget.calibrate.clicked.connect(self.openCalibrationWindow)
        #self.__mainWidget.calibrate.setEnabled(False)
        self.__mainWidget.reanalyze.clicked.connect(self.reanalyze)
        self.__mainWidget.loadImage.clicked.connect(self.loadImage)
        self.__mainWidget.scanpath.clicked.connect(self.showScanpath)
        self.__mainWidget.heatmap.clicked.connect(self.showHeatmap)
        self.__mainWidget.rawImage.clicked.connect(self.showRawImage)
        self.__mainWidget.imagePath.setText("Choose image")
        self.__mainWidget.imagePath.setText(self.__mainWidget.imagePath.fontMetrics().elidedText(self.__mainWidget.imagePath.text(), Qt.ElideRight, self.__mainWidget.imagePath.width()))

    def openCalibrationWindow(self):
        calibrationImage = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if calibrationImage[0] != "":
            self.calibration = CalibrationWindow(self, calibrationImage[0])
            self.calibration.show()
            self.openedWindows.append(self.calibration)

    def showHeatmap(self):
        visualizationImage = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if visualizationImage[0] != "":
            self.popup = VisualizationWindow(visualizationImage[0], heatmap=True, rawData=self.rawDataFromDetection)
            self.popup.show()
            self.openedWindows.append(self.popup)

    def showScanpath(self):
        visualizationImage = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if visualizationImage[0] != "":
            self.popup = VisualizationWindow(visualizationImage[0], scanpath=True, rawData=self.rawDataFromDetection)
            self.popup.show()
            self.openedWindows.append(self.popup)

    def showRawImage(self):
        visualizationImage = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if visualizationImage[0] != "":
            self.popup = VisualizationWindow(visualizationImage[0])
            self.popup.show()
            self.openedWindows.append(self.popup)

    def radioClicked(self, button):
        self.imageFlag = button.text().split(" ")[0]
        if self.clickedItem:
            self.imageClicked(item=self.clickedItem)
        elif self.lastDetectionImage:
            self.__mainWidget.rayRadio.setEnabled(False)
            self.imageClicked(lastImage=self.lastDetectionImage)

    def configChanged(self):
        pupil_size_min = int(self.__mainWidget.pupilSizeMin.text()) if self.__mainWidget.pupilSizeMin.text() != "" else None
        pupil_size_max = int(self.__mainWidget.pupilSizeMax.text()) if self.__mainWidget.pupilSizeMax.text() != "" else None
        strong_perimeter_min = float(self.__mainWidget.strongPerimeterMin.text()) if self.__mainWidget.strongPerimeterMin.text() != "" else None
        strong_perimeter_max = float(self.__mainWidget.strongPerimeterMax.text()) if self.__mainWidget.strongPerimeterMax.text() != "" else None
        strong_area_min = float(self.__mainWidget.strongAreaMin.text()) if self.__mainWidget.strongAreaMin.text() != "" else None
        strong_area_max = float(self.__mainWidget.strongAreaMax.text()) if self.__mainWidget.strongAreaMax.text() != "" else None
        final_perimeter_min = float(self.__mainWidget.finalPerimeterMin.text()) if self.__mainWidget.finalPerimeterMin.text() != "" else None
        final_perimeter_max = float(self.__mainWidget.finalPerimeterMax.text()) if self.__mainWidget.finalPerimeterMax.text() != "" else None

        if self.__mainWidget.focalLength.text() != ""  \
            and self.__mainWidget.elev.text() != "" \
            and self.__mainWidget.azim.text() != "" \
            and self.__mainWidget.scaleFactor.text() != "" \
            and self.__mainWidget.thresholdSwirski.text() != "" \
            and self.__mainWidget.thresholdKalman.text() != "" \
            and self.__mainWidget.thresholdShortTerm.text() != "" \
            and self.__mainWidget.thresholdLongTerm.text() != "" \
            and self.__mainWidget.longTermBufferSize.text() != "" \
            and self.__mainWidget.longTermForgetTime.text() != "" \
            and self.__mainWidget.longTermForgetObservations.text() != "" \
            and (self.__mainWidget.longTermMode.text() == "blocking" or self.__mainWidget.longTermMode.text() == "asynchronous") \
            and self.__mainWidget.modelUpdateIntervalLongTerm.text() != "" \
            and self.__mainWidget.modelUpdateIntervalUltLongTerm.text() != "" \
            and self.__mainWidget.modelWarmupDuration.text() != "" \
            and (self.__mainWidget.calculateRmsResidual.text() == "True" or self.__mainWidget.calculateRmsResidual.text() == "False") \
            and self.__mainWidget.intensityRange.text() != "" \
            and self.__mainWidget.pupilSizeMax.text() != "" \
            and self.__mainWidget.pupilSizeMin.text() != "" \
            and self.__mainWidget.blurSize.text() != "" \
            and self.__mainWidget.cannyTreshold.text() != "" \
            and self.__mainWidget.cannyRation.text() != "" \
            and self.__mainWidget.cannyAperture.text() != "" \
            and self.__mainWidget.coarseFilterMin.text() != "" \
            and self.__mainWidget.coarseFilterMax.text() != "" \
            and (self.__mainWidget.coarseDetection.text() == "True" or self.__mainWidget.coarseDetection.text() == "False") \
            and self.__mainWidget.contourSizeMin.text() != "" \
            and self.__mainWidget.strongPerimeterMin.text() != "" \
            and self.__mainWidget.strongPerimeterMax.text() != "" \
            and self.__mainWidget.strongAreaMin.text() != "" \
            and self.__mainWidget.strongAreaMax.text() != "" \
            and self.__mainWidget.ellipseRoudnessRatio.text != "" \
            and self.__mainWidget.initialEllipseTreshhold.text() != "" \
            and self.__mainWidget.finalPerimeterMin.text() != "" \
            and self.__mainWidget.finalPerimeterMax.text() != "" \
            and self.__mainWidget.ellipseSupportMinDist.text() != "" \
            and self.__mainWidget.supportPixelRatio.text() != "":
            if pupil_size_min is not None and pupil_size_max is not None and pupil_size_max > pupil_size_min \
                and strong_perimeter_min is not None and strong_perimeter_max is not None and strong_perimeter_max > strong_perimeter_min \
                    and strong_area_min is not None and strong_area_max is not None and strong_area_max > strong_area_min \
                        and final_perimeter_min is not None and final_perimeter_max is not None and final_perimeter_max > final_perimeter_min:
                self.setParameters()
            else:
                self.__mainWidget.saveParameters.setEnabled(False)

        else:
            self.__mainWidget.saveParameters.setEnabled(False)
    
    def setParameters(self):
        # Graph parameters
        self.config['elev'] = int(self.__mainWidget.elev.text())
        self.config['azim'] = int(self.__mainWidget.azim.text())
        self.config['scaleFactor'] = float(self.__mainWidget.scaleFactor.text())

        # Camera parameters
        self.config['focal_length'] = float(self.__mainWidget.focalLength.text())

        # 2D detector parameters
        self.config["detector_2d"]['intensity_range'] = int(self.__mainWidget.intensityRange.text())
        self.config["detector_2d"]['pupil_size_max'] = int(self.__mainWidget.pupilSizeMax.text())
        self.config["detector_2d"]['pupil_size_min'] = int(self.__mainWidget.pupilSizeMin.text())
        self.config["detector_2d"]['blur_size'] = int(self.__mainWidget.blurSize.text())
        self.config["detector_2d"]['canny_threshold'] = int(self.__mainWidget.cannyTreshold.text())
        self.config["detector_2d"]['canny_ration'] = int(self.__mainWidget.cannyRation.text())
        self.config["detector_2d"]['canny_aperture'] = int(self.__mainWidget.cannyAperture.text())
        self.config["detector_2d"]['coarse_filter_min'] = int(self.__mainWidget.coarseFilterMin.text())
        self.config["detector_2d"]['coarse_filter_max'] = int(self.__mainWidget.coarseFilterMax.text())
        self.config["detector_2d"]['coarse_detection'] = int(self.__mainWidget.coarseDetection.text() == "True")
        self.config["detector_2d"]['contour_size_min'] = int(self.__mainWidget.contourSizeMin.text())
        self.config["detector_2d"]['strong_perimeter_ratio_range_min'] = float(self.__mainWidget.strongPerimeterMin.text())
        self.config["detector_2d"]['strong_perimeter_ratio_range_max'] = float(self.__mainWidget.strongPerimeterMax.text())
        self.config["detector_2d"]['strong_area_ratio_range_min'] = float(self.__mainWidget.strongAreaMin.text())
        self.config["detector_2d"]['strong_area_ratio_range_max'] = float(self.__mainWidget.strongAreaMax.text())
        self.config["detector_2d"]['ellipse_roundness_ratio'] = float(self.__mainWidget.ellipseRoudnessRatio.text())
        self.config["detector_2d"]['initial_ellipse_fit_threshhold'] = float(self.__mainWidget.initialEllipseTreshhold.text())
        self.config["detector_2d"]['final_perimeter_ratio_range_min'] = float(self.__mainWidget.finalPerimeterMin.text())
        self.config["detector_2d"]['final_perimeter_ratio_range_max'] = float(self.__mainWidget.finalPerimeterMax.text())
        self.config["detector_2d"]['ellipse_true_support_min_dist'] = float(self.__mainWidget.ellipseSupportMinDist.text())
        self.config["detector_2d"]['support_pixel_ratio_exponent'] = float(self.__mainWidget.supportPixelRatio.text())

        # 3D detector parameters
        self.config["detector_3d"]["threshold_swirski"] = float(self.__mainWidget.thresholdSwirski.text())
        self.config["detector_3d"]["threshold_kalman"] = float(self.__mainWidget.thresholdKalman.text())
        self.config["detector_3d"]["threshold_short_term"] = float(self.__mainWidget.thresholdShortTerm.text())
        self.config["detector_3d"]["threshold_long_term"] = float(self.__mainWidget.thresholdLongTerm.text())
        self.config["detector_3d"]["long_term_buffer_size"] = int(self.__mainWidget.longTermBufferSize.text())
        self.config["detector_3d"]["long_term_forget_time"] = int(self.__mainWidget.longTermForgetTime.text())
        self.config["detector_3d"]["long_term_forget_observations"] = int(self.__mainWidget.longTermForgetObservations.text())
        self.config["detector_3d"]["long_term_mode"] = 1 if self.__mainWidget.longTermMode.text() == "asynchronous" else 0
        self.config["detector_3d"]["model_update_interval_long_term"] = float(self.__mainWidget.modelUpdateIntervalLongTerm.text())
        self.config["detector_3d"]["model_update_interval_ult_long_term"] = float(self.__mainWidget.modelUpdateIntervalUltLongTerm.text())
        self.config["detector_3d"]["model_warmup_duration"] = float(self.__mainWidget.modelWarmupDuration.text())
        self.config["detector_3d"]["calculate_rms_residual"] = int(self.__mainWidget.calculateRmsResidual.text() == "True")
      
        self.detector_2d_config = self.config["detector_2d"].copy()
        self.detector_2d_config["coarse_detection"] = bool(self.detector_2d_config["coarse_detection"])
        self.detector_3d_config = self.config["detector_3d"].copy()
        self.detector_3d_config["long_term_mode"] = DetectorMode.blocking if int(self.detector_3d_config["long_term_mode"]) == 0 else DetectorMode.asynchronous
        self.detector_3d_config["calculate_rms_residual"] = bool(self.detector_3d_config["calculate_rms_residual"])
        self.previewDetector2d = Detector2D(self.detector_2d_config)
        if self.clickedItem:
            self.imageClicked(item=self.clickedItem)
        elif self.lastDetectionImage:
            self.imageClicked(lastImage=self.lastDetectionImage)
        self.__mainWidget.saveParameters.setEnabled(True)

    def resetParameters(self):
        # Camera parameters
        self.__mainWidget.focalLength.setText(str(self.default_config['focal_length']))

        # Graph parameters
        self.__mainWidget.elev.setText(str(self.default_config['elev']))
        self.__mainWidget.azim.setText(str(self.default_config['azim']))
        self.__mainWidget.scaleFactor.setText(str(self.default_config['scaleFactor']))

        # 2D detector parameters
        self.__mainWidget.intensityRange.setText(str(self.default_config["detector_2d"]['intensity_range']))
        self.__mainWidget.pupilSizeMax.setText(str(self.default_config["detector_2d"]['pupil_size_max']))
        self.__mainWidget.pupilSizeMin.setText(str(self.default_config["detector_2d"]['pupil_size_min']))
        self.__mainWidget.blurSize.setText(str(self.default_config["detector_2d"]['blur_size']))
        self.__mainWidget.cannyTreshold.setText(str(self.default_config["detector_2d"]['canny_threshold']))
        self.__mainWidget.cannyRation.setText(str(self.default_config["detector_2d"]['canny_ration']))
        self.__mainWidget.cannyAperture.setText(str(self.default_config["detector_2d"]['canny_aperture']))
        self.__mainWidget.coarseFilterMin.setText(str(self.default_config["detector_2d"]['coarse_filter_min']))
        self.__mainWidget.coarseFilterMax.setText(str(self.default_config["detector_2d"]['coarse_filter_max']))
        self.__mainWidget.coarseDetection.setText(str(bool(self.default_config["detector_2d"]['coarse_detection'])))
        self.__mainWidget.contourSizeMin.setText(str(self.default_config["detector_2d"]['contour_size_min']))
        self.__mainWidget.strongPerimeterMin.setText(str(self.default_config["detector_2d"]['strong_perimeter_ratio_range_min']))
        self.__mainWidget.strongPerimeterMax.setText(str(self.default_config["detector_2d"]['strong_perimeter_ratio_range_max']))
        self.__mainWidget.strongAreaMin.setText(str(self.default_config["detector_2d"]['strong_area_ratio_range_min']))
        self.__mainWidget.strongAreaMax.setText(str(self.default_config["detector_2d"]['strong_area_ratio_range_max']))
        self.__mainWidget.ellipseRoudnessRatio.setText(str(self.default_config["detector_2d"]['ellipse_roundness_ratio']))
        self.__mainWidget.initialEllipseTreshhold.setText(str(self.default_config["detector_2d"]['initial_ellipse_fit_threshhold']))
        self.__mainWidget.finalPerimeterMin.setText(str(self.default_config["detector_2d"]['final_perimeter_ratio_range_min']))
        self.__mainWidget.finalPerimeterMax.setText(str(self.default_config["detector_2d"]['final_perimeter_ratio_range_max']))
        self.__mainWidget.ellipseSupportMinDist.setText(str(self.default_config["detector_2d"]['ellipse_true_support_min_dist']))
        self.__mainWidget.supportPixelRatio.setText(str(self.default_config["detector_2d"]['support_pixel_ratio_exponent']))

        # 3D detector parameters
        self.__mainWidget.thresholdSwirski.setText(str(self.default_config["detector_3d"]['threshold_swirski']))
        self.__mainWidget.thresholdKalman.setText(str(self.default_config["detector_3d"]['threshold_kalman']))
        self.__mainWidget.thresholdShortTerm.setText(str(self.default_config["detector_3d"]['threshold_short_term']))
        self.__mainWidget.thresholdLongTerm.setText(str(self.default_config["detector_3d"]['threshold_long_term']))
        self.__mainWidget.longTermBufferSize.setText(str(self.default_config["detector_3d"]['long_term_buffer_size']))
        self.__mainWidget.longTermForgetTime.setText(str(self.default_config["detector_3d"]['long_term_forget_time']))
        self.__mainWidget.longTermForgetObservations.setText(str(self.default_config["detector_3d"]['long_term_forget_observations']))
        self.__mainWidget.longTermMode.setText("asynchronous" if str(self.default_config["detector_3d"]['long_term_mode']) == "1" else "blocking")
        self.__mainWidget.modelUpdateIntervalLongTerm.setText(str(self.default_config["detector_3d"]['model_update_interval_long_term']))
        self.__mainWidget.modelUpdateIntervalUltLongTerm.setText(str(self.default_config["detector_3d"]['model_update_interval_ult_long_term']))
        self.__mainWidget.modelWarmupDuration.setText(str(self.default_config["detector_3d"]['model_warmup_duration']))
        self.__mainWidget.calculateRmsResidual.setText(str(bool(self.default_config["detector_3d"]['calculate_rms_residual'])))

    def saveParameters(self):
        self.config["detector_3d"]["long_term_mode"] = self.config["detector_3d"]["long_term_mode"]
        self.config["detector_3d"]["calculate_rms_residual"] = 1 if self.config["detector_3d"]["calculate_rms_residual"] else 0
        self.config["detector_2d"]["coarse_detection"] = 1 if self.config["detector_2d"]["coarse_detection"] else 0
        with open('config/config.json', 'w') as outfile:
            json.dump(self.config, outfile)
        self.__mainWidget.saveParameters.setEnabled(False)

    def uploadSavedParameters(self):
        with open('config/config.json') as json_file:
            self.original_config = json.load(json_file)

        # Camera parameters
        self.__mainWidget.focalLength.setText(str(self.original_config['focal_length']))

        # Graph parameters
        self.__mainWidget.elev.setText(str(self.original_config['elev']))
        self.__mainWidget.azim.setText(str(self.original_config['azim']))
        self.__mainWidget.scaleFactor.setText(str(self.original_config['scaleFactor']))

        # 2D detector parameters
        self.__mainWidget.coarseDetection.setText(str(bool(self.original_config["detector_2d"]['coarse_detection'])))
        self.__mainWidget.coarseFilterMin.setText(str(self.original_config["detector_2d"]['coarse_filter_min']))
        self.__mainWidget.coarseFilterMax.setText(str(self.original_config["detector_2d"]['coarse_filter_max'])) 
        self.__mainWidget.intensityRange.setText(str(self.original_config["detector_2d"]['intensity_range']))
        self.__mainWidget.blurSize.setText(str(self.original_config["detector_2d"]['blur_size']))
        self.__mainWidget.cannyTreshold.setText(str(self.original_config["detector_2d"]['canny_threshold']))
        self.__mainWidget.cannyRation.setText(str(self.original_config["detector_2d"]['canny_ration']))
        self.__mainWidget.cannyAperture.setText(str(self.original_config["detector_2d"]['canny_aperture']))
        self.__mainWidget.pupilSizeMax.setText(str(self.original_config["detector_2d"]['pupil_size_max']))
        self.__mainWidget.pupilSizeMin.setText(str(self.original_config["detector_2d"]['pupil_size_min']))
        self.__mainWidget.strongPerimeterMin.setText(str(self.original_config["detector_2d"]['strong_perimeter_ratio_range_min']))
        self.__mainWidget.strongPerimeterMax.setText(str(self.original_config["detector_2d"]['strong_perimeter_ratio_range_max']))
        self.__mainWidget.strongAreaMin.setText(str(self.original_config["detector_2d"]['strong_area_ratio_range_min']))
        self.__mainWidget.strongAreaMax.setText(str(self.original_config["detector_2d"]['strong_area_ratio_range_max']))
        self.__mainWidget.contourSizeMin.setText(str(self.original_config["detector_2d"]['contour_size_min']))
        self.__mainWidget.ellipseRoudnessRatio.setText(str(self.original_config["detector_2d"]['ellipse_roundness_ratio']))
        self.__mainWidget.initialEllipseTreshhold.setText(str(self.original_config["detector_2d"]['initial_ellipse_fit_threshhold']))
        self.__mainWidget.finalPerimeterMin.setText(str(self.original_config["detector_2d"]['final_perimeter_ratio_range_min'])) 
        self.__mainWidget.finalPerimeterMax.setText(str(self.original_config["detector_2d"]['final_perimeter_ratio_range_max']))
        self.__mainWidget.ellipseSupportMinDist.setText(str(self.original_config["detector_2d"]['ellipse_true_support_min_dist']))
        self.__mainWidget.supportPixelRatio.setText(str(self.original_config["detector_2d"]['support_pixel_ratio_exponent']))

        # 3D detector parameters
        self.__mainWidget.thresholdSwirski.setText(str(self.original_config["detector_3d"]['threshold_swirski']))
        self.__mainWidget.thresholdKalman.setText(str(self.original_config["detector_3d"]['threshold_kalman']))
        self.__mainWidget.thresholdShortTerm.setText(str(self.original_config["detector_3d"]['threshold_short_term']))
        self.__mainWidget.thresholdLongTerm.setText(str(self.original_config["detector_3d"]['threshold_long_term']))
        self.__mainWidget.longTermBufferSize.setText(str(self.original_config["detector_3d"]['long_term_buffer_size']))
        self.__mainWidget.longTermForgetTime.setText(str(self.original_config["detector_3d"]['long_term_forget_time']))
        self.__mainWidget.longTermForgetObservations.setText(str(self.original_config["detector_3d"]['long_term_forget_observations']))
        self.__mainWidget.longTermMode.setText("asynchronous" if str(self.original_config["detector_3d"]['long_term_mode']) == "1" else "blocking")
        self.__mainWidget.modelUpdateIntervalLongTerm.setText(str(self.original_config["detector_3d"]['model_update_interval_long_term']))
        self.__mainWidget.modelUpdateIntervalUltLongTerm.setText(str(self.original_config["detector_3d"]['model_update_interval_ult_long_term']))
        self.__mainWidget.modelWarmupDuration.setText(str(self.original_config["detector_3d"]['model_warmup_duration']))
        self.__mainWidget.calculateRmsResidual.setText(str(bool(self.original_config["detector_3d"]['calculate_rms_residual'])))

    def resetDetectors(self):
        self.detector_2d = Detector2D(self.detector_2d_config)
        self.previewDetector2d = Detector2D(self.detector_2d_config)
        self.camera = CameraModel(focal_length=self.config['focal_length'], resolution=[640, 480])
        self.detector_3d = Detector3D(camera=self.camera)
        self.detector_3d.update_properties(self.detector_3d_config)

    def reanalyze(self):
        self.detector_2d = Detector2D(self.detector_2d_config)
        self.previewDetector2d = Detector2D(self.detector_2d_config)
        self.camera = CameraModel(focal_length=self.config['focal_length'], resolution=[640, 480])
        self.detector_3d = Detector3D(camera=self.camera)
        self.detector_3d.update_properties(self.detector_3d_config)
        self.detectionRound = 0
        self.lastDetectionImage = None
        self.clickedItem = None
        self.rawDataFromDetection = {}
        self.pointsOnDisplay = []
        self.startDetection()

    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if fname[0] != "":
            self.imageName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
            self.imagePath = fname[0]
            self.rawDataFromDetection = {}
            self.pointsOnDisplay = []
            self.fillImageList = 0
            self.detectionRound = 0
            self.imagesPaths = {}
            self.lastDetectionImage = None
            self.clickedItem = None
            self.resetDetectors()
            self.folderPath = os.path.dirname(fname[0])
            file_list = glob.glob(os.path.join(self.folderPath, "*"))
            self.imageAmount = len(file_list)
            self.__mainWidget.startButton.setEnabled(True)
            self.__mainWidget.rayRadio.setEnabled(False)
            self.__mainWidget.imageLabel.clear()
            self.__mainWidget.listImages.clear()
            self.__mainWidget.imagePath.setText(self.imageName)
            self.__mainWidget.imagePath.setText(self.__mainWidget.imagePath.fontMetrics().elidedText(self.__mainWidget.imagePath.text(), Qt.ElideRight, self.__mainWidget.imagePath.width()))
        else:
            self.imagePath = None
            self.folderPath = None
            self.imageAmount = 0
            self.fillImageList = 0
            self.lastDetectionImage = None
            self.clickedItem = None
            self.detectionRound = 0
            self.imagesPaths = {}
            self.resetDetectors()
            self.rawDataFromDetection = {}
            self.pointsOnDisplay = []
            self.__mainWidget.imagePath.setText("Choose image")
            self.__mainWidget.startButton.setEnabled(False)
            self.__mainWidget.rayRadio.setEnabled(False)
            self.__mainWidget.listImages.clear()
            self.__mainWidget.imageLabel.clear()

    def startDetection(self):
        if self.imagePath:
            self.clickedItem = None
            self.__mainWidget.rayRadio.setEnabled(False)
            #self.__mainWidget.calibrate.setEnabled(False)
            self.isRunning = True
            if self.__mainWidget.rayRadio.isChecked():
                self.__mainWidget.rawRadio.setChecked(True)
                self.imageFlag = "Raw"
            if self.detectionRound == 0:
                self.renderImage()
                if self.fillImageList == 0:
                    self.fillImageList = 1
                self.detectionRound = 1
                self.renderImage()
            else:
                self.rawDataFromDetection = {}
                self.pointsOnDisplay = []
                self.renderImage()
            #self.__mainWidget.calibrate.setEnabled(True)
            self.isRunning = False

    def renderImage(self):
        fileName = self.imageName.split("_")
        fileNumber = int(fileName[1].split(".")[0])
        fileFormat = self.imageName.split(".")[-1]
        for i in range(fileNumber, self.imageAmount):
            newPath = self.folderPath + "/" + fileName[0] + "_" + str(i) + "." + fileFormat
            if self.fillImageList == 0:
                listImageName = fileName[0] + "_" + str(i) + "." + fileFormat
                self.__mainWidget.listImages.addItem(listImageName)
                self.imagesPaths[listImageName] = newPath

            self.lastDetectionImage = newPath
            image = cv2.imread(newPath)

            grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if self.imageFlag == "2D":
                result_2d = self.previewDetector2d.detect(grayscale_array, image)
        
            elif self.imageFlag == "Simple": 
                result_2d = self.detector_2d.detect(grayscale_array)  
                cv2.ellipse(
                    image,
                    tuple(int(v) for v in result_2d["ellipse"]["center"]),
                    tuple(int(v / 2) for v in result_2d["ellipse"]["axes"]),
                    result_2d["ellipse"]["angle"],
                    0,
                    360,
                    (0, 255, 0), 
                )
            else: 
                result_2d = self.detector_2d.detect(grayscale_array) 
            result_2d["timestamp"] = i
            result_3d = self.detector_3d.update_and_detect(result_2d, grayscale_array, apply_refraction_correction=False)

            if self.detectionRound == 1:
                self.rawDataFromDetection[i] = result_3d
                print(result_3d)
                eyePosWorld = self.transform(np.array(result_3d["sphere"]["center"]), self.cameraPos, self.cameraRotMat)
                gazeRay = self.normalize(self.rotate(result_3d["circle_3d"]["normal"], self.cameraRotMat))

                print("eyePosWorld: ", eyePosWorld)
                print("gazeRay: ", gazeRay)
                intersectionTime = self.intersectPlane(self.displayNormalWorld, self.displayPos, eyePosWorld, gazeRay)

                planeIntersection = np.array([0, 0, 0])
                if (intersectionTime > 0.0):
                    planeIntersection = self.getPoint([eyePosWorld, gazeRay], intersectionTime)

                print("planeIntersection: ", planeIntersection)                
                self.pointsOnDisplay.append(planeIntersection)

            self.displayImage(image)
            cv2.waitKey(10)

    def imageClicked(self, item = None, lastImage = None):
        self.clickedItem = item
        if self.clickedItem and not self.isRunning:
            self.__mainWidget.rayRadio.setEnabled(True)
        path = self.lastDetectionImage if lastImage else self.imagesPaths[item.text()]
        image = cv2.imread(path)

        grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.imageFlag == "2D":
            result_2d = self.previewDetector2d.detect(grayscale_array, image)
        
        elif self.imageFlag == "Simple": 
            result_2d = self.previewDetector2d.detect(grayscale_array)                 
            cv2.ellipse(
                image,
                tuple(int(v) for v in result_2d["ellipse"]["center"]),
                tuple(int(v / 2) for v in result_2d["ellipse"]["axes"]),
                result_2d["ellipse"]["angle"],
                0,
                360,
                (0, 255, 0), 
            )

        # TODO: zistiť či nebeži cyklus .. pretože data sa premažu ale v liste obrazky ostanu
        # TODO: nastane index error ked chceli 3D debug
        elif self.imageFlag == "3D" and self.clickedItem and self.rawDataFromDetection:
            index = self.__mainWidget.listImages.row(self.clickedItem)
            data = self.rawDataFromDetection[index]
            eyePosWorld = self.transform(np.array(data["sphere"]["center"]), self.cameraPos, self.cameraRotMat)
            gazeRay = self.normalize(self.rotate(data["circle_3d"]["normal"], self.cameraRotMat))

            planeIntersection = self.pointsOnDisplay[index]
            image = cv2.cvtColor(self.visualizeRaycast(self.pointsOnDisplay, planeIntersection, 
                                                       self.cameraPos, eyePosWorld, self.cameraDirsWorld, gazeRay, 
                                                       screenWidth=self.displaySize[0], screenHeight=self.displaySize[1], 
                                                       rayNumber=0), cv2.COLOR_BGR2RGB) # rayNumber = index + 1
            
            image = image[80:560, 80:720]
            image = np.ascontiguousarray(image)

        self.displayImage(image)

    def displayImage(self, img):
        qformat = QImage.Format_Indexed8

        if len(img.shape) == 3:
            if img.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        outImage = QImage(img, img.shape[1], img.shape[0], img.strides[0], qformat)
        outImage = outImage.rgbSwapped()
        self.__mainWidget.imageLabel.setPixmap(QPixmap.fromImage(outImage))
        self.__mainWidget.imageLabel.setScaledContents(True)

    def visualizeRaycast(self, allRays, raycastEnd, cameraPos, cameraTarget, cameraDirs, gazeDir, screenWidth = 250, screenHeight = 250, rayNumber = 1):
        fig = pyplot.figure()
        ax = fig.add_subplot(111, projection='3d')
        scale = 1.0 / max(self.config['scaleFactor'], 0.1)

        # Set limit for each axis
        ax.set_xlim(250 * scale, -250 * scale)
        ax.set_ylim(0, -500 * scale)
        ax.set_zlim(-250 * scale, 250 * scale)

        # Set label for each axis
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Display
        x1 = screenWidth / 2
        y1 = -500
        z1 = screenHeight / 2
        x2 = - screenWidth / 2
        y2 = -500
        z2 = - screenHeight / 2
        verts = [(x1, y1, z1), (x2, y1, z1), (x2, y2, z2), (x1, y2, z2)]
        ax.add_collection3d(Poly3DCollection([verts], facecolors='gray', linewidths=1, edgecolors='r', alpha=.25))

        # Display label
        x, y, z = screenWidth / 2 + 10, -500, screenHeight / 2 + 10
        text = Text3D(x, y, z, 'Display', zdir='x')
        ax.add_artist(text)

        # Eye
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x3 = 10 * np.cos(u)*np.sin(v)
        y3 = 10 * np.sin(u)*np.sin(v)
        z3 = 10 * np.cos(v)
        ax.plot_wireframe(x3, y3, z3, color="gray", facecolors='gray')

        # Eye label
        x, y, z = 7, 0, 7
        text = Text3D(x, y, z, 'Eye', zdir='x')
        ax.add_artist(text)

        # Camera
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x3 = 10 * np.cos(u)*np.sin(v) + cameraPos[0] # switch x axis because of the global axis
        y3 = 10 * np.sin(u)*np.sin(v) + cameraPos[1] # switch y axis because of the global axis
        z3 = 10 * np.cos(v) + cameraPos[2]
        ax.plot_wireframe(x3, y3, z3, color="green", facecolors='green')

        # Camera label
        x, y, z = cameraPos[0] + 7, cameraPos[1], cameraPos[2] + 7
        text = Text3D(x, y, z, 'Camera', zdir='x')
        ax.add_artist(text)

        # Camera target
        x_start, y_start, z_start = cameraPos[0], cameraPos[1], cameraPos[2]
        x_end, y_end, z_end = cameraTarget[0], cameraTarget[1], cameraTarget[2]
        ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='green', linewidth=1)

        # Camera axes
        ax.quiver(*cameraPos, *cameraDirs[0], length=25, normalize=True, color='red')
        ax.quiver(*cameraPos, *cameraDirs[1], length=25, normalize=True, color='green')
        ax.quiver(*cameraPos, *cameraDirs[2], length=25, normalize=True, color='blue')

        # Eye axes
        origin = (0, 0, 0)
        dirs = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

        # Camera axes
        ax.quiver(*origin, *dirs[0], length=25, normalize=True, color='red')
        ax.quiver(*origin, *dirs[1], length=25, normalize=True, color='green')
        ax.quiver(*origin, *dirs[2], length=25, normalize=True, color='blue')

        # Gaze direction
        x_start, y_start, z_start = origin
        x_end, y_end, z_end = raycastEnd[0], raycastEnd[1], raycastEnd[2]
        ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='red', linewidth=1)

        for i in range(rayNumber):
            ax.scatter(allRays[i][0], allRays[i][1], allRays[i][2], color='blue')


        ax.view_init(elev=self.config['elev'], azim=self.config['azim'])
        fig.set_size_inches(8, 6)
        fig.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pyplot.close(fig)
        return data

    def closeEvent(self, event):
        for i in self.openedWindows:
            i.close()
        event.accept()

class VisualizationWindow(FramelessMainWindow, GlobalSharedClass):
    def __init__(self, imagePath = None, rawData = None, heatmap = None, scanpath = None):
        GlobalSharedClass.__init__(self)
        super().__init__()

        with open('./coordinates/random_uv_coords.csv') as f:
            reader = csv.reader(f)
            self.raw = list(map(lambda q: (float(q[0]), float(q[1])), reader))

        self.raw = {i: self.raw[i] for i in range(0, len(self.raw))}

        self.paddedImage = None
        self.imageWidth = None
        self.imageHeight = None
        self.paddingMax = None
        self.qimg = None
        self.color1 = (0, 0, 0)
        self.color2 = (255, 255, 255) 
        self.threshold = 1

        self.__mainWidget = QWidget()
        ui = QFile("ui/popupWindow.ui")
        ui.open(QFile.ReadOnly)
        self.__mainWidget = self.loader.load(ui)
        ui.close()
        self.setWindowTitle('Image preview')
        self.setWindowIcon('public/light.png')
        self.mainWidget = self.centralWidget()
        lay = self.mainWidget.layout()
        lay.addWidget(self.__mainWidget)
        self.mainWidget.setLayout(lay)
        self.setCentralWidget(self.mainWidget)
        self.setGeometry(200, 50, 1024, 636)
        self.setFixedSize(1024, 636)

        # Title bar design
        self.setupTitleBar(self)

        self.uv_coords = []	
        self.outliers = []
        self.outliersToDraw = []
        self.dir_vectors = {}
        self.points_group = {}
        self.repeat = False
        self.thresholdChanged = False
        self.points_group_keys = []
        self.imagePath = imagePath
        self.rawData = rawData
        self.heatmap = heatmap
        self.scanpath = scanpath
        self.__mainWidget.saveImage.clicked.connect(self.saveImage)
        self.__mainWidget.color1.clicked.connect(self.setFirstColor)
        self.__mainWidget.color2.clicked.connect(self.setSecondColor)

        self.__mainWidget.thresholdInput.setValidator(QRegularExpressionValidator(self.thresholdRegex))
        self.__mainWidget.thresholdInput.setText(str(self.threshold))
        self.__mainWidget.thresholdInput.textChanged.connect(self.thresholdChange)
        self.__mainWidget.thresholdButton.clicked.connect(self.changeThreshold)
        self.__mainWidget.color1.setStyleSheet(f'QPushButton {{background-color: #000000; border: 5px solid #FFE81F;}}')
        self.__mainWidget.color2.setStyleSheet(f'QPushButton {{background-color: #FFFFFF; border: 5px solid #FFE81F;}}')
        if not self.scanpath:
            self.__mainWidget.color1.hide()
            self.__mainWidget.color2.hide()
            self.__mainWidget.thresholdInput.hide()
            self.__mainWidget.thresholdButton.hide()

        if self.rawData:
            self.rawToPoint()

        if self.imagePath:
            self.displayImage()

    def changeThreshold(self):
        self.displayImage()

    def thresholdChange(self):
        if self.__mainWidget.thresholdInput.text() != "":
            self.thresholdChanged = False
            self.threshold = int(self.__mainWidget.thresholdInput.text())

    def setFirstColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.__mainWidget.color1.setStyleSheet(f'QPushButton {{background-color: {color.name()}; border: 5px solid #FFE81F;}}')
            self.color1 = (b, g, r)
            self.displayImage()

    def setSecondColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.__mainWidget.color2.setStyleSheet(f'QPushButton {{background-color: {color.name()}; border: 5px solid #FFE81F;}}')
            self.color2 = (b, g, r)
            self.displayImage()

    def saveImage(self):
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG (*.png);;JPEG (*.jpg *.jpeg);;All Files (*)")
        if fileName:
            self.qimg.save(fileName)

    def rawToPoint(self):
        for i in self.rawData:
            eyePosWorld = self.transform(np.array(self.rawData[i]["sphere"]["center"]), self.cameraPos, self.cameraRotMat)
            gazeRay = self.normalize(self.rotate(self.rawData[i]["circle_3d"]["normal"], self.cameraRotMat))

            intersectionTime = self.intersectPlane(self.displayNormalWorld, self.displayPos, eyePosWorld, gazeRay)

            if (intersectionTime > 0.0):
                planeIntersection = self.getPoint([eyePosWorld, gazeRay], intersectionTime)
                planeIntersection = self.transform(planeIntersection, self.displayPos, self.displayRotMat)
                result = self.convert_to_uv(planeIntersection, includeOutliers=True)
                if result[0] > 1 or result[0] < 0 or result[1] > 1 or result[1] < 0:
                    self.outliers.append(result)
                else:
                    self.uv_coords.append(result)


    def displayImage(self):
        image = None
        if self.scanpath and self.rawData:
            image = self.scanpathVisualization()
            if len(self.outliers):
                image = self.outliersVisualization(image)
            self.repeat = True
        elif self.heatmap and self.rawData:
            image = self.heatmapVisualization()
            if len(self.outliers):
                image = self.outliersVisualization(image)
            self.repeat = True
        else:
            image = cv2.imread(self.imagePath)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        self.qimg = QImage(image.data, w, h, c * w, QImage.Format_RGB888)

        if image.shape[1] > self.__mainWidget.image.width() or image.shape[0] > self.__mainWidget.image.height():
            self.__mainWidget.image.setPixmap(QPixmap.fromImage(self.qimg).scaled(self.__mainWidget.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            self.__mainWidget.image.setPixmap(QPixmap.fromImage(self.qimg))
            self.__mainWidget.image.setAlignment(Qt.AlignCenter)

    def outliersVisualization(self, image):
        if not self.repeat:
            img = cv2.imread(self.imagePath)
            self.imageHeight = img.shape[0]
            self.imageWidth = img.shape[1]
            self.paddingMax = min(self.imageHeight, self.imageWidth) // 10
            
            for i in range(0, len(self.outliers)):
                self.outliers[i] = self.convert_uv_to_px(self.outliers[i], self.imageWidth, self.imageHeight)     
                if self.outliers[i][0] + self.paddingMax < 0:
                    continue
                if self.outliers[i][1] + self.paddingMax < 0:
                    continue
                if self.outliers[i][0] > self.imageWidth + self.paddingMax:
                    continue
                if self.outliers[i][1] > self.imageHeight + self.paddingMax:
                    continue
                self.outliersToDraw.append(self.outliers[i])

            self.paddedImage = np.zeros((self.imageHeight + 2 * self.paddingMax,
                                    self.imageWidth + 2 * self.paddingMax, 3), np.uint8)
        

        if len(self.outliersToDraw):
            self.paddedImage[self.paddingMax:self.paddingMax + self.imageHeight,
                            self.paddingMax:self.paddingMax + self.imageWidth] = image
            
            for i in range(0, len(self.outliersToDraw)):
                cv2.circle(self.paddedImage, (self.outliers[i][0] + self.paddingMax, self.outliers[i][1] + self.paddingMax),
                        5, (0, 0, 255), -1)

            return self.paddedImage
        
        return image
            

    def scanpathVisualization(self):
        if not len(self.uv_coords):
            return cv2.imread(self.imagePath)

        image = cv2.imread(self.imagePath)
        image_width = image.shape[1]
        image_height = image.shape[0]
        circle_radius = min(image_width, image_height) // 100

        overlay_circles = image.copy()
        overlay_lines = image.copy()
        alpha_circles = 0.6
        outline_width = 3
        alpha_lines = 0.2
        TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        TEXT_SCALE = 0.8
        TEXT_THICKNESS = 2
        TEXT_COLOR = (0, 0, 0)

        colors = {}
        order = 0
    
        if not self.repeat:
            for i in range(0, len(self.uv_coords)):
                self.uv_coords[i] = self.convert_uv_to_px(self.uv_coords[i], image_width, image_height)

        if not self.thresholdChanged:
            self.thresholdChanged = True
            self.points_group = {}
            main_point = None

            for i in range(0, len(self.uv_coords)):
                if not main_point:
                    main_point = (self.uv_coords[i][0], self.uv_coords[i][1])

                if abs(self.uv_coords[i][0] - main_point[0]) <= self.threshold and abs(self.uv_coords[i][1] - main_point[1]) <= self.threshold:
                    if not self.points_group.get(order):
                        self.points_group[order] = {'points': [self.uv_coords[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0, 'index': order + 1}
                    else:
                        self.points_group[order]['points'].append(self.uv_coords[i])
                
                else:
                    order += 1
                    main_point = (self.uv_coords[i][0], self.uv_coords[i][1])
                    self.points_group[order] = {'points': [self.uv_coords[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0, 'index': order + 1}

            # self.points_group = dict(sorted(self.points_group.items(), key=lambda item: len(item[1]['points']), reverse=False))
            self.points_group_keys = list(self.points_group)    

            for key in self.points_group:
                points = self.points_group[key]['points']
                x = 0
                y = 0

                for point in points:
                    x += point[0]
                    y += point[1]

                x = int(x / len(points))
                y = int(y / len(points))
                self.points_group[key]['middle']['x'] = x
                self.points_group[key]['middle']['y'] = y            

            different_lengths = {}
            for key in self.points_group:
                if not different_lengths.get(len(self.points_group[key]['points'])):
                    different_lengths[len(self.points_group[key]['points'])] = [key]
                else:
                    different_lengths[len(self.points_group[key]['points'])].append(key)

            # base pixel for diameter ---- diameter = 20
            # normalize between new_min and new_max ---- normalized_value = ((original_value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min
            new_min = 1
            new_max = 4
            # normalize between 1 and 2 ---- normalized_value = (value - min_length) / (max_length - min_length) + 1

            if len(different_lengths) > 1:
                max_length = max(different_lengths)
                min_length = min(different_lengths)
                for value in different_lengths:
                    normalized_value = ((value - min_length) / (max_length - min_length)) * (new_max - new_min) + new_min
                    for key in different_lengths[value]:
                        self.points_group[key]['diameter'] = int(circle_radius * normalized_value)
            else:
                for key in different_lengths:
                    for value in different_lengths[key]:
                        self.points_group[value]['diameter'] = circle_radius

        if len(self.points_group) > 1:
            t = 0
            for key in self.points_group:
                r = min(255, max(0, int(self.lerp(self.color1[0], self.color2[0], t))))
                g = min(255, max(0, int(self.lerp(self.color1[1], self.color2[1], t))))
                b = min(255, max(0, int(self.lerp(self.color1[2], self.color2[2], t))))
                colors[key] = (r, g, b)
                t += 1 / (len(self.points_group) - 1)
        else:
            colors[0] = (self.color1[0], self.color1[1], self.color1[2])

        for key in range(0, len(self.points_group)):
            x1 = self.points_group[self.points_group_keys[key]]['middle']['x']
            y1 = self.points_group[self.points_group_keys[key]]['middle']['y']
            radius1 = self.points_group[self.points_group_keys[key]]['diameter']
            if key < len(self.points_group) - 1:
                x2 = self.points_group[self.points_group_keys[key + 1]]['middle']['x']
                y2 = self.points_group[self.points_group_keys[key + 1]]['middle']['y']
                radius2 = self.points_group[self.points_group_keys[key + 1]]['diameter']

                distance = sqrt((x2 - x1)**2 + (y2 - y1)**2)

                if distance >= (radius1 + radius2):   
                    angle = atan2(y2 - y1, x2 - x1)
                    point1_x = x1 + (radius1 + outline_width // 2) * cos(angle)
                    point1_y = y1 + (radius1 + outline_width // 2) * sin(angle)
                    point2_x = x2 - (radius2 + outline_width // 2) * cos(angle)
                    point2_y = y2 - (radius2 + outline_width // 2) * sin(angle)
                    cv2.line(overlay_lines, (int(point1_x), int(point1_y)), (int(point2_x), int(point2_y)), colors[list(self.points_group)[key]], 4)
            
            text_size, _ = cv2.getTextSize(str(self.points_group[self.points_group_keys[key]]['index']), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
            text_origin = (int(x1 - text_size[0] / 2), int(y1 + text_size[1] / 2))
            
            #cv2.circle(overlay_circles, (x1, y1), radius1, colors[points_group_keys[key]], -1)
            cv2.circle(overlay_circles, (x1, y1), radius1, colors[self.points_group_keys[key]], outline_width)
            #cv2.putText(overlay_circles, str(points_group[points_group_keys[key]]['index']), text_origin, TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


        result = cv2.addWeighted(overlay_circles, alpha_circles, image, 1 - alpha_circles, 0)
        result = cv2.addWeighted(overlay_lines, alpha_lines, result, 1 - alpha_lines, 0)
        return result

    
    def heatmapVisualization(self):
        if not len(self.uv_coords):
            return cv2.imread(self.imagePath)

        img = cv2.imread(self.imagePath)

        dpi = 100.0
        alpha = 0.5
        ngaussian = 200
        sd = 8
        width = img.shape[1]
        height = img.shape[0]
        
        for i in range(0, len(self.uv_coords)):
            self.uv_coords[i] = self.convert_uv_to_px(self.uv_coords[i], width, height)

        figsize = (width / dpi, height / dpi)
        fig = pyplot.figure(figsize=figsize, dpi=dpi, frameon=False)

        ax = pyplot.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.axis([0, width, 0, height])
        ax.imshow(img)

        # HEATMAP
        # Gaussian
        gwh = ngaussian
        gsdwh = sd
        
        xo = gwh / 2
        yo = gwh / 2

        gaus = np.zeros([gwh, gwh], dtype=float)

        for i in range(gwh):
            for j in range(gwh):
                gaus[j, i] = np.exp(
                    -1.0 * (((float(i) - xo) ** 2 / (2 * gsdwh * gsdwh)) + ((float(j) - yo) ** 2 / (2 * gsdwh * gsdwh))))

        # matrix of zeroes
        strt = gwh // 2
        heatmapsize = height + 2 * strt, width + 2 * strt
        heatmap = np.zeros(heatmapsize, dtype=float)
        # create heatmap
        for i in range(0, len(self.uv_coords)):
            # get x and y coordinates
            x = strt + self.uv_coords[i][0] - int(gwh / 2)
            y = strt + self.uv_coords[i][1] - int(gwh / 2)
            # correct Gaussian size if either coordinate falls outside of
            # display boundaries
            if (not 0 < x < width) or (not 0 < y < height):
                hadj = [0, gwh];
                vadj = [0, gwh]
                if 0 > x:
                    hadj[0] = abs(x)
                    x = 0
                elif width < x:
                    hadj[1] = gwh - int(x - width)
                if 0 > y:
                    vadj[0] = abs(y)
                    y = 0
                elif height < y:
                    vadj[1] = gwh - int(y - height)
                # add adjusted Gaussian to the current heatmap
                try:
                    heatmap[y:y + vadj[1], x:x + hadj[1]] += gaus[vadj[0]:vadj[1], hadj[0]:hadj[1]] * self.uv_coords[i][2]
                except:
                    # fixation was probably outside of display
                    pass
            else:
                # add Gaussian to the current heatmap
                heatmap[y:y + gwh, x:x + gwh] += gaus
        # resize heatmap
        heatmap = heatmap[strt:height + strt, strt:width + strt]
        # remove zeros
        lowbound = np.mean(heatmap[heatmap > 0])
        heatmap[heatmap < lowbound] = np.NaN
        # draw heatmap on top of image
        ax.imshow(heatmap, cmap='jet', alpha=alpha)

        # FINISH PLOT
        # invert the y axis, as (0,0) is top left on a display
        ax.invert_yaxis()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pyplot.close(fig)
        return data
    
class CalibrationWindow(FramelessMainWindow, GlobalSharedClass):
    def __init__(self, mainApp, imagePath):
        GlobalSharedClass.__init__(self)
        super().__init__()

        self.rawData = mainApp.rawDataFromDetection
        self.image = cv2.imread(imagePath)
        self.imageCopy = None
        self.qimg = None
        self.imageX = None
        self.imageY = None
        self.scaleX = None
        self.scaleY = None
        self.imageWidth = None
        self.imageHeight = None
        self.button = None
        self.radius = 20
        self.colorInactive = (0, 0, 0)
        self.colorActive = (0, 0, 255)
        self.colorMapped = (0, 255, 255)
        self.circleRadius = 10
        self.repeat = False
        self.uv_coords = []
        self.dir_vectors = {}
        self.pointsInRadius = []
        self.mappedPoints = {}
        self.mappedPointsToDraw = []
        self.calPointIndex = 1
        self.orderText = {1: "st", 2: "nd", 3: "rd", 4: "th", 5: "th"}
        self.circleCenter = None
        self.circleActive = False

        self.loader = QUiLoader()
        self.planeNormal = np.array([0, 1, 0])
        self.planeCenter = np.array([0, -500, 0])
        self.planeRot = np.array([0, 0, 180])
        self.cameraPos = np.array([0, -50, 0])
        self.cameraRot = np.array([90, 0, 0])

        self.__mainWidget = QWidget()
        ui = QFile("ui/calibrationPopupWindow.ui")
        ui.open(QFile.ReadOnly)
        self.__mainWidget = self.loader.load(ui)
        ui.close()
        self.setWindowTitle('Calibration Window')
        self.setWindowIcon('public/light.png')
        self.mainWidget = self.centralWidget()
        lay = self.mainWidget.layout()
        lay.addWidget(self.__mainWidget)
        self.mainWidget.setLayout(lay)
        self.setCentralWidget(self.mainWidget)
        self.setGeometry(200, 50, 1024, 636)
        self.setFixedSize(1024, 636)

        # Title bar design
        self.setupTitleBar(self)

        self.__mainWidget.radiusInput.setValidator(QRegularExpressionValidator(self.radiusRegex))
        self.__mainWidget.radiusInput.setText(str(self.radius))
        self.__mainWidget.radiusInput.textChanged.connect(self.setRadius)

        self.__mainWidget.colorActive.clicked.connect(self.setActiveColor)
        self.__mainWidget.colorInactive.clicked.connect(self.setInactiveColor)
        self.__mainWidget.colorActive.setStyleSheet(f'QPushButton {{background-color: #FF0000; border: 5px solid #FFE81F;}}')
        self.__mainWidget.colorInactive.setStyleSheet(f'QPushButton {{background-color: #000000; border: 5px solid #FFE81F;}}')
        self.__mainWidget.label.setText(f'Image size in pixels: {self.image.shape[1]}x{self.image.shape[0]}')

        self.__mainWidget.save.clicked.connect(self.saveData)
        self.__mainWidget.save.setEnabled(False)
        self.__mainWidget.save.setText(f"Save for {self.calPointIndex}{self.orderText[self.calPointIndex]} point")

        self.rawToPoint()
        self.renderImage()

    def saveData(self):
        if len(self.pointsInRadius) == 0 and self.calPointIndex != 6:
            return
        
        if self.calPointIndex < 6:
            self.mappedPoints[self.calPointIndex] = self.pointsInRadius
            self.mappedPointsToDraw = [*self.mappedPointsToDraw, *self.pointsInRadius]
            self.pointsInRadius = []
            self.circleActive = False
            self.image = self.imageCopy.copy()
            self.displayImage(self.image)
            self.calPointIndex += 1
            self.__mainWidget.save.setEnabled(False if self.calPointIndex != 6 else True)
            self.__mainWidget.save.setText(f"Save for {self.calPointIndex}{self.orderText[self.calPointIndex]} point" if self.calPointIndex != 6 else "Calibrate")

        else:
            self.calibrate()

    def calibrate(self):
        for i in self.mappedPoints:
            print("Point: ", i)
            for j in self.mappedPoints[i]:
                print(self.rawData[j[1]])

            print("\n\n")

    def setRadius(self):
        if self.__mainWidget.radiusInput.text() != '':
            self.radius = int(self.__mainWidget.radiusInput.text())

    def setActiveColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.__mainWidget.colorActive.setStyleSheet(f'QPushButton {{background-color: {color.name()}; border: 5px solid #FFE81F;}}')
            self.colorActive = (b, g, r)
            self.drawPoints()

    def setInactiveColor(self):
        color = QColorDialog.getColor()
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self.__mainWidget.colorInactive.setStyleSheet(f'QPushButton {{background-color: {color.name()}; border: 5px solid #FFE81F;}}')
            self.colorInactive = (b, g, r)
            self.drawPoints()

    def rawToPoint(self):
        for i in self.rawData:
            self.dir_vectors[i] = {"sphere": np.array(self.transfer_vector(self.rawData[i]["sphere"]["center"], 
                                                                           self.cameraPos, self.cameraRot)),
                                   "circle_3d": np.array(self.transfer_vector(self.rawData[i]["circle_3d"]["center"],
                                                                              self.cameraPos, self.cameraRot))}

        for i in self.dir_vectors:
            rayOrigin = self.dir_vectors[i]["sphere"]
            rayDirection = self.normalize(np.array(self.dir_vectors[i]["circle_3d"]) - self.dir_vectors[i]["sphere"])
            intersectionTime = self.intersectPlane(self.planeNormal, self.planeCenter, rayOrigin, rayDirection)
            
            if (intersectionTime > 0.0):
                planeIntersection = self.getPoint([rayOrigin, rayDirection], intersectionTime)
                # TODO: world to display local transformation
                planeIntersection = self.transfer_vector(planeIntersection, self.planeCenter, self.planeRot)
                #planeIntersection[0] = -planeIntersection[0] # otočena obrazovka
                result = self.convert_to_uv(planeIntersection)
                if result:
                    self.uv_coords.append((result, i))

    def renderImage(self):
        for i in range(0, len(self.uv_coords)):
            self.uv_coords[i] = (self.convert_uv_to_px(self.uv_coords[i][0], self.image.shape[1], self.image.shape[0]), self.uv_coords[i][1])

        for i in self.uv_coords:
            cv2.circle(self.image, i[0], 1, (0, 0, 0), -1)

        self.displayImage(self.image)

    def drawCircle(self):
        if not self.circleActive and self.circleCenter:
            self.imageCopy = self.image.copy()
            cv2.circle(self.imageCopy, self.circleCenter, self.radius, (128, 250, 200), -1)
            self.circleActive = True

            self.displayImage(self.imageCopy)

        elif self.circleActive and self.circleCenter == "reset":
            self.imageCopy = self.image.copy()
            self.displayImage(self.imageCopy)
            self.circleActive = False

    def displayImage(self, img=None):
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = image.shape
        self.qimg = QImage(image.data, w, h, c * w, QImage.Format_RGB888)

        if image.shape[1] > self.__mainWidget.image.width() or image.shape[0] > self.__mainWidget.image.height():
            scaled = QPixmap.fromImage(self.qimg).scaled(self.__mainWidget.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.__mainWidget.image.setPixmap(scaled)

            if image.shape[1] > self.__mainWidget.image.width() and image.shape[0] > self.__mainWidget.image.height():
                self.scaleX = self.__mainWidget.image.width() / image.shape[1]
                self.scaleY = self.__mainWidget.image.height() / image.shape[0]

            elif image.shape[1] > self.__mainWidget.image.width():
                self.scaleX = self.__mainWidget.image.width() / image.shape[1]
                self.scaleY = self.scaleX

            elif image.shape[0] > self.__mainWidget.image.height():
                self.scaleY = self.__mainWidget.image.height() / image.shape[0]
                self.scaleX = self.scaleY 

            self.imageX = (self.__mainWidget.image.width() - scaled.width()) / 2
            self.imageY = (self.__mainWidget.image.height() - scaled.height()) / 2
            self.imageHeight = scaled.height()
            self.imageWidth = scaled.width()
        else:
            self.imageX = (self.__mainWidget.image.width() - image.shape[1]) / 2
            self.imageY = (self.__mainWidget.image.height() - image.shape[0]) / 2
            self.scaleX = 1
            self.scaleY = 1
            self.imageHeight = image.shape[0]
            self.imageWidth = image.shape[1]
            self.__mainWidget.image.setPixmap(QPixmap.fromImage(self.qimg))
            self.__mainWidget.image.setAlignment(Qt.AlignCenter)

    def mousePressEvent(self, event):
        x, y = event.pos().x(), event.pos().y()

        if (event.button() == Qt.LeftButton or event.button() == Qt.RightButton) and self.calPointIndex < 6:
            self.button = 'left' if event.button() == Qt.LeftButton else 'right'
            if y > 35 + self.imageY and y < 35 + self.imageY + self.imageHeight and x >= 0 + self.imageX and x <= self.imageX + self.imageWidth:
                x = (x - self.imageX) / self.scaleX
                y = (y - self.imageY - 35) / self.scaleY
                for i in self.uv_coords:
                    if self.distance(i[0], (x, y)) <= self.radius:
                        if self.button == 'left' and not self.circleActive:
                            if not True in map(lambda x: x[1] == i[1], self.pointsInRadius) \
                                and not True in map(lambda x: x[1] == i[1], self.mappedPointsToDraw):
                                self.pointsInRadius.append(i)               

                if self.button == 'left' and not self.circleActive:
                    self.circleCenter = (int(x), int(y))
                elif self.button == 'right' and self.circleActive:
                    self.circleCenter = "reset"
                    self.pointsInRadius = []

                if len(self.pointsInRadius) > 0:
                    self.__mainWidget.save.setEnabled(True)
                else:
                    self.__mainWidget.save.setEnabled(False)
                self.drawCircle()


if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow {background: '#171923';}") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# TODO: kalibracia a validacia
# TODO: neskor pridať dlib na detekciu zrenice .. funguje na zaklade machine learningu
# TODO: pre kameru pridať velkosť obrazku do configu
# TODO: dalšiu iteraciu navrhu .. že čo sa zmenilo
# TODO: prerobiť veci v scanpath aby sa volalo iba to čo je potrebné
# TODO: obrazky v overleaf dať vedla seba ale iba tie ktore spolu suvisia
# TODO: kapitola nema byť prazdna .. doplniť nejaky uvod ..
# TODO: heatmapa a scanpath .. dať do vlastnej kapitoly a vysvetliť to .. dať to do analyzy
# TODO: z druhej iteracie navrhu dať .. budeme potrebovať marker ..
# TODO: možno opisať ako ziskať ppi monitora

# TODO: tie uv ktore su mimo 0 a 1 ... vratiť asi len pri hlavnom okne .. tam mame 3D debug kde to bude vidieť