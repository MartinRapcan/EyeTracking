import sys
import os
import glob
import cv2
import re
import json
import numpy as np

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, QRegularExpressionValidator
from PySide6.QtCore import QFile, QRegularExpression, Qt, QCoreApplication
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QWidget, QButtonGroup, QColorDialog
from pyqt_frameless_window import FramelessMainWindow
from math import sqrt, atan2, cos, sin
from matplotlib import pyplot, use
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
use('Agg')


class MainWindow(FramelessMainWindow):
    def __init__(self):
        super().__init__()

        self.detector_2d = Detector2D()
        self.previewDetector2d = Detector2D()
        self.camera = None
        self.detector_3d = None
        self.images = {}
        self.imageB = None
        self.angle = 0
        self.loader = QUiLoader()
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
        self.openedWindows = []

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
        titleBar = self.getTitleBar()
        titleBar.setFixedHeight(35)
        
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

        # Validators
        floatingRegex = QRegularExpression("^(0|[1-9]\\d*)(\\.\\d+)?$")
        integerRegex = QRegularExpression("^0|[1-9]\\d*$")
        boolRegex = QRegularExpression("^True|False$")
        detectorModeRegex = QRegularExpression("^blocking|asynchronous$")

        # Camera validation
        self.__mainWidget.focalLength.setValidator(QRegularExpressionValidator(floatingRegex))
        
        # Detector 2D validation
        self.__mainWidget.coarseDetection.setValidator(QRegularExpressionValidator(boolRegex))
        self.__mainWidget.coarseFilterMin.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.coarseFilterMax.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.intensityRange.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.blurSize.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.cannyTreshold.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.cannyRation.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.cannyAperture.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.pupilSizeMax.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.pupilSizeMin.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.strongPerimeterMin.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.strongPerimeterMax.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.strongAreaMin.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.strongAreaMax.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.contourSizeMin.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.ellipseRoudnessRatio.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.initialEllipseTreshhold.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.finalPerimeterMin.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.finalPerimeterMax.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.ellipseSupportMinDist.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.supportPixelRatio.setValidator(QRegularExpressionValidator(floatingRegex))

        # Detector 3D validation
        self.__mainWidget.thresholdSwirski.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.thresholdKalman.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.thresholdShortTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.thresholdLongTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.longTermBufferSize.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.longTermForgetTime.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.longTermForgetObservations.setValidator(QRegularExpressionValidator(integerRegex))
        self.__mainWidget.longTermMode.setValidator(QRegularExpressionValidator(detectorModeRegex))
        self.__mainWidget.modelUpdateIntervalLongTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.modelUpdateIntervalUltLongTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.modelWarmupDuration.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__mainWidget.calculateRmsResidual.setValidator(QRegularExpressionValidator(boolRegex))

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

        # Setup scripts
        self.detector_2d.update_properties(self.detector_2d_config)
        self.previewDetector2d.update_properties(self.detector_2d_config)
        self.camera = CameraModel(focal_length=self.config['focal_length'], resolution=[640, 480])
        self.detector_3d = Detector3D(camera=self.camera)
        self.detector_3d.update_properties(self.detector_3d_config)

        for button in titleBar.findChildren(QPushButton):
            button.setStyleSheet("QPushButton {background-color: #FFE81F; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:hover {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:pressed {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px}")
        titleBar.findChildren(QLabel)[1].setStyleSheet("QLabel {font-size: 15px; color: #F7FAFC; font-weight: bold; margin-left: 10px}")
        titleBar.findChildren(QLabel)[0].setStyleSheet("QLabel {margin-left: 10px}")
        self.__mainWidget.listImages.itemClicked.connect(self.imageClicked)
        self.__mainWidget.startButton.setEnabled(False)
        self.__mainWidget.startButton.clicked.connect(self.startDetection)
        self.__mainWidget.reanalyze.clicked.connect(self.reanalyze)
        self.__mainWidget.loadImage.clicked.connect(self.loadImage)
        self.__mainWidget.scanpath.clicked.connect(self.showScanpath)
        self.__mainWidget.heatmap.clicked.connect(self.showHeatmap)
        self.__mainWidget.rawImage.clicked.connect(self.showRawImage)
        self.__mainWidget.imagePath.setText("Choose image")
        self.__mainWidget.imagePath.setText(self.__mainWidget.imagePath.fontMetrics().elidedText(self.__mainWidget.imagePath.text(), Qt.ElideRight, self.__mainWidget.imagePath.width()))
        self.radioButtons = QButtonGroup()
        self.radioButtons.addButton(self.__mainWidget.rawRadio)
        self.radioButtons.addButton(self.__mainWidget.ellipseRadio)
        self.radioButtons.addButton(self.__mainWidget.debugRadio)
        self.__mainWidget.ellipseRadio.setChecked(True)
        self.radioButtons.buttonClicked.connect(self.radioClicked)

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
        self.__mainWidget.focalLength.setText(str(self.default_config['focal_length']))
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
        self.__mainWidget.focalLength.setText(str(self.original_config['focal_length']))
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
        self.startDetection()

    def loadImage(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\\', "Image files (*.jpg *.png *.jpeg)")
        if fname[0] != "":
            self.imageName = re.search(r'[^/\\&\?]+\.\w+$', fname[0]).group(0)
            self.imagePath = fname[0]
            self.rawDataFromDetection = {}
            self.fillImageList = 0
            self.detectionRound = 0
            self.imagesPaths = {}
            self.lastDetectionImage = None
            self.clickedItem = None
            self.folderPath = os.path.dirname(fname[0])
            file_list = glob.glob(os.path.join(self.folderPath, "*"))
            self.imageAmount = len(file_list)
            self.__mainWidget.startButton.setEnabled(True)
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
            self.rawDataFromDetection = {}
            self.__mainWidget.imagePath.setText("Choose image")
            self.__mainWidget.startButton.setEnabled(False)
            self.__mainWidget.listImages.clear()
            self.__mainWidget.imageLabel.clear()

    def startDetection(self):
        if self.imagePath:
            self.clickedItem = None
            if self.detectionRound == 0:
                self.renderImage()
                if self.fillImageList == 0:
                    self.fillImageList = 1
                self.detectionRound = 1
                self.renderImage()
            else:
                self.rawDataFromDetection = {}
                self.renderImage()

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

            self.displayImage(image)
            cv2.waitKey(10)

    def imageClicked(self, item = None, lastImage = None):
        self.clickedItem = item
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

    def closeEvent(self, event):
        for i in self.openedWindows:
            i.close()
        event.accept()

class VisualizationWindow(FramelessMainWindow):
    def __init__(self, imagePath = None, rawData = None, heatmap = None, scanpath = None):
        super().__init__()

        self.loader = QUiLoader()
        self.planeNormal = np.array([0, 1, 0])
        self.planeCenter = np.array([0, -500, 0])
        self.qimg = None
        self.color1 = (0, 0, 0)
        self.color2 = (255, 255, 255) 

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
        titleBar = self.getTitleBar()
        titleBar.setFixedHeight(35)
        for button in titleBar.findChildren(QPushButton):
            button.setStyleSheet("QPushButton {background-color: #FFE81F; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:hover {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px} QPushButton:pressed {background-color: #ccba18; border-radius: 7px; margin-right: 15px; width: 25px; height: 25px}")
        titleBar.findChildren(QLabel)[1].setStyleSheet("QLabel {font-size: 15px; color: #F7FAFC; font-weight: bold; margin-left: 10px}")
        titleBar.findChildren(QLabel)[0].setStyleSheet("QLabel {margin-left: 10px}")

        self.uv_coords = []	
        self.dir_vectors = {}
        self.points_group = {}
        self.repeat = False
        self.points_group_keys = []
        self.imagePath = imagePath
        self.rawData = rawData
        self.heatmap = heatmap
        self.scanpath = scanpath
        self.__mainWidget.saveImage.clicked.connect(self.saveImage)
        self.__mainWidget.color1.clicked.connect(self.setFirstColor)
        self.__mainWidget.color2.clicked.connect(self.setSecondColor)
        self.__mainWidget.color1.setStyleSheet(f'QPushButton {{background-color: #000000; border: 5px solid #FFE81F;}}')
        self.__mainWidget.color2.setStyleSheet(f'QPushButton {{background-color: #FFFFFF; border: 5px solid #FFE81F;}}')
        if not self.scanpath:
            self.__mainWidget.color1.hide()
            self.__mainWidget.color2.hide()

        if self.rawData:
            self.rawToPoint()

        if self.imagePath:
            self.displayImage()

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


    def dir_vector(self, vec1, vec2):
        return [vec2[0] - vec1[0], vec2[1] - vec1[1], vec2[2] - vec1[2]]

    def transfer_vector(self, vec):
        return [round(vec[0], 2), round(vec[2] - 50, 2), round(-vec[1], 2)]

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

    def convert_to_uv(self, vec, size_x=250, size_y=250, flip_y=True):
        x = (vec[0] + size_x / 2) / size_x
        if flip_y:
            y = (-vec[2] + size_y / 2) / size_y
        else:
            y = (vec[2] + size_y / 2) / size_y
        return (max(0, min(1, x)), max(0, min(1, y)))

    def rawToPoint(self):
        for i in self.rawData:
            self.dir_vectors[i] = {"sphere": np.array(self.transfer_vector(self.rawData[i]["sphere"]["center"])),
                                   "circle_3d": np.array(self.transfer_vector(self.rawData[i]["circle_3d"]["center"]))}

        for i in self.dir_vectors:
            rayOrigin = self.dir_vectors[i]["sphere"]
            rayDirection = self.normalize(np.array(self.dir_vectors[i]["circle_3d"]) - self.dir_vectors[i]["sphere"])
            intersectionTime = self.intersectPlane(self.planeNormal, self.planeCenter, rayOrigin, rayDirection)
            
            if (intersectionTime > 0.0):
                planeIntersection = self.getPoint([rayOrigin, rayDirection], intersectionTime)
                planeIntersection[0] = -planeIntersection[0]
                self.uv_coords.append(self.convert_to_uv(planeIntersection))

    def displayImage(self):
        image = None
        if self.scanpath and self.rawData:
            image = self.scanpathVisualization()
        elif self.heatmap and self.rawData:
            image = self.heatmapVisualization()
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

    def scanpathVisualization(self):
        image = cv2.imread(self.imagePath)
        image_width = image.shape[1]
        image_height = image.shape[0]

        overlay_circles = image.copy()
        overlay_lines = image.copy()
        alpha_circles = 0.6
        outline_width = 10
        alpha_lines = 0.2
        TEXT_FACE = cv2.FONT_HERSHEY_SIMPLEX
        TEXT_SCALE = 0.8
        TEXT_THICKNESS = 2
        TEXT_COLOR = (0, 0, 0)

        colors = {}
        threshold = 50
        order = 0

        if not self.repeat:
            for i in range(0, len(self.uv_coords)):
                self.uv_coords[i] = self.convert_uv_to_px(self.uv_coords[i], image_width, image_height)

            main_point = None
            for i in range(0, len(self.uv_coords)):
                if not main_point:
                    main_point = (self.uv_coords[i][0], self.uv_coords[i][1])

                if abs(self.uv_coords[i][0] - main_point[0]) <= threshold and abs(self.uv_coords[i][1] - main_point[1]) <= threshold:
                    if not self.points_group.get(order):
                        self.points_group[order] = {'points': [self.uv_coords[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0, 'index': order + 1}
                    else:
                        self.points_group[order]['points'].append(self.uv_coords[i])
                
                else:
                    order += 1
                    main_point = (self.uv_coords[i][0], self.uv_coords[i][1])
                    self.points_group[order] = {'points': [self.uv_coords[i]], 'middle': {'x': 0, 'y': 0}, 'diameter': 0, 'index': order + 1}

            self.points_group = dict(sorted(self.points_group.items(), key=lambda item: len(item[1]['points']), reverse=False))
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
                        self.points_group[key]['diameter'] = int(20 * normalized_value)
            else:
                for key in different_lengths:
                    for value in different_lengths[key]:
                        self.points_group[value]['diameter'] = 20
                    

            self.points_group = dict(sorted(self.points_group.items(), key=lambda item: item[1]['index'], reverse=False))
            self.points_group_keys = list(self.points_group)
            self.repeat = True

        t = 1 / (len(self.points_group) - 1) 
        for key in self.points_group:
            r = min(255, max(0, int(self.lerp(self.color1[0], self.color2[0], t))))
            g = min(255, max(0, int(self.lerp(self.color1[1], self.color2[1], t))))
            b = min(255, max(0, int(self.lerp(self.color1[2], self.color2[2], t))))
            colors[key] = (r, g, b)
            t += 1 / (len(self.points_group) - 1)

        for key in range(0, len(self.points_group) - 1):
            x1 = self.points_group[self.points_group_keys[key]]['middle']['x']
            y1 = self.points_group[self.points_group_keys[key]]['middle']['y']
            x2 = self.points_group[self.points_group_keys[key + 1]]['middle']['x']
            y2 = self.points_group[self.points_group_keys[key + 1]]['middle']['y']
            radius1 = self.points_group[self.points_group_keys[key]]['diameter']
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
            if key == len(self.points_group) - 2:
                text_size, _ = cv2.getTextSize(str(self.points_group[self.points_group_keys[key]]['index']), TEXT_FACE, TEXT_SCALE, TEXT_THICKNESS)
                text_origin = (int(x2 - text_size[0] / 2), int(y2 + text_size[1] / 2))
                #cv2.circle(overlay_circles, (x2, y2), radius2, colors[points_group_keys[key + 1]], -1)
                cv2.circle(overlay_circles, (x2, y2), radius2, colors[self.points_group_keys[key + 1]], outline_width)
                #cv2.putText(overlay_circles, str(points_group[points_group_keys[key + 1]]['index']), text_origin, TEXT_FACE, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS, cv2.LINE_AA)


        result = cv2.addWeighted(overlay_circles, alpha_circles, image, 1 - alpha_circles, 0)
        result = cv2.addWeighted(overlay_lines, alpha_lines, result, 1 - alpha_lines, 0)
        return result
    
    def heatmapVisualization(self):
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

        return data

if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow {background: '#171923';}") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# TODO: kalibracia a validacia
# TODO: transforms lib knižnica
# TODO: prevod medzi lokal a global coord systemom
# TODO: neskor pridať dlib na detekciu zrenice .. funguje na zaklade machine learningu
# TODO: pre kameru pridať velkosť obrazku do configu
# TODO: dalšiu iteraciu navrhu .. že čo sa zmenilo