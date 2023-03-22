import sys
import os
import glob
import cv2
import re
import json

from PySide6.QtCore import Qt
from pyqt_frameless_window import FramelessMainWindow
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QPixmap, QImage, QRegularExpressionValidator
from PySide6.QtCore import QFile, QRegularExpression
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QWidget, QButtonGroup
from pupil_detectors import Detector2D
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

class MainWindow(FramelessMainWindow):
    detector_2d = Detector2D()
    previewDetector2d = Detector2D()
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
    image = None
    imageFlag = 'Simple'
    lastDetectionImage = None

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

        # Detector 3D config values
        self.__testWidget.thresholdSwirski.setText(str(self.config["detector_3d"]['threshold_swirski']))
        self.__testWidget.thresholdKalman.setText(str(self.config["detector_3d"]['threshold_kalman']))
        self.__testWidget.thresholdShortTerm.setText(str(self.config["detector_3d"]['threshold_short_term']))
        self.__testWidget.thresholdLongTerm.setText(str(self.config["detector_3d"]['threshold_long_term']))
        self.__testWidget.longTermBufferSize.setText(str(self.config["detector_3d"]['long_term_buffer_size']))
        self.__testWidget.longTermForgetTime.setText(str(self.config["detector_3d"]['long_term_forget_time']))
        self.__testWidget.longTermForgetObservations.setText(str(self.config["detector_3d"]['long_term_forget_observations']))
        self.__testWidget.longTermMode.setText("asynchronous" if str(self.config["detector_3d"]['long_term_mode']) == "1" else "blocking")
        self.__testWidget.modelUpdateIntervalLongTerm.setText(str(self.config["detector_3d"]['model_update_interval_long_term']))
        self.__testWidget.modelUpdateIntervalUltLongTerm.setText(str(self.config["detector_3d"]['model_update_interval_ult_long_term']))
        self.__testWidget.modelWarmupDuration.setText(str(self.config["detector_3d"]['model_warmup_duration']))
        self.__testWidget.calculateRmsResidual.setText(str(bool(self.config["detector_3d"]['calculate_rms_residual'])))

        # Validators
        floatingRegex = QRegularExpression("^(0|[1-9]\\d*)(\\.\\d+)?$")
        integerRegex = QRegularExpression("^0|[1-9]\\d*$")
        boolRegex = QRegularExpression("^True|False$")
        detectorModeRegex = QRegularExpression("^blocking|asynchronous$")

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

        # Detector 3D validation
        self.__testWidget.thresholdSwirski.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.thresholdKalman.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.thresholdShortTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.thresholdLongTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.longTermBufferSize.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.longTermForgetTime.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.longTermForgetObservations.setValidator(QRegularExpressionValidator(integerRegex))
        self.__testWidget.longTermMode.setValidator(QRegularExpressionValidator(detectorModeRegex))
        self.__testWidget.modelUpdateIntervalLongTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.modelUpdateIntervalUltLongTerm.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.modelWarmupDuration.setValidator(QRegularExpressionValidator(floatingRegex))
        self.__testWidget.calculateRmsResidual.setValidator(QRegularExpressionValidator(boolRegex))

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

        # Detector 3D event listeners
        self.__testWidget.thresholdSwirski.textChanged.connect(self.configChanged)
        self.__testWidget.thresholdKalman.textChanged.connect(self.configChanged)
        self.__testWidget.thresholdShortTerm.textChanged.connect(self.configChanged)
        self.__testWidget.thresholdLongTerm.textChanged.connect(self.configChanged)
        self.__testWidget.longTermBufferSize.textChanged.connect(self.configChanged)
        self.__testWidget.longTermForgetTime.textChanged.connect(self.configChanged)
        self.__testWidget.longTermForgetObservations.textChanged.connect(self.configChanged)
        self.__testWidget.longTermMode.textChanged.connect(self.configChanged)
        self.__testWidget.modelUpdateIntervalLongTerm.textChanged.connect(self.configChanged)
        self.__testWidget.modelUpdateIntervalUltLongTerm.textChanged.connect(self.configChanged)
        self.__testWidget.modelWarmupDuration.textChanged.connect(self.configChanged)
        self.__testWidget.calculateRmsResidual.textChanged.connect(self.configChanged)

        # Manipulate config
        self.__testWidget.saveParameters.setEnabled(False)
        self.__testWidget.saveParameters.clicked.connect(self.saveParameters)
        self.__testWidget.resetParameters.clicked.connect(self.resetParameters)
        self.__testWidget.uploadSaved.clicked.connect(self.uploadSavedParameters)

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
        self.__testWidget.listImages.itemClicked.connect(self.imageClicked)
        self.__testWidget.startButton.setEnabled(False)
        self.__testWidget.startButton.clicked.connect(self.startDetection)
        self.__testWidget.reanalyze.clicked.connect(self.reanalyze)
        self.__testWidget.loadImage.clicked.connect(self.loadImage)
        self.__testWidget.imagePath.setText("Choose image")
        self.__testWidget.imagePath.setText(self.__testWidget.imagePath.fontMetrics().elidedText(self.__testWidget.imagePath.text(), Qt.ElideRight, self.__testWidget.imagePath.width()))
        self.radioButtons = QButtonGroup()
        self.radioButtons.addButton(self.__testWidget.rawRadio)
        self.radioButtons.addButton(self.__testWidget.ellipseRadio)
        self.radioButtons.addButton(self.__testWidget.debugRadio)
        self.__testWidget.ellipseRadio.setChecked(True)
        self.radioButtons.buttonClicked.connect(self.radioClicked)

    def radioClicked(self, button):
        self.imageFlag = button.text().split(" ")[0]
        if self.clickedItem:
            self.imageClicked(item=self.clickedItem)
        elif self.lastDetectionImage:
            self.imageClicked(lastImage=self.lastDetectionImage)

    def configChanged(self):
        pupil_size_min = int(self.__testWidget.pupilSizeMin.text()) if self.__testWidget.pupilSizeMin.text() != "" else None
        pupil_size_max = int(self.__testWidget.pupilSizeMax.text()) if self.__testWidget.pupilSizeMax.text() != "" else None
        strong_perimeter_min = float(self.__testWidget.strongPerimeterMin.text()) if self.__testWidget.strongPerimeterMin.text() != "" else None
        strong_perimeter_max = float(self.__testWidget.strongPerimeterMax.text()) if self.__testWidget.strongPerimeterMax.text() != "" else None
        strong_area_min = float(self.__testWidget.strongAreaMin.text()) if self.__testWidget.strongAreaMin.text() != "" else None
        strong_area_max = float(self.__testWidget.strongAreaMax.text()) if self.__testWidget.strongAreaMax.text() != "" else None
        final_perimeter_min = float(self.__testWidget.finalPerimeterMin.text()) if self.__testWidget.finalPerimeterMin.text() != "" else None
        final_perimeter_max = float(self.__testWidget.finalPerimeterMax.text()) if self.__testWidget.finalPerimeterMax.text() != "" else None

        if self.__testWidget.focalLength.text() != ""  \
            and self.__testWidget.thresholdSwirski.text() != "" \
            and self.__testWidget.thresholdKalman.text() != "" \
            and self.__testWidget.thresholdShortTerm.text() != "" \
            and self.__testWidget.thresholdLongTerm.text() != "" \
            and self.__testWidget.longTermBufferSize.text() != "" \
            and self.__testWidget.longTermForgetTime.text() != "" \
            and self.__testWidget.longTermForgetObservations.text() != "" \
            and (self.__testWidget.longTermMode.text() == "blocking" or self.__testWidget.longTermMode.text() == "asynchronous") \
            and self.__testWidget.modelUpdateIntervalLongTerm.text() != "" \
            and self.__testWidget.modelUpdateIntervalUltLongTerm.text() != "" \
            and self.__testWidget.modelWarmupDuration.text() != "" \
            and (self.__testWidget.calculateRmsResidual.text() == "True" or self.__testWidget.calculateRmsResidual.text() == "False") \
            and self.__testWidget.intensityRange.text() != "" \
            and self.__testWidget.pupilSizeMax.text() != "" \
            and self.__testWidget.pupilSizeMin.text() != "" \
            and self.__testWidget.blurSize.text() != "" \
            and self.__testWidget.cannyTreshold.text() != "" \
            and self.__testWidget.cannyRation.text() != "" \
            and self.__testWidget.cannyAperture.text() != "" \
            and self.__testWidget.coarseFilterMin.text() != "" \
            and self.__testWidget.coarseFilterMax.text() != "" \
            and (self.__testWidget.coarseDetection.text() == "True" or self.__testWidget.coarseDetection.text() == "False") \
            and self.__testWidget.contourSizeMin.text() != "" \
            and self.__testWidget.strongPerimeterMin.text() != "" \
            and self.__testWidget.strongPerimeterMax.text() != "" \
            and self.__testWidget.strongAreaMin.text() != "" \
            and self.__testWidget.strongAreaMax.text() != "" \
            and self.__testWidget.ellipseRoudnessRatio.text != "" \
            and self.__testWidget.initialEllipseTreshhold.text() != "" \
            and self.__testWidget.finalPerimeterMin.text() != "" \
            and self.__testWidget.finalPerimeterMax.text() != "" \
            and self.__testWidget.ellipseSupportMinDist.text() != "" \
            and self.__testWidget.supportPixelRatio.text() != "":
            if pupil_size_min is not None and pupil_size_max is not None and pupil_size_max > pupil_size_min \
                and strong_perimeter_min is not None and strong_perimeter_max is not None and strong_perimeter_max > strong_perimeter_min \
                    and strong_area_min is not None and strong_area_max is not None and strong_area_max > strong_area_min \
                        and final_perimeter_min is not None and final_perimeter_max is not None and final_perimeter_max > final_perimeter_min:
                self.setParameters()
            else:
                self.__testWidget.saveParameters.setEnabled(False)

        else:
            self.__testWidget.saveParameters.setEnabled(False)
    
    def setParameters(self):
        # Camera parameters
        self.config['focal_length'] = float(self.__testWidget.focalLength.text())

        # 2D detector parameters
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

        # 3D detector parameters
        self.config["detector_3d"]["threshold_swirski"] = float(self.__testWidget.thresholdSwirski.text())
        self.config["detector_3d"]["threshold_kalman"] = float(self.__testWidget.thresholdKalman.text())
        self.config["detector_3d"]["threshold_short_term"] = float(self.__testWidget.thresholdShortTerm.text())
        self.config["detector_3d"]["threshold_long_term"] = float(self.__testWidget.thresholdLongTerm.text())
        self.config["detector_3d"]["long_term_buffer_size"] = int(self.__testWidget.longTermBufferSize.text())
        self.config["detector_3d"]["long_term_forget_time"] = int(self.__testWidget.longTermForgetTime.text())
        self.config["detector_3d"]["long_term_forget_observations"] = int(self.__testWidget.longTermForgetObservations.text())
        self.config["detector_3d"]["long_term_mode"] = 1 if self.__testWidget.longTermMode.text() == "asynchronous" else 0
        self.config["detector_3d"]["model_update_interval_long_term"] = float(self.__testWidget.modelUpdateIntervalLongTerm.text())
        self.config["detector_3d"]["model_update_interval_ult_long_term"] = float(self.__testWidget.modelUpdateIntervalUltLongTerm.text())
        self.config["detector_3d"]["model_warmup_duration"] = float(self.__testWidget.modelWarmupDuration.text())
        self.config["detector_3d"]["calculate_rms_residual"] = int(self.__testWidget.calculateRmsResidual.text() == "True")
      
        self.detector_2d_config = self.config["detector_2d"].copy()
        self.detector_2d_config["coarse_detection"] = bool(self.detector_2d_config["coarse_detection"])
        self.detector_3d_config = self.config["detector_3d"].copy()
        self.detector_3d_config["long_term_mode"] = DetectorMode.blocking if int(self.detector_3d_config["long_term_mode"]) == 0 else DetectorMode.asynchronous
        self.detector_3d_config["calculate_rms_residual"] = bool(self.detector_3d_config["calculate_rms_residual"])
        self.previewDetector2d = Detector2D(self.detector_2d_config)
        if self.clickedItem:
            self.imageClicked(self.clickedItem)
        self.__testWidget.saveParameters.setEnabled(True)

    def resetParameters(self):
        self.__testWidget.focalLength.setText(str(self.default_config['focal_length']))
        self.__testWidget.intensityRange.setText(str(self.default_config["detector_2d"]['intensity_range']))
        self.__testWidget.pupilSizeMax.setText(str(self.default_config["detector_2d"]['pupil_size_max']))
        self.__testWidget.pupilSizeMin.setText(str(self.default_config["detector_2d"]['pupil_size_min']))
        self.__testWidget.blurSize.setText(str(self.default_config["detector_2d"]['blur_size']))
        self.__testWidget.cannyTreshold.setText(str(self.default_config["detector_2d"]['canny_threshold']))
        self.__testWidget.cannyRation.setText(str(self.default_config["detector_2d"]['canny_ration']))
        self.__testWidget.cannyAperture.setText(str(self.default_config["detector_2d"]['canny_aperture']))
        self.__testWidget.coarseFilterMin.setText(str(self.default_config["detector_2d"]['coarse_filter_min']))
        self.__testWidget.coarseFilterMax.setText(str(self.default_config["detector_2d"]['coarse_filter_max']))
        self.__testWidget.coarseDetection.setText(str(bool(self.default_config["detector_2d"]['coarse_detection'])))
        self.__testWidget.contourSizeMin.setText(str(self.default_config["detector_2d"]['contour_size_min']))
        self.__testWidget.strongPerimeterMin.setText(str(self.default_config["detector_2d"]['strong_perimeter_ratio_range_min']))
        self.__testWidget.strongPerimeterMax.setText(str(self.default_config["detector_2d"]['strong_perimeter_ratio_range_max']))
        self.__testWidget.strongAreaMin.setText(str(self.default_config["detector_2d"]['strong_area_ratio_range_min']))
        self.__testWidget.strongAreaMax.setText(str(self.default_config["detector_2d"]['strong_area_ratio_range_max']))
        self.__testWidget.ellipseRoudnessRatio.setText(str(self.default_config["detector_2d"]['ellipse_roundness_ratio']))
        self.__testWidget.initialEllipseTreshhold.setText(str(self.default_config["detector_2d"]['initial_ellipse_fit_threshhold']))
        self.__testWidget.finalPerimeterMin.setText(str(self.default_config["detector_2d"]['final_perimeter_ratio_range_min']))
        self.__testWidget.finalPerimeterMax.setText(str(self.default_config["detector_2d"]['final_perimeter_ratio_range_max']))
        self.__testWidget.ellipseSupportMinDist.setText(str(self.default_config["detector_2d"]['ellipse_true_support_min_dist']))
        self.__testWidget.supportPixelRatio.setText(str(self.default_config["detector_2d"]['support_pixel_ratio_exponent']))
        self.__testWidget.thresholdSwirski.setText(str(self.default_config["detector_3d"]['threshold_swirski']))
        self.__testWidget.thresholdKalman.setText(str(self.default_config["detector_3d"]['threshold_kalman']))
        self.__testWidget.thresholdShortTerm.setText(str(self.default_config["detector_3d"]['threshold_short_term']))
        self.__testWidget.thresholdLongTerm.setText(str(self.default_config["detector_3d"]['threshold_long_term']))
        self.__testWidget.longTermBufferSize.setText(str(self.default_config["detector_3d"]['long_term_buffer_size']))
        self.__testWidget.longTermForgetTime.setText(str(self.default_config["detector_3d"]['long_term_forget_time']))
        self.__testWidget.longTermForgetObservations.setText(str(self.default_config["detector_3d"]['long_term_forget_observations']))
        self.__testWidget.longTermMode.setText("asynchronous" if str(self.default_config["detector_3d"]['long_term_mode']) == "1" else "blocking")
        self.__testWidget.modelUpdateIntervalLongTerm.setText(str(self.default_config["detector_3d"]['model_update_interval_long_term']))
        self.__testWidget.modelUpdateIntervalUltLongTerm.setText(str(self.default_config["detector_3d"]['model_update_interval_ult_long_term']))
        self.__testWidget.modelWarmupDuration.setText(str(self.default_config["detector_3d"]['model_warmup_duration']))
        self.__testWidget.calculateRmsResidual.setText(str(bool(self.default_config["detector_3d"]['calculate_rms_residual'])))

    def saveParameters(self):
        self.config["detector_3d"]["long_term_mode"] = self.config["detector_3d"]["long_term_mode"]
        self.config["detector_3d"]["calculate_rms_residual"] = 1 if self.config["detector_3d"]["calculate_rms_residual"] else 0
        self.config["detector_2d"]["coarse_detection"] = 1 if self.config["detector_2d"]["coarse_detection"] else 0
        with open('config/config.json', 'w') as outfile:
            json.dump(self.config, outfile)
        self.__testWidget.saveParameters.setEnabled(False)

    def uploadSavedParameters(self):
        with open('config/config.json') as json_file:
            self.original_config = json.load(json_file)
        self.__testWidget.focalLength.setText(str(self.original_config['focal_length']))
        self.__testWidget.coarseDetection.setText(str(bool(self.original_config["detector_2d"]['coarse_detection'])))
        self.__testWidget.coarseFilterMin.setText(str(self.original_config["detector_2d"]['coarse_filter_min']))
        self.__testWidget.coarseFilterMax.setText(str(self.original_config["detector_2d"]['coarse_filter_max'])) 
        self.__testWidget.intensityRange.setText(str(self.original_config["detector_2d"]['intensity_range']))
        self.__testWidget.blurSize.setText(str(self.original_config["detector_2d"]['blur_size']))
        self.__testWidget.cannyTreshold.setText(str(self.original_config["detector_2d"]['canny_threshold']))
        self.__testWidget.cannyRation.setText(str(self.original_config["detector_2d"]['canny_ration']))
        self.__testWidget.cannyAperture.setText(str(self.original_config["detector_2d"]['canny_aperture']))
        self.__testWidget.pupilSizeMax.setText(str(self.original_config["detector_2d"]['pupil_size_max']))
        self.__testWidget.pupilSizeMin.setText(str(self.original_config["detector_2d"]['pupil_size_min']))
        self.__testWidget.strongPerimeterMin.setText(str(self.original_config["detector_2d"]['strong_perimeter_ratio_range_min']))
        self.__testWidget.strongPerimeterMax.setText(str(self.original_config["detector_2d"]['strong_perimeter_ratio_range_max']))
        self.__testWidget.strongAreaMin.setText(str(self.original_config["detector_2d"]['strong_area_ratio_range_min']))
        self.__testWidget.strongAreaMax.setText(str(self.original_config["detector_2d"]['strong_area_ratio_range_max']))
        self.__testWidget.contourSizeMin.setText(str(self.original_config["detector_2d"]['contour_size_min']))
        self.__testWidget.ellipseRoudnessRatio.setText(str(self.original_config["detector_2d"]['ellipse_roundness_ratio']))
        self.__testWidget.initialEllipseTreshhold.setText(str(self.original_config["detector_2d"]['initial_ellipse_fit_threshhold']))
        self.__testWidget.finalPerimeterMin.setText(str(self.original_config["detector_2d"]['final_perimeter_ratio_range_min'])) 
        self.__testWidget.finalPerimeterMax.setText(str(self.original_config["detector_2d"]['final_perimeter_ratio_range_max']))
        self.__testWidget.ellipseSupportMinDist.setText(str(self.original_config["detector_2d"]['ellipse_true_support_min_dist']))
        self.__testWidget.supportPixelRatio.setText(str(self.original_config["detector_2d"]['support_pixel_ratio_exponent']))

        self.__testWidget.thresholdSwirski.setText(str(self.original_config["detector_3d"]['threshold_swirski']))
        self.__testWidget.thresholdKalman.setText(str(self.original_config["detector_3d"]['threshold_kalman']))
        self.__testWidget.thresholdShortTerm.setText(str(self.original_config["detector_3d"]['threshold_short_term']))
        self.__testWidget.thresholdLongTerm.setText(str(self.original_config["detector_3d"]['threshold_long_term']))
        self.__testWidget.longTermBufferSize.setText(str(self.original_config["detector_3d"]['long_term_buffer_size']))
        self.__testWidget.longTermForgetTime.setText(str(self.original_config["detector_3d"]['long_term_forget_time']))
        self.__testWidget.longTermForgetObservations.setText(str(self.original_config["detector_3d"]['long_term_forget_observations']))
        self.__testWidget.longTermMode.setText("asynchronous" if str(self.original_config["detector_3d"]['long_term_mode']) == "1" else "blocking")
        self.__testWidget.modelUpdateIntervalLongTerm.setText(str(self.original_config["detector_3d"]['model_update_interval_long_term']))
        self.__testWidget.modelUpdateIntervalUltLongTerm.setText(str(self.original_config["detector_3d"]['model_update_interval_ult_long_term']))
        self.__testWidget.modelWarmupDuration.setText(str(self.original_config["detector_3d"]['model_warmup_duration']))
        self.__testWidget.calculateRmsResidual.setText(str(bool(self.original_config["detector_3d"]['calculate_rms_residual'])))

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
            self.lastDetectionImage = None
            self.clickedItem = None
            self.detectionRound = 0
            self.imagesPaths = {}
            self.rawDataFromDetection = {}
            self.__testWidget.imagePath.setText("Choose image")
            self.__testWidget.startButton.setEnabled(False)
            self.__testWidget.listImages.clear()
            self.__testWidget.imageLabel.clear()

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
        self.__testWidget.imageLabel.setPixmap(QPixmap.fromImage(outImage))
        self.__testWidget.imageLabel.setScaledContents(True)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet("QMainWindow {background: '#171923';}") 
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

# TODO: scan path podobne ako heatmap .. čiarky a body
# TODO: kalibracia a validacia
# TODO: spraviť nejake opatrenie ked je otvoreny overlay aby nenastala nejaka šarapata keby sa vymazalo nieco z obrazku
# TODO: transforms lib knižnica
# TODO: prevod medzi lokal a global coord systemom
# TODO: neskor pridať dlib na detekciu zrenice .. funguje na zaklade machine learningu
# TODO: pre kameru pridať velkosť obrazku do configu
# TODO: filter requirements
# TODO: default config .. pre 3D zmeniť ten blocking and boolean
# TODO: pridať radio button ... aby som mal global seting pre detection aj pre preview
# či chcem elipsu , nič , alebo cely ten debug pre 2D aj 3D