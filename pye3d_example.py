import argparse

import cv2
from pupil_detectors import Detector2D
import numpy as np
import math
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

# create 2D detector
detector_2d = Detector2D()
# create pye3D detector
camera = CameraModel(focal_length=772.55, resolution=[640, 480])
detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
# load eye video
counter = 0
rep = 0
data = {}

def loop(video):
    global counter, data, rep
    while video.isOpened():
        frame_number = video.get(cv2.CAP_PROP_POS_FRAMES)
        fps = video.get(cv2.CAP_PROP_FPS)
        ret, eye_frame = video.read()
        if ret:
            # read video frame as numpy array
            grayscale_array = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
            # run 2D detector on video frame
            result_2d = detector_2d.detect(grayscale_array)
            result_2d["timestamp"] = frame_number / fps
            # pass 2D detection result to 3D detector
            result_3d = detector_3d.update_and_detect(result_2d, grayscale_array)
            
            if rep:
                data[counter] = result_3d["circle_3d"]
            ellipse_3d = result_3d["ellipse"]
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
            # show frame
            cv2.imshow("eye_frame", eye_frame)  
            cv2.waitKey(1)
        
        counter += 1
        if counter == 120:
            counter = 0
            rep = 1
            break

def main(eye_video_path):

    
    eye_video = cv2.VideoCapture(eye_video_path)
    # read each frame of video and run pupil detectors
    loop(eye_video)
    eye_video.release()
    eye_video = cv2.VideoCapture(eye_video_path)
    loop(eye_video)
    eye_video.release()
    cv2.destroyAllWindows()

    for i in data:
        print(data[i], "\n\n")

EYE_RADIUS = 12
CAMERA_POSITION = [20, -50, -10]
CAMERA_TARGET = [0, -EYE_RADIUS, 0]
EYE_POSITION = [0, 0, 0]
EYE_TARGET = [None, -500, None]

def calcPoint():
    pass

if __name__ == "__main__":
    main('dataset-Vincur/synthetizedImages_y_offset_only/example_0.png')
    #calcPoint()

    for i in data:
        print(data[i]['normal'], "\n\n")