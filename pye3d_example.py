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
detector_3d_original = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
detector_3d_test = Detector3D(camera=camera, 
                              long_term_mode=DetectorMode.blocking,
                              threshold_swirski=0.95,
                              threshold_kalman=0.98,
                              long_term_forget_time=20,
                              long_term_buffer_size=50)
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
            eye_frame_test = eye_frame.copy()
            # run 2D detector on video frame
            result_2d = detector_2d.detect(grayscale_array)
            result_2d["timestamp"] = frame_number / fps
            # pass 2D detection result to 3D detector
            result_3d_original = detector_3d_original.update_and_detect(result_2d, grayscale_array)
            result_3d_test = detector_3d_test.update_and_detect(result_2d, grayscale_array)

            # TODO: uncomment for future data collection
            if rep:
                data[counter] = result_3d_original
            ellipse_3d_original = result_3d_original["ellipse"]
            ellipse_3d_test = result_3d_test["ellipse"]
            # draw 3D detection result on eye frame
            cv2.ellipse(
                eye_frame,
                tuple(int(v) for v in ellipse_3d_original["center"]),
                tuple(int(v / 2) for v in ellipse_3d_original["axes"]),
                ellipse_3d_test["angle"],
                0,
                360,  # start/end angle for drawing
                (0, 255, 0),  # color (BGR): red
            )
            cv2.ellipse(
                eye_frame_test,
                tuple(int(v) for v in ellipse_3d_test["center"]),
                tuple(int(v / 2) for v in ellipse_3d_test["axes"]),
                ellipse_3d_test["angle"],
                0,
                360,  # start/end angle for drawing
                (0, 255, 0),  # color (BGR): red
            )

            # show frame
            cv2.imshow("Detector3D-original", eye_frame)  
            cv2.imshow("Detector3D-test", eye_frame_test)
            #print("Frame number: ", counter)
            cv2.waitKey(5)
        
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


EYE_RADIUS = 12
CAMERA_POSITION = [20, -50, -10]
CAMERA_TARGET = [0, -EYE_RADIUS, 0]
EYE_POSITION = [0, 0, 0]
EYE_TARGET = [None, -500, None]

def dir_vector(vec1, vec2):
    return [vec2[0] - vec1[0], vec2[1] - vec1[1], vec2[2] - vec1[2]]

dir_vectors = {}
def calcPoint():
    for i in data:
        dir_vectors[i] = dir_vector(data[i]['sphere']['center'], data[i]['circle_3d']['center'])

    for i in dir_vectors:
        print(dir_vectors[i])

if __name__ == "__main__":
    #main('dataset-Vincur/test/example_0.png')
    #TODO: images that not being detected correctly 18 43 71 86 87 108
    main('dataset-Vincur/synthetizedImages_no_glint_denoised/example_0.png')
    calcPoint()