import argparse

import cv2
from pupil_detectors import Detector2D
import numpy as np
import math
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

# create 2D detector
detector_2d = Detector2D()
# create pye3D detector
camera = CameraModel(focal_length=561.5, resolution=[400, 400])
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
            print(result_3d, "\n\n")
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

def subtrackVector(v1, v2):
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]

def divVector(v, n):
    return [v[0] / n, v[1] / n, v[2] / n]

def norm(v):
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]

def quaternion(axis, angle):
    return [math.cos(angle/2), 
            axis[0]*math.sin(angle/2), 
            axis[1]*math.sin(angle/2), 
            axis[2]*math.sin(angle/2)]

def dot(a, b):
    return a[0]*b + a[1]*b + a[2]*b

def arccos(v):
    return math.acos(dot([0, 0, 1], v) / norm(v))

# rewrite eye function to work with lists
def identity(n):
    return [[1 if i == j else 0 for i in range(n)] for j in range(n)]

def translate(v):
    return [[1, 0, 0, -v[0]],
            [0, 1, 0, -v[1]],
            [0, 0, 1, -v[2]],
            [0, 0, 0, 1]]

def calcPoint():
    start_point = CAMERA_POSITION
    target_point = CAMERA_TARGET

    dir_vector = subtrackVector(target_point, start_point)
    dir_unit_vector = divVector(dir_vector, norm(dir_vector))

    rotation_axis = cross([0, 0, 1], dir_unit_vector)
    angle = arccos(dot([0, 0, 1], dir_unit_vector))

    q = quaternion(rotation_axis, angle)

    q = q / norm(q)
    w, x, y, z = q
    R = [[1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w), 0],
         [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w), 0],
         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2), 0],
         [0, 0, 0, 1]]

    print(R)

    eye_normal = data[0]['normal']

    T1 = identity(4)
    T1 = translate(-start_point)

    T2 = identity(4)
    T2 = translate(start_point)

    # Combine all transformations
    M = dot(dot(T2, R), T1)

    # Apply transformation to point (including homogeneous coordinates)
    point_homogeneous = np.concatenate([start_point, [1]])
    new_point_homogeneous = np.dot(M, point_homogeneous)
    new_point = new_point_homogeneous[:3]

    # Print rotated point
    print(new_point, R)
    v = list(np.dot(R[:3, :3].T, data[0]['normal']))

    print(v)

if __name__ == "__main__":
    main('dataset-Vincur/synthetizedImages/example_0.png')
    calcPoint()