import argparse

import cv2
from pupil_detectors import Detector2D
import numpy as np
import sys
import math
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode

# create 2D detector
detector_2d = Detector2D()
detector_2d_properties = {
    "intensity_range":32,
    "pupil_size_max": 120,
    "pupil_size_min": 50,
}
detector_2d.update_properties(detector_2d_properties)

# create pye3D detector
camera = CameraModel(focal_length=772.55, resolution=[640, 480])
detector_3d_original = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)

rep = 0
data = {}

def loop():
    for i in range(121):
        image = cv2.imread(f"dataset-Vincur/synthetizedImages_no_glint_denoised/example_{i}.png")
        # read video frame as numpy array
        grayscale_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # run 2D detector on video frame
        result_2d = detector_2d.detect(grayscale_array)
        result_2d["timestamp"] = i
        # pass 2D detection result to 3D detector
        result_3d_original = detector_3d_original.update_and_detect(result_2d, grayscale_array, apply_refraction_correction=False)

        # TODO: uncomment for future data collection
        if rep:
            data[i] = result_3d_original
        ellipse_3d_original = result_3d_original["ellipse"]
        # draw 3D detection result on eye frame
        cv2.ellipse(
            image,
            tuple(int(v) for v in ellipse_3d_original["center"]),
            tuple(int(v / 2) for v in ellipse_3d_original["axes"]),
            ellipse_3d_original["angle"],
            0,
            360,  # start/end angle for drawing
            (0, 255, 0),  # color (BGR): red
        )

        # show frame
        cv2.imshow("Detector3D-original", image)  
        cv2.waitKey(5)


def main(eye_video_path):
    global rep
    loop()
    rep = 1
    loop()
    cv2.destroyAllWindows()


EYE_RADIUS = 12
CAMERA_POSITION = [20, -50, -10]
CAMERA_TARGET = [0, -EYE_RADIUS, 0]
EYE_POSITION = [0, 0, 0]
EYE_TARGET = [None, -500, None]

def dir_vector(vec1, vec2):
    return [vec2[0] - vec1[0], vec2[1] - vec1[1], vec2[2] - vec1[2]]

def transfer_vector(vec):
    return [round(vec[0], 2), round(vec[2] - 50, 2), round(-vec[1], 2)]

def intersectPlane(n, p0, l0, l):
    denom = matmul(-n, l)
    if (denom > sys.float_info.min):
        p0l0 = p0 - l0
        t = matmul(p0l0, -n) / denom
        return t
    return -1.0
    
def matmul(v1, v2, pad=False, padBy=1.0):
    if(pad is True):
        return np.matmul(v1, np.append(v2, padBy))[:-1]
    return np.matmul(v1, v2)
    
def getPoint(ray, distance):
    return ray[0] + ray[1] * distance

def normalize(v):
    return v / magnitude(v)
    
def magnitude(v):
    return np.sqrt(sqrMagnitude(v))
        
def sqrMagnitude(v):
    return matmul(v, v)

def convert_to_uv(vec, size_x=250, size_y=250, flip_y=True):
    x = (vec[0] + size_x / 2) / size_x
    if flip_y:
        y = (-vec[2] + size_y / 2) / size_y
    else:
        y = (vec[2] + size_y / 2) / size_y
    return (max(0, min(1, x)), max(0, min(1, y)))

planeNormal = np.array([0, 1, 0])
planeCenter = np.array([0, -500, 0])
dir_vectors = {}
uv_coords = []
def calcPoint():
    for i in data:
        dir_vectors[i] = {"sphere": np.array(transfer_vector(data[i]["sphere"]["center"])),
                          "circle_3d": np.array(transfer_vector(data[i]["circle_3d"]["center"]))}

    for i in dir_vectors:
        rayOrigin = dir_vectors[i]["sphere"]
        rayDirection = normalize(np.array(dir_vectors[i]["circle_3d"]) - dir_vectors[i]["sphere"])
        intersectionTime = intersectPlane(planeNormal, planeCenter, rayOrigin, rayDirection)
        if (intersectionTime > 0.0):
            planeIntersection = getPoint([rayOrigin, rayDirection], intersectionTime)
            planeIntersection[0] = -planeIntersection[0]
            uv_coords.append(convert_to_uv(planeIntersection))

# save it as csv x, y
def saveUVCoords():
    with open('coordinates/uv_coords.csv', 'w') as f:
        for uv in uv_coords:
            f.write(f'{uv[0]}, {uv[1]}\n')

import json
if __name__ == "__main__":
    main('dataset-Vincur/synthetizedImages_no_glint_denoised/example_0.png')
    calcPoint()
    saveUVCoords()
    value ={  
        "a": "1",  
        "b": "2",  
        "c": "4",  
        "d": "8"  
    }  
    # the json file to save the output data   
    save_file = open("config/config.json", "w")  
    json.dump(value, save_file, indent = 6)  
    save_file.close()  