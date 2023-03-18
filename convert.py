import os
from PIL import Image
import cv2
from pye3d import Detector2D, Detector3D, CameraModel, DetectorMode
from os.path import isfile, join
# generate ui pyside6-uic ./ui/main.ui > ui_mainwindow.py

for file in os.listdir("dataset/data"):
    filename, extension  = os.path.splitext(file)
    if extension == ".PGM":
        new_file = f"{filename}.png"
        with Image.open(f"dataset/data/{file}") as im:
            im.save(f"dataset/data_png/{new_file}")


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