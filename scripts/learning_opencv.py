import numpy as np
import cv2
from pye3d import Detector2D, Detector3D, CameraModel, DetectorMode

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    print(cap.get(3))
    print(cap.get(4))
    print(cap.get(5))
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


# def main(path):
#     # create 2D detector
#     detector_2d = Detector2D()
#     # create pye3D detector
#     camera = CameraModel(focal_length=561.5, resolution=[400, 400])
#     detector_3d = Detector3D(camera=camera, long_term_mode=DetectorMode.blocking)
    
#     eye_coordinates = ()
#     eye_video = cv2.VideoCapture(path)
#     # read each frame of video and run pupil detectors
#     while eye_video.isOpened():
#         frame_number = eye_video.get(cv2.CAP_PROP_POS_FRAMES)
#         fps = eye_video.get(cv2.CAP_PROP_FPS)
#         ret, eye_frame = eye_video.read()
    
#         if ret:
#             # dobre pre iris ...
#             #(thresh, blackAndWhiteImage) = cv2.threshold(eye_frame, 50, 255, cv2.THRESH_TOZERO_INV)
#             # (thresh, blackAndWhiteImage) = cv2.threshold(eye_frame, 120, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
#             # cv2.GaussianBlur(blackAndWhiteImage, (17, 17), cv2.BORDER_DEFAULT)
#             # cv2.medianBlur(blackAndWhiteImage, 21)

#             # edges = cv2.Canny(blackAndWhiteImage, 100, 200)
#             # cv2.imshow('frame', blackAndWhiteImage)
#             # cv2.imshow('edges', edges)

#             # read video frame as numpy array
#             grayscale_array = cv2.cvtColor(eye_frame, cv2.COLOR_BGR2GRAY)
#             # run 2D detector on video frame
#             result_2d = detector_2d.detect(grayscale_array)

#             result_2d["timestamp"] = frame_number / fps
#             # pass 2D detection result to 3D detector
#             result_3d = detector_3d.update_and_detect(result_2d, grayscale_array)

#             ellipse_3d = result_3d["ellipse"]
            
#             #print(ellipse_3d)
#             # draw 3D detection result on eye frame
#             cv2.ellipse(
#                 eye_frame,
#                 tuple(int(v) for v in ellipse_3d["center"]),
#                 tuple(int(v / 2) for v in ellipse_3d["axes"]),
#                 ellipse_3d["angle"],
#                 0,
#                 360,  # start/end angle for drawing
#                 (0, 255, 0),  # color (BGR): red
#             )
#             cv2.circle(
#                 eye_frame,
#                 tuple(int(v) for v in ellipse_3d["center"]),
#                 2,
#                 (0, 0, 255),  # color (BGR): blue
#                 thickness=-2,
#             )
#             # show frame
#             cv2.imshow("eye_frame", eye_frame)
#             # press esc to exit
#             if cv2.waitKey(1) & 0xFF == 27:
#                 break
#         else:
#             break
#     eye_video.release()
#     cv2.destroyAllWindows()