import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')


while True:
    roi_gray = []
    circles = None

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.GaussianBlur(gray, (17, 17), 0)
    cv2.medianBlur(gray, 5)
    cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        eyes = eyes_cascade.detectMultiScale(gray, 1.3, 5)
        if len(eyes) > 0:
            for (x, y, w, h) in eyes:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                roi_gray.append([(x, y), (x + w, y + h)])

            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                                       1, 20, param1=110, param2=30, minRadius=0, maxRadius=30)

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    if roi_gray:
                        for (x, y), (x2, y2) in roi_gray:
                            if x < i[0] < x2 and y < i[1] < y2:
                                cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 1)
                                cv2.circle(frame, (i[0], i[1]), 1, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
