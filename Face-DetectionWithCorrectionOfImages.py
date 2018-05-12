import cv2
import os
import numpy as np

images = []

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = colored_img.copy()

    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    # let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return img_copy

for filename in os.listdir(r"Images"):
    img = cv2.imread(os.path.join(r"Images", filename))

    if img is not None:
        images.append(img)

haar_face_caccade = cv2.CascadeClassifier("OpenCV-files\haarcascade_frontalface_alt.xml")

for img in images:
    faces_detected_img = detect_faces(haar_face_caccade, img)

    # clahe algorithm
    lab = cv2.cvtColor(faces_detected_img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # more brightness filter
    hsv = cv2.cvtColor(faces_detected_img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 1]
    v = np.where(v <= 255 - 30, v + 30, 255)
    hsv[:, :, 1] = v
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # noise reduction
    dst = cv2.fastNlMeansDenoisingColored(faces_detected_img, None, 10, 10, 7, 21)

    cv2.imshow('Image after face detection', faces_detected_img)
    cv2.imshow('Image after cl1', bgr)
    cv2.imshow('Image with more brightness hsv[:, :, 1]', image)
    cv2.imshow('After noise reduction 1', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()