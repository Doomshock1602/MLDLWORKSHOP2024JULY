# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:27:31 2024

@author: kmmur
"""

import cv2
import os

# Initialize the camera
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Load the face cascade classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id and press <return> ==>  ')

print("\n [INFO] Initializing face capture. Look at the camera and wait ...")

# Initialize individual sampling face count
count = 0

# Ensure the dataset directory exists
dataset_dir = 'dataset/' + str(face_id)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

while True:
    ret, img = cam.read()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the dataset folder
        cv2.imwrite(f"{dataset_dir}/User.{face_id}.{count}.jpg", gray[y:y+h, x:x+w])

        # Display the image with rectangle around the face
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >= 200: # Capture 200 face samples and stop video
         break

# Clean up
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
