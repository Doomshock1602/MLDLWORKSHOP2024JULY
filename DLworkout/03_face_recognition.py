import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the trained face recognition model
model = load_model('face_recognition_model.h5')

# Path to the dataset directory
dataset_dir = 'dataset'

# Fetch class names from the dataset directory
class_names = {}
for class_index, class_name in enumerate(sorted(os.listdir(dataset_dir))):
    class_names[class_index] = class_name

# Initialize OpenCV
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height

# Load the face cascade classifier
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, img = cam.read()

    if not ret:
        print("[ERROR] Camera capture failed")
        break

    img = cv2.flip(img, 1)  # Flip video image horizontally
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]  # Extract the face region
        face_roi = cv2.resize(face_roi, (128, 128))  # Resize to match the model input size
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)  # Convert to 3 channels
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict the person using the trained model
        predictions = model.predict(face_roi)
        predicted_class = np.argmax(predictions)

        # Get the name of the recognized person
        person_name = class_names.get(predicted_class, "Unknown")

        # Display the recognized person's name on the image
        cv2.putText(img, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('Face Recognition', img)

    k = cv2.waitKey(1)
    if k == 27:  # Press 'ESC' to exit video
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
