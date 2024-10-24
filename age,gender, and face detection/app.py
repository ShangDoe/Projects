import streamlit as st
import tensorflow as tf 
import cv2 
import numpy as np 
from ultralytics import YOLO


age_model = tf.keras.models.load_model("age_classification.h5")
gender_model = tf.keras.models.load_model("gender_classification.h5")
yolo_model = YOLO("yolov8-face.pt")

age_labels = ["Young", "Middle-Aged", "Old"]
gender_labels = ["Male", "Female"]

def preprocess_image(image):
    image =  cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image 


st.title("Real-time Face Detection with Gender and Age Prediction")

checkbox = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

cam = cv2.VideoCapture(0)

while checkbox:
    ret, frame = cam.read()

    if not ret:
        st.write("Failed to access webcam")
        break

    results =  yolo_model(frame)
    for result in results:
        for box  in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = box.conf[0]

            preprocessed_face = preprocess_image(frame)

            gender_pred = gender_model.predict(preprocessed_face)
            age_pred = age_model.predict(preprocessed_face)

            gender_class = gender_labels[int(np.round(gender_pred[0]))]
            age_class = age_labels[np.argmax(age_pred[0])]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Gender: {gender_class}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Age: {age_class}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

else:
    st.write('Stopped')
    cam.release()