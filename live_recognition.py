import cv2
import os
import numpy as np
from datetime import datetime

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

data_path = 'dataset'
people = os.listdir(data_path)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked_names = set()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (200, 200))

        label, confidence = recognizer.predict(face)

        if confidence < 80:
            name = people[label]
            color = (0, 255, 0)
            confidence_text = f"{round(100 - confidence)}%"

            if name not in marked_names:
                with open("attendance.csv", "a") as f:
                    now = datetime.now()
                    time_string = now.strftime("%H:%M:%S")
                    f.write(f"{name},{time_string}\n")
                marked_names.add(name)

        else:
            name = "Unknown"
            color = (0, 0, 255)
            confidence_text = ""

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{name} {confidence_text}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()