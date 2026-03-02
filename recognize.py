import cv2
import os
import numpy as np

data_path = 'dataset'
people = os.listdir(data_path)

faces = []
labels = []
label_map = {}

label_id = 0

for person in people:
    label_map[label_id] = person
    person_path = os.path.join(data_path, person)
    
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces.append(gray)
        labels.append(label_id)
    
    label_id += 1

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")

print("Training Completed!")