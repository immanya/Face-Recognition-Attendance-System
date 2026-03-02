import face_recognition
import os
import pickle

known_encodings = []
known_names = []

for person in os.listdir("dataset"):
    for image_name in os.listdir(f"dataset/{person}"):
        image = face_recognition.load_image_file(f"dataset/{person}/{image_name}")
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(person)

data = {"encodings": known_encodings, "names": known_names}

with open("encodings.pickle", "wb") as f:
    pickle.dump(data, f)

print("Training Done ✅")