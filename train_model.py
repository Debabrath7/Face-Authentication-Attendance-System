import cv2
import os
import numpy as np

dataset_path = "faces"
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

print("Training model")

for user_id in os.listdir(dataset_path):
    user_folder = os.path.join(dataset_path, user_id)

    if not os.path.isdir(user_folder):
        continue

    for image_name in os.listdir(user_folder):
        image_path = os.path.join(user_folder, image_name)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        faces.append(img)
        labels.append(int(user_id))

recognizer.train(faces, np.array(labels))
recognizer.save("model.yml")

print("Model training completed and saved as model.yml")
