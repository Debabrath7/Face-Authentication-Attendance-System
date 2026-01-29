import cv2
import os

# User details
user_id = input("Enter numeric User ID: ")
user_name = input("Enter User Name: ")

# Directory setup
dataset_path = "faces"
user_path = os.path.join(dataset_path, user_id)

if not os.path.exists(user_path):
    os.makedirs(user_path)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

camera = cv2.VideoCapture(0)
sample_count = 0

print("Starting face registration .. Press ESC to exit")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sample_count += 1
        face_img = gray[y:y+h, x:x+w]
        cv2.imwrite(f"{user_path}/{sample_count}.jpg", face_img)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) == 27 or sample_count >= 30:
        break

camera.release()
cv2.destroyAllWindows()

print("Face registration completed successfully.")
