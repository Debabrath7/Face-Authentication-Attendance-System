import cv2
import csv
import os
from datetime import datetime

# File name
attendance_file = "attendance.csv"

# Create attendance file if it doesn't exist
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["UserID", "Status", "Timestamp"])

# Load trained model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

camera = cv2.VideoCapture(0)
attendance_marked = {}

def mark_attendance(user_id, status):
    time_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_file, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([user_id, status, time_stamp])

print("Press 'i' for Punch-In | 'o' for Punch-Out | ESC to exit")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        user_id, confidence = recognizer.predict(face_img)

        # Basic spoof prevention (eye detection)
        eyes = eye_cascade.detectMultiScale(face_img)
        if len(eyes) == 0:
            cv2.putText(frame, "No eyes detected", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            continue

        if confidence < 70:
            cv2.putText(frame, f"User {user_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            key = cv2.waitKey(1)

            if key == ord('i') and user_id not in attendance_marked:
                mark_attendance(user_id, "Punch-In")
                attendance_marked[user_id] = "IN"

            if key == ord('o') and attendance_marked.get(user_id) == "IN":
                mark_attendance(user_id, "Punch-Out")
                attendance_marked[user_id] = "OUT"
        else:
            cv2.putText(frame, "Unknown Face", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
