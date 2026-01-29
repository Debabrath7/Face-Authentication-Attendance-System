Face Authentication Attendance System

This repository contains a simple Face Authentication based Attendance System developed as part of an AI/ML Intern assignment. The project demonstrates how classical computer vision techniques can be used to build a basic real-time face recognition system.

The main focus of this project is clarity, simplicity, and explainability rather than building a production-ready system.

Project Overview

The system uses a webcam to capture face images, trains a face recognition model using LBPH, and marks attendance through punch-in and punch-out actions. The project is modular and each stage is implemented as a separate Python script.

Technologies Used

Python

OpenCV

Haar Cascade Classifiers

LBPH Face Recognizer

CSV file handling

How the System Works
1. Face Registration

Captures multiple face samples using a webcam

Stores face images locally for each user

Helps improve recognition accuracy

2. Model Training

Trains a Local Binary Pattern Histogram (LBPH) model

LBPH is lightweight and suitable for small datasets

The trained model is saved locally during runtime

3. Attendance System

Performs real-time face detection and recognition

Allows users to mark attendance using keyboard input

Attendance is stored in a CSV file

Attendance file is automatically created when the system is run

Spoof Prevention

A basic spoof prevention mechanism is implemented using eye detection. If eyes are not detected, the face is ignored. This helps reduce simple photo-based spoofing attempts.

Accuracy and Performance

Works well under normal lighting conditions

Accuracy decreases in low-light or extreme face angles

Performance depends on camera quality

Known Limitations

Does not handle advanced spoofing techniques

Manual punch-in and punch-out is required

Not designed for large-scale deployment

Conclusion

This project was developed within a limited timeframe to demonstrate the fundamentals of face recognition using computer vision. The focus was on building a working prototype that is easy to understand and explain.
