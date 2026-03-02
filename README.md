# Real-Time Face Recognition Attendance System

##  Overview
A real-time face recognition based attendance system built using OpenCV and the LBPH algorithm.

## Features
- Real-time face detection using Haarcascade
- Face recognition using LBPH
- Confidence score display
- Automated attendance logging (CSV)
- Duplicate prevention during runtime

## Tech Stack
- Python
- OpenCV
- NumPy

## Project Structure
dataset/
live_recognition.py
recognize.py
trainer.yml (ignored)
attendance.csv (generated)

## How It Works
1. Dataset organized by person name.
2. Faces are converted to grayscale.
3. LBPH extracts local texture features.
4. System predicts closest match using distance metric.
5. Attendance saved automatically.

##  Run Instructions
pip install opencv-contrib-python numpy  
python recognize.py  
python live_recognition.py  

##  Future Improvements
- Deep learning based face embeddings
- GUI interface
- Database integration
- Cloud deployment
