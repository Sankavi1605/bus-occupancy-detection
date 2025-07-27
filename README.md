
# Real-Time Bus Occupancy Detection 🚌

A computer vision system built with Python and OpenCV that uses the YOLOv4 object detection model to count passengers from a video stream and classify the bus's occupancy level in real-time.

## Features
- Analyzes video streams to detect and count passengers frame-by-frame.
- Utilizes both YOLOv4 and the faster YOLOv4-tiny models.
- Classifies occupancy into levels like "Low", "Medium", and "Full".
- Designed for efficiency and potential deployment on edge devices.

## Technologies Used
- Python
- OpenCV
- NumPy
- YOLOv4 / YOLOv4-tiny

## Setup & How to Run

1. Clone the repository
2. Install dependencies : pip install -r requirements.txt
3. Download YOLOv4 Model Files : mkdir yolo_files
Download the following three files and place them inside the yolo_files folder:
Config: yolov4.cfg
Weights: yolov4.weights
Names: coco.names

4.Add a Sample Video : Place a sample video file named bus_video.mp4 in the main project folder.
5.Run the script : python detect_occupancy.py



