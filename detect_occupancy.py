# detect_occupancy.py (Corrected Version)

import cv2
import numpy as np

# --- Constants ---
VIDEO_PATH = 'bus_video.mp4'
CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.5
OCCUPANCY_LEVELS = {
    'Low': (0, 10),
    'Medium': (11, 20),
    'Full': (21, 100)
}

# --- Load YOLOv4 Model ---
print("Loading YOLOv4 model...")
net = cv2.dnn.readNet("yolo_files/yolov4.weights", "yolo_files/yolov4.cfg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("yolo_files/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
PERSON_CLASS_ID = classes.index('person')

# --- Process Video ---
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

print("Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # --- Object Detection ---
    # CORRECTED THIS LINE: cv. -> cv2.
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # --- Post-processing Detections ---
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if class_id == PERSON_CLASS_ID and confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # CORRECTED THIS LINE: cv. -> cv2.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    person_count = len(indexes)

    # --- Determine Occupancy Level ---
    occupancy = 'Unknown'
    for level, (min_val, max_val) in OCCUPANCY_LEVELS.items():
        if min_val <= person_count <= max_val:
            occupancy = level
            break

    # --- Draw Bounding Boxes and Display Info ---
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    info_text = f"Passenger Count: {person_count} | Occupancy: {occupancy}"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Bus Occupancy Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
print("Processing finished.")