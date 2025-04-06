import cv2
import numpy as np
import requests

# Function to download a file from a URL
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as file:
        file.write(response.content)

# Download YOLOv3 files if they don't exist
yolo_weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
yolo_cfg_url = 'https://github.com/pjreddie/darknet/raw/master/cfg/yolov3.cfg'
coco_names_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

# Define file paths
weights_file = "yolov3.weights"
cfg_file = "yolov3.cfg"
names_file = "coco.names"

# Download files if not already present
for url, filename in [(yolo_weights_url, weights_file), 
                       (yolo_cfg_url, cfg_file), 
                       (coco_names_url, names_file)]:
    try:
        with open(filename, 'rb') as f:
            print(f"{filename} already exists. Skipping download.")
    except FileNotFoundError:
        print(f"Downloading {filename}...")
        download_file(url, filename)

# Load YOLO pre-trained model and configuration
yolo_net = cv2.dnn.readNet(weights_file, cfg_file)
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Initialize Video Capture (use 0 for webcam or your video file path)
cap = cv2.VideoCapture('soccer_game_video.mp4')

# Process each frame in the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare frame for YOLO (resize and normalize)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    # Initialize lists for bounding boxes and confidences
    boxes = []
    confidences = []
    class_ids = []

    # Process detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Set confidence threshold for detecting players

                # Get the bounding box coordinates
                center_x = int(obj[0] * frame.shape[1])
                center_y = int(obj[1] * frame.shape[0])
                width = int(obj[2] * frame.shape[1])
                height = int(obj[3] * frame.shape[0])

                # Rectangle coordinates (x, y)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                # Append the box coordinates and confidence values
                boxes.append([x, y, width, height])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply Non-Maximum Suppression (NMS) to avoid multiple boxes for the same player
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes for players
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Ball detection (color detection based on green ball color)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours to find the ball
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small contours
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Draw the ball's bounding box

    # Display the frame
    cv2.imshow("Soccer Game Player and Ball Detection", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
