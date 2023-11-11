'''
import cv2
import numpy as np
import sys
sys.path.append('/home/shussain/Downloads/yolo_model/sort')  # Replace with the actual path to the SORT directory
from sort import Sort


# Load your YOLOv5 model
net = cv2.dnn.readNet('/home/shussain/Downloads/yolo_model/yolov5x.pt')  # Replace with your model's path
classes = ['vehicle']  # Modify with your vehicle class label

# Initialize SORT tracker
tracker = Sort()

# Open video file for reading
cap = cv2.VideoCapture('/home/shussain/Downloads/yolo_model/res/out.mp4')  # Replace with your video file

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height), True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()

    tracked_objects = tracker.update(detections)

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        label = f'Track {int(track_id)}'
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

'''

import cv2
import numpy as np
import torch
from sort import Sort

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

# Set device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Initialize SORT tracker
tracker = Sort()

video_path = '/home/shussain/Downloads/yolo_model/res/out.mp4'
cap = cv2.VideoCapture(video_path)

# Output video parameters
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f'/home/shussain/Downloads/yolo_model/res/res.mp4'
out = cv2.VideoWriter(output_path, output_fourcc, output_fps, (output_width, output_height))
classes = ['car', 'truck', 'motorcycle', 'bus']

# Optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Function to compute optical flow and determine wrong-side movement
def is_wrong_moving(prev_frame, x1, y1, x2, y2):
    # Convert bounding box coordinates to int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Select the region of interest (ROI) within the bounding box
    roi_prev = prev_frame[y1:y2, x1:x2]
    roi_next = frame[y1:y2, x1:x2]

    # Calculate optical flow using Lucas-Kanade
    flow = cv2.calcOpticalFlowPyrLK(
        roi_prev, roi_next, None, None, **lk_params
    )

    # Calculate the mean optical flow in the x-direction
    flow_mean_x = np.mean(flow[..., 0])

    # Define a threshold for detecting wrong-side movement
    movement_threshold = 2.0  # Adjust as needed

    # Check if the mean optical flow in the x-direction is below the threshold
    if flow_mean_x < -movement_threshold:
        return True  # Wrong-side movement
    else:
        return False  # Correct movement

prev_frame = None

while cap.isOpened:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects using YOLOv5
    detections = model(frame)
    vehicle_detections = [d for d in detections.pred[0] if d[5] in [classes.index('car'), classes.index('truck')]]

    # Track vehicles using SORT
    trackers = tracker.update(np.array(vehicle_detections))
    for track in trackers:
        x1, y1, x2, y2, track_id = track

        if prev_frame is not None and is_wrong_moving(prev_frame, x1, y1, x2, y2):
            color = (0, 0, 255)  # Red for wrong moving
        else:
            color = (0, 255, 0)  # Green for correct moving

        # Draw bounding box with ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, str(int(track_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        prev_frame = frame.copy()

cap.release()
out.release()

