'''

import cv2
import shutil


video_path = '/home/shussain/Downloads/yolo_model/vid/capture_1.mp4'
cap = cv2.VideoCapture(video_path)

# Output video parameters
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f'/home/shussain/Downloads/yolo_model/res/out.mp4'
out = cv2.VideoWriter(output_path, output_fourcc, output_fps, (1280, 720))

x1 = 644
y1 =122
x2 =644+1205
y2 = 122+720

while cap.isOpened:
    ret, frame = cap.read()
    print('frame--------->',frame.shape)
    
    img = frame[y1:y2,x1:x2]
    img = cv2.resize(img,(1280,720))
    print('img------->',img.shape)
    
    out.write(img)
    if not ret:
        break
        
cap.release()
out.release()

'''
'''
import cv2
import numpy as np
import torch
from sort import *
#from sort import Sort
from PIL import Image


model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

video_path = '/home/shussain/Downloads/yolo_model/res/out.mp4'
cap = cv2.VideoCapture(video_path)

tracker = Sort()

# Output video parameters
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f'/home/shussain/Downloads/yolo_model/res/res.mp4'
out = cv2.VideoWriter(output_path, output_fourcc, output_fps, (output_width, output_height))
classes = ['car', 'truck', 'motorcycle', 'bus']

# Initialize variables for tracking
prev_frame = None

while cap.isOpened:
    ret, frame = cap.read()
    
    if not ret:
        break

    
    detections = model(frame)
    vehicle_detections = [d for d in detections.pred[0] if d[5] in [classes.index('car'), classes.index('truck'),classes.index('motorcycle'),classes.index('bus')]]

    trackers = tracker.update(np.array(vehicle_detections))
    
    for track in trackers:
        x1, y1, x2, y2, track_id = track

        
        if prev_frame is None:
            prev_frame = np.array([x1, y1, x2, y2])
            continue

        # Calculate the center of the current bounding box
        current_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Calculate the center of the previous bounding box
        prev_center = np.array([(prev_frame[0] + prev_frame[2]) / 2, (prev_frame[1] + prev_frame[3]) / 2])

        # Determine the direction based on the x-coordinate change
        if current_center[0] > prev_center[0]:
            # Right to left, consider it wrong direction
            color = (0, 0, 255)
        else:
            # Left to right, consider it right direction
            color = (0, 255, 0)

        
        prev_frame = np.array([x1, y1, x2, y2])

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, str(int(track_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

cap.release()
out.release()

'''
import cv2
import numpy as np
import torch
#from deepsort import DeepSort
from PIL import Image
from sort.tracker import SortTracker
#from deepsort.tracker import DeepSortTracker

#tracker = DeepSortTracker(args)
'''
for image in images:
   dets = detector(image)
   online_targets = tracker.update(dets)
'''

model = torch.hub.load('ultralytics/yolov5', 'yolov5x')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

video_path = '/home/shussain/Downloads/yolo_model/res/out.mp4'
cap = cv2.VideoCapture(video_path)

#deepsort = DeepSortTracker(model_path='/home/shussain/Downloads/yolo_model/osnet_ibn_x1_0_MSMT17.pth', use_cuda=True)
tracker = SortTracker()
# Output video parameters
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f'/home/shussain/Downloads/yolo_model/res/res2.mp4'
out = cv2.VideoWriter(output_path, output_fourcc, output_fps, (output_width, output_height))
classes = ['car', 'truck', 'motorcycle', 'bus']

# Initialize variables for tracking
prev_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    detections = model(frame)
    vehicle_detections = [d for d in detections.pred[0] if d[5] in [classes.index('car'), classes.index('truck')]]

    # Format detections for deepsort (x, y, w, h, conf, class)
    detections_deepsort = []
    for detection in vehicle_detections:
        x1, y1, x2, y2, conf, class_id = detection
        #detections_deepsort.append([x1, y1, x2 - x1, y2 - y1, conf, class_id])

    trackers = tracker.update(np.array(detections))

    for track in trackers:
        x1, y1, x2, y2, track_id = track

        if prev_frame is None:
            prev_frame = np.array([x1, y1, x2, y2])
            continue

        # Calculate the center of the current bounding box
        current_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        # Calculate the center of the previous bounding box
        prev_center = np.array([(prev_frame[0] + prev_frame[2]) / 2, (prev_frame[1] + prev_frame[3]) / 2])

        # Determine the direction based on the x-coordinate change
        if current_center[0] > prev_center[0]:
            # Right to left, consider it wrong direction
            color = (0, 0, 255)
        else:
            # Left to right, consider it right direction
            color = (0, 255, 0)

        prev_frame = np.array([x1, y1, x2, y2])

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2), color, 2))
        cv2.putText(frame, str(int(track_id)), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)

cap.release()
out.release()


