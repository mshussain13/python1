'''
import cv2
from ultralytics import YOLO

model = YOLO('/home/shussain/Downloads/yolo_model/yolov8n.pt')

#video_path = "/home/shussain/cam1.mp4"
cap = cv2.VideoCapture(0)

while cap.isOpened():
    
    success, frame = cap.read()
    if success:
        
        results = model.track(frame, persist=True)
        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

'''

import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path


weights_path = "/home/shussain/Downloads/yolo_model/yolov5/25_sep_best.pt"

confidence_threshold = 0.6

model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

video_source = 1
#cap = cv2.VideoCapture(video_path)
root = tk.Tk()
root.title("Fire and Smoke Detection")

# Function to detect fire and smoke
def detect_fire_smoke(video_source):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.detect(frame, imgsz=640)

        alert = False
        for result in results.pred[0]:
            class_name = model.names[int(result[-1])]
            confidence = result[4]

            if class_name in ['Fire', 'Smoke'] and confidence >= 0.5:
                alert = True
                break

        # Display the frame in a Tkinter window
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(image))
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.photo = photo

        # Show an alert if fire or smoke is detected
        if alert:
            messagebox.showwarning("Alert", "Fire or Smoke Detected!")

    cap.release()

canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

start_button = tk.Button(root, text="Start Detection", command=lambda: detect_fire_smoke(1))
start_button.pack()

# Create a button to open a video file
def open_file():
    file_path = tk.filedialog.askopenfilename()
    detect_fire_smoke(file_path)
    
open_button = tk.Button(root, text="Open Video File", command=open_file)
open_button.pack()

# Run the Tkinter main loop
root.mainloop()


