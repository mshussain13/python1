
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from pathlib import Path

video_path = '/home/shussain/camVId.mkv'
output_folder = '/home/shussain/Downloads/out/'


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.eval()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

cap = cv2.VideoCapture(video_path)
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Process each frame
frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = model(image, size=640)

    person_class_index = 0
    person_predictions = results.pred[0][results.pred[0][:, -1] == person_class_index]

    # Check if person class is present and no other classes
    if len(person_predictions) > 0 and len(person_predictions) == results.pred[0].shape[0]:
        
        frame_output_path = f'{output_folder}/frame_{frame_num}.jpg'
        cv2.imwrite(frame_output_path, frame)
        frame_num += 1

cap.release()

