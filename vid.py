import cv2
import numpy as np
import torch
from PIL import Image

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Set device (CPU or GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device).eval()

# Load video
video_path = '/home/shussain/video23-04-10_17-06-03-13.mkv'
cap = cv2.VideoCapture(video_path)

# Output video parameters
output_fps = cap.get(cv2.CAP_PROP_FPS)
output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = f'/home/shussain/v0.mp4'
out = cv2.VideoWriter(output_path, output_fourcc, output_fps, (output_width, output_height))
classes = ['car','truck','motorcycle','bus']



frame_count = 0
detect_flag = 1
cnt=0
vC=0
try:
    while cap.isOpened:
        ret, frame = cap.read()
        #print("frame",frame.shape)
        if not ret:
            break
        
        if detect_flag:
            #pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = model(frame, size=640)
            #rint('results -->> ',type(results))
            print(results)
            #print('results -->> ',results)
            for result in results.pandas().xyxy[0].itertuples(index=False):
                #_, _, class_id, xmin, ymin, xmax, ymax = result
                if result[-1] in classes:
                    print(result[-1]+" Found. Not detecting for next 180 frame")
                    detect_flag=0
             

        if frame_count<180 and not detect_flag:
           #print('fff--------',frame)
           out.write(frame)
           frame_count+=1
           if cnt==18000 :
            
            output_path = f'/home/shussain/n{vC}.mp4'
            out = cv2.VideoWriter(output_path, output_fourcc, output_fps, (output_width, output_height))
            
            vC+=1
            cnt=0
            else:
            cnt+=1
        else:
           frame_count = 0
           detect_flag = 1
        
        
        
        
    
            
        print(cap.get(cv2.CAP_PROP_POS_FRAMES),frame_count,detect_flag) 
        
except Exception as e:
    print(e)
finally:
    print("release")
    cap.release()
    out.release()
    #cv2.destroyAllWindows()

