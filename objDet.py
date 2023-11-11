from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
model = YOLO('/home/shussain/Downloads/yolov8n.pt')

classes = ['person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
           'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie''suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove',
           'skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
           'donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
           'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
while True:
    success, img = cap.read()
    results = model(img, stream= True)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1,y1,x2,y2)
            print('box-----------', box)
            
            cv2.rectangle(img,(x1,y1), (x2,y2), (255,0,255), 3)
            
            conf = math.ceil((box.conf[0]* 100))/100
            print(conf)
            try:
                
                cls = int(box.cls[0])
                print('class-------',cls)
                # cropping image
                
                cv2.putText(img, f'{classes[cls]} { conf}', (max(0,x1), max(35,y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 2)
            except:
                print('detection successfuly')
    crop = img[x1:y1,x2:y2]
    #cv2.imshow('Crop',crop)
    cv2.imshow('image', img)
    if cv2.waitKey(1) == 27:
        break
    


#results = model('/home/shussain/Downloads/bs.mp4', show = True)



cv2.destroyAllWindows
