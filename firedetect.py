import cv2
import cv2
import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

def detect_fire_smoke(model_path, conf_thres=0.5, iou_thres=0.4):
    device = select_device('')
    model = attempt_load(model_path)
    img_size = model.stride.max()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = torch.from_numpy(frame[:, :, ::-1]).to(device).float() / 255.0
        img_copy = np.copy(img)
        #img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model(img_copy, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        for det in pred[0]:
            if det is not None:
                x1, y1, x2, y2, conf, cls = det
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                if int(cls) == 0:
                    color = (0, 0, 255)
                elif int(cls) == 1:
                    color = (0, 255, 0)
                
                label = f'Class: {int(cls)} | Confidence: {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Webcam Fire/Smoke Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = '/home/shussain/Downloads/yolo_model/yolov5/25_sep_best.pt'
    detect_fire_smoke(model_path)
    
    



