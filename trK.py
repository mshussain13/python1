import os
import cv2
import time
from deep_sort_realtime.deepsort_tracker import DeepSort

cap = cv2.VideoCapture('/home/shussain/Downloads/yolo_model/res/out.mp4')

#Initialise the object tracker class
object_tracker = DeepSort()

while cap.isOpened():
    success, img = cap.read()

    start = time.perf_counter()

    results = detector.score_frame(img)
    img,detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)

    tracks = object_tracker.update_tracks(detections, frame=img) 
    # NOTE: Bounding box expects to be a list of detections, each in tuples of ([left, top, w, h], confidence, detection class)
    
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()

        bbox = ltrb

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
        cv2.putText(img, "ID: " + str(track_id), (int(bbox[0]),int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

    end = time.perf_counter()
    totalTime = end-start
    fps = 1/totalTime

    cv2.putText(img, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()

cv2.destroyAllWindows()
