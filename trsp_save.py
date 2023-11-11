import cv2

rtsp_url = "rtsp://root:admin123@192.168.128.201:554/axis-media/media.amp?videocodec=h264"

output_file = "/home/shussain/Downloads/video_1207.mp4"
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Failed to open the RTSP stream.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break
    out.write(frame)

    cv2.imshow("RTSP Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

