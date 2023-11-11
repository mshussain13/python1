'''
import cv2
import os

video_path = "/home/shussain/Downloads/Fire_Demo/fire_ms_video/B2_vid2.mp4"
output_folder = "/home/shussain/Downloads/Fire_Demo/frame_b2_2"
frame_interval = 25

def vidFrame(video_path, output_folder, frame_interval):
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    extracted_frames = 0

    while True:
        
        ret, frame = cap.read()
        resize = cv2.resize(frame, (640, 480))
        width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #print('shape---------', width, height)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #print('fps', fps)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #print('total frame', total_frames)
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"B2_vid2_{extracted_frames}.jpg")
            cv2.imwrite(image_path, resize)
            extracted_frames += 1
        frame_count += 1

    cap.release()
    #print(f"Extracted {extracted_frames} frames from the video.")
vidFrame(video_path, output_folder, frame_interval)

'''
# ------------------- make video from frame -------------------------------
'''
import cv2
import os

image_folder = '/home/shussain/Downloads/Fire_Demo/frame_b2_1'
video_name = 'B2_res.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
'''

import cv2
import os

input_vid = ['/Fire_Demo/fire_ms_video/B2_vid1.mp4','2.mp4',']
            
output_video = 'fire_merged.mp4'
output_frame_size = (640, 480)
output_fps = 20

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Choose the codec (e.g., 'mp4v', 'XVID')
out = cv2.VideoWriter(output_video, fourcc, output_fps, output_frame_size)

for video_file in input_vid:
    video_capture = cv2.VideoCapture(video_file)

    while True:
        ret, frame = video_capture.read()

        if not ret:
            break

        frame = cv2.resize(frame, output_frame_size)
        out.write(frame)

    video_capture.release()
out.release()

