'''
import cv2
import glob
import os

folder_name = os.makedirs('frame1', exist_ok= True)

vid_cap = cv2.VideoCapture('/home/shussain/Downloads/vid_new/37948765.mp4')

length = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('total frames ', length )

success,image = vid_cap.read()
print(image.shape)

count = 0
success = True

while success:
    if count%30 == 1:
        
        cv2.imwrite('frame1/xframe%d.jpeg' % count,image)
    count += 1
'''
''' 
import cv2
import numpy as np
import os

# set video file path of input video with name and extension
vid = cv2.VideoCapture('/home/shussain/Downloads/vid_new/37948765.mp4')


if not os.path.exists('images'):
    os.makedirs('images')

#for frame identity
index = 0
while(True):
    # Extract images
    ret, frame = vid.read()
    # end of frames
    if not ret: 
        break
    # Saves images
    name = './images/frame' + str(index) + '.jpg'
    print ('Creating...' + name)
    cv2.imwrite(name, frame)

    # next frame
    index +=1
'''
'''
#---------------------------------------file read and modify---------------------------------------------------    
import os
import sys

files = os.listdir(".")


for file in files:
    if ".txt" in file:
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line[0] == "1":
                    line = line.replace("1", "0")
                    with open(file, "w") as f:
                        f.write(line)
                    print(line)
                else:
                    print(line)
                    continue
    else:
        continue
        
'''

import cv2
import os

def video_to_images(video_path, output_folder, frame_interval):
    
    video = cv2.VideoCapture(video_path)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0
    extracted_frames = 0

    while True:
        # Read the next frame from the video
        ret, frame = video.read()

        if not ret:
            # Break the loop if there are no more frames
            break

        if frame_count % frame_interval == 0:
            # Save the frame as an image
            image_path = os.path.join(output_folder, f"frame_{extracted_frames}.jpg")
            cv2.imwrite(image_path, frame)
            extracted_frames += 1

        frame_count += 1

    # Release the video file
    video.release()

    print(f"Extracted {extracted_frames} frames from the video.")

# Example usage
video_path = "/home/shussain/Downloads/cr/vid_04/video23-07-20_10-19-16-28.mkv"
output_folder = "/home/shussain/Downloads/cr/f"
frame_interval = 25

video_to_images(video_path, output_folder, frame_interval)

