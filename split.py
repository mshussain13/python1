import os
import cv2
'''
# Input video path
video_path = '/home/shussain/video23-04-10_17-06-03-13.mkv'

# Output directory to save the split videos
output_directory = '/home/shussain/v'

# Size limit for each split (in bytes)
split_size_limit = 1 * 1024 * 1024 * 1024  # 1 GB

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Open the input video file
cap = cv2.VideoCapture(video_path)

# Calculate the total number of frames in the video
total_frames = 0
while cap.isOpened():
    ret, _ = cap.read()
    if not ret:
        break
    total_frames += 1

# Reset the video capture to the beginning
a= cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
print('count-------',a)

# Split the video
start_frame = 0
split_index = 0

while start_frame < total_frames:
    # Set the end frame for the split
    end_frame = min(start_frame + split_size_limit, total_frames)

    # Set the output video path for the split
    output_path = os.path.join(output_directory, f'split_video_{split_index}.mkv')

    # Create a new video writer for the split
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Extract and write the frames to the output video
    for frame_index in range(start_frame, end_frame):
        # Read a frame from the input video
        ret, frame = cap.read()

        # Write the frame to the output video
        out.write(frame)

    # Release the video writer
    out.release()

    # Update the start frame and split index
    start_frame = end_frame
    split_index += 1

# Release the input video capture
cap.release()
'''

import os
import random
import shutil

def split_data(data_dir, train_ratio=0.5):
    image_dir = data_dir
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'train1')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(image_files)
    num_train = int(len(image_files) * train_ratio)

    train_images = image_files[:num_train]
    val_images = image_files[num_train:]

    for image in train_images:
        image_path = os.path.join(image_dir, image)
        label_path = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(label_path):
            shutil.move(image_path, os.path.join(train_dir, image))
            shutil.move(label_path, os.path.join(train_dir, os.path.basename(label_path)))

    for image in val_images:
        image_path = os.path.join(image_dir, image)
        label_path = os.path.splitext(image_path)[0] + '.txt'
        if os.path.exists(label_path):
            shutil.move(image_path, os.path.join(val_dir, image))
            shutil.move(label_path, os.path.join(val_dir, os.path.basename(label_path)))

if __name__ == "__main__":
    data_folder = "/home/shussain/Downloads/backup_anpr/train"
    split_data(data_folder, train_ratio=0.5)
