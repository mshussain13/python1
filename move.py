import os
import shutil

source_folders = ["WB", "UP", "UK","TS", "TR", "TN", "SK", "RJ", "PY","PB","OD","NL","MZ","MP","MN","ML","LA","KA","KL"]  # Replace with your source folders
destination_folder = "new_folder"  # Replace with your destination folder

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Iterate through each source folder
for folder in source_folders:
    # Iterate through files in the source folder
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            source_path = os.path.join(folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            # Move the JPG file to the destination folder
            shutil.move(source_path, destination_path)

