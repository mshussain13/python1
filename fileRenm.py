'''
import os

def rename(img_files):
    files = os.listdir(img_files)
    print(len(files))
    
    counter = 1
    
    for f in files:
        if f.lower().endswith(('.jpg','.jpeg','.png')):
            new = f"cars_{counter}.jpg"
            
            current_path = os.path.join(img_files, f)
            new_path = os.path.join(img_files, new)

            os.rename(current_path, new_path)
            
            counter += 1
            
if __name__ == "__main__":

    img_files = "/home/shussain/Downloads/Car_Dataset/All_Car"
    rename(img_files)
    print('done-----------')
    
'''

import os

# Define the path to the folder containing YOLO label files
folder_path = "/home/shussain/Downloads/Fire_Demo/results/ALL_car/labels"

# List all files in the folder
files = os.listdir(folder_path)

# Iterate through each file
for file_name in files:
    file_path = os.path.join(folder_path, file_name)

    # Read the contents of the file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Modify the class labels
    for i in range(len(lines)):
        parts = lines[i].split()
        if len(parts) > 0:
            class_id = int(parts[0])
            if class_id >= 9:
                # Replace class labels 10 and above with class label 10
                parts[0] = "9"

        # Update the line in the list
        lines[i] = ' '.join(parts)

    # Write the modified contents back to the file
    with open(file_path, 'w') as file:
        file.writelines(lines)

print("Class labels replaced successfully.")

