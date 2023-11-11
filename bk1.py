import os
import cv2
import shutil

# Define the folders for different colors
color_folders = {
    'black': "black",
    'blue': "blue",
    'brown': "brown",
    'green': "green",
    'pink': "pink",
    'red': "red",
    'silver': "silver",
    'white': "white",
    'yellow': "yellow",
    'unknown':"unknown",
    'gray':"gray",
    'multi':"multi"
}
# Function to move the image to the corresponding folder based on the key pressed
def move_image_to_folder(image_path, color):
    destination_folder = color_folders.get(color.lower())

    if destination_folder is None:
        print("Invalid color. Image not moved.")
        return

    # Check if the destination folder exists, otherwise create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Move the image to the respective folder
    shutil.move(image_path, os.path.join(destination_folder, os.path.basename(image_path)))
    print("Image moved successfully to", destination_folder)

# Replace 'source_folder' with the path to the folder containing your images
source_folder = '/home/shussain/Downloads/Car_Dataset/All_Car'

# List all files in the source folder
image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

for image_file in image_files:
    image_path = os.path.join(source_folder, image_file)
    try:
        img = cv2.imread(image_path)
        cv2.imshow("Image", img)
        key_pressed = cv2.waitKey(0)
        color = ''
        if key_pressed == ord('r'):
            color = 'red'
        elif key_pressed == ord('b'):
            color = 'blue'
        elif key_pressed == ord('g'):
            color = 'green'
        elif key_pressed == ord('p'):
            color = 'pink'
        elif key_pressed == ord('w'):
            color = 'white'
        elif key_pressed == ord('y'):
            color = 'yellow'
        elif key_pressed == ord('k'):
            color = 'black'
        elif key_pressed == ord('s'):
            color = 'silver'
        elif key_pressed == ord('n'):
            color = 'brown'
        elif key_pressed == ord('u'):
            color = 'unknown'
        elif key_pressed == ord('a'):
            color = 'gray'
        elif key_pressed == ord('m'):
            color = 'multi'
        move_image_to_folder(image_path, color)
        cv2.destroyAllWindows()  # Close the image window
    except FileNotFoundError:
        print(f"Image file {image_file} not found.")
    except Exception as e:
        print(f"An error occurred for {image_file}:", e)


