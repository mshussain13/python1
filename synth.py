
'''
from PIL import Image, ImageDraw, ImageFont
import random

# Define the dimensions of the license plate image
plate_width = 400
plate_height = 200
background_color = (255, 255, 255)  # White

# Define the font properties
font_path = "/home/shussain/Downloads/font/e3t5euGtX-Co5MNzeAOqinEYo23yqtxI6oYtBA.ttf"
font_size = 100
font_color = (0, 0, 0)  # Black

# Generate a random license plate number
def generate_license_plate():
    # Replace this with your own logic to generate random license plate numbers
    letters = [chr(random.randint(65, 90)) for _ in range(3)]  # Random uppercase letters
    digits = [str(random.randint(0, 9)) for _ in range(4)]  # Random digits
    license_plate = ''.join(letters + digits)  # Combine the letters and digits
    return license_plate

# Create a blank license plate image
plate_image = Image.new('RGB', (plate_width, plate_height), background_color)
draw = ImageDraw.Draw(plate_image)

# Load the font
font = ImageFont.truetype(font_path, font_size)

# Generate a license plate number
license_plate_number = generate_license_plate()

# Calculate the position to center the license plate number on the image
text_width, text_height = draw.textsize(license_plate_number, font=font)
text_x = (plate_width - text_width) // 2
text_y = (plate_height - text_height) // 2

# Draw the license plate number on the image
draw.text((text_x, text_y), license_plate_number, font=font, fill=font_color)

# Save the license plate image
plate_image.save("synthetic_plate.png")
'''
#----------------------------------------------------------------------------------

import cv2
import numpy as np
import random
import string

# Define the dimensions of the license plate image
plate_width = 500
plate_height = 150

# Define the font and size
font_path = "/home/shussain/Downloads/Yq6I-LyHWTfz9rGoqDaUbHvhkAUsSZECy9CY94XsnPc.ttf"  # Replace with the path to your desired font file
font_size = 60

# Generate a random Indian license plate string
def generate_license_plate():
    state_code = random.choice(["KA", "MH", "DL", "TN", "GJ", "CH"])  # Random state code
    numbers = ''.join(random.choices(string.digits, k=2))  # Random 2-digit number
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))  # Random 2-letter combinationi
    suffix = ''.join(random.choices(string.digits, k=4))  # Random 4-digit suffix
    license_plate = f"{state_code}{numbers} {letters} {suffix}"  # Concatenate the components
    return license_plate

# Create a blank license plate image
license_plate_image = np.ones((plate_height, plate_width, 3), dtype=np.uint8) * 255  # White background

# Generate a license plate string
license_plate_text = generate_license_plate()

# Load the font
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.freetype.initFont(cv2.freetype.FONTSTYLE_NORMAL, font_size)
font_height = cv2.freetype.getFontData(font)[0]

# Calculate the position to align the text on the image
text_width, _ = cv2.getTextSize(license_plate_text, font, font_size, font_height)
text_x = (plate_width - text_width[0]) // 2
text_y = (plate_height + font_height) // 2

# Put the license plate text on the image
cv2.freetype.putText(license_plate_image, license_plate_text, (text_x, text_y), font, font_size, (0, 0, 0), thickness=-1, line_type=cv2.LINE_AA)

# Add state-specific patterns or designs
# You can implement additional logic here to apply specific patterns or designs based on the state code

# Display the license plate image
cv2.imshow("License Plate", license_plate_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
import cv2
import numpy as np
import random
import string
import os

# Define the dimensions of the number plate image
plate_width = 400
plate_height = 100

# Generate a random Indian number plate string
def generate_number_plate():
    prefix = random.choice(["KA", "MH", "DL", "TN", "GJ", "BR", "JK"])  # Random state code
    numbers = ''.join(random.choices(string.digits, k=2))  # Random 2-digit number
    letters = ''.join(random.choices(string.ascii_uppercase, k=2))  # Random 2-letter combination
    suffix = random.choice([''.join(random.choices(string.digits, k=4))])  # Random 4-digit suffix or blank
    plate_text = f"{prefix}{numbers} {letters} {suffix}"  # Concatenate the components
    return plate_text

# Create a folder to save the generated images
output_folder = "generated_images"
os.makedirs(output_folder, exist_ok=True)

# Specify the number of images to generate
num_images = 10

# Generate and save the synthetic number plate images
for i in range(num_images):
    # Create a blank number plate image
    plate_image = np.ones((plate_height, plate_width, 3), dtype=np.uint8) * 255  # White background

    # Generate a number plate text
    plate_text = generate_number_plate()

    # Set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5

    # Calculate the size of the text
    (_, text_height), _ = cv2.getTextSize(plate_text, font, font_scale, font_thickness)

    # Calculate the position to center the text on the image
    text_x = plate_width
    text_y = plate_height

    # Put the number plate text on the image
    cv2.putText(plate_image, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    # Generate a filename for the image
    filename = f"plate_{i+1}.jpg"
    file_path = os.path.join(output_folder, filename)

    # Save the image
    cv2.imwrite(file_path, plate_image)

    print(f"Generated and saved {filename}")

print("All images generated and saved.")
'''
