import cv2
import numpy as np

def augment_image(image):
    # Randomly flip the image horizontally
    #if np.random.rand() < 0.5:
        #image = cv2.flip(image, 1)

    # Randomly flip the image vertically
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 0)

    # Randomly rotate the image by a certain angle
    angle = np.random.randint(-10, 10)
    M = cv2.getRotationMatrix2D((image.shape[1] // 2, image.shape[0] // 2), angle, 1)
    image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # Randomly resize the image
    scale_factor = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    # Randomly change the brightness and contrast of the image
    brightness = np.random.randint(-30, 30)
    contrast = np.random.uniform(0.8, 1.2)
    image = cv2.addWeighted(image, contrast, np.zeros_like(image), 0, brightness)

    return image

# Load an image
image = cv2.imread('/home/shussain/Downloads/Nadia.jpg')

# Perform data augmentation
augmented_image = augment_image(image)

# Display the original and augmented images side by side
cv2.imshow('Original Image', image)
cv2.imshow('Augmented Image', augmented_image)


cv2.waitKey(0)
cv2.destroyAllWindows()

