import cv2
import os
import numpy as np

# Load an image
image_path = os.path.join(os.getcwd(), "test_image.jpg")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the size and location of the patch
x, y, width, height = 100, 100, 50, 50
patch = image[y:y+height, x:x+width]

# Calculate the mean and standard deviation of the patch
mean, std_dev = cv2.meanStdDev(patch)

# Normalize the patch by subtracting the mean and dividing by the standard deviation
normalized_patch = (patch - mean) / std_dev

# Optionally, you can reshape the normalized patch into a 1D feature vector
feature_vector = normalized_patch.flatten()

# Print the feature vector
print(feature_vector)