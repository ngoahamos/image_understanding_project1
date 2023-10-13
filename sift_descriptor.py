import cv2
import os

# Load an image
image_path = os.path.join(os.getcwd(), "test_image.jpg")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# 'keypoints' contains the detected keypoints
# 'descriptors' contains the computed SIFT descriptors

# You can access individual keypoints and descriptors like this:
for i, keypoint in enumerate(keypoints):
    x, y = keypoint.pt
    size = keypoint.size
    angle = keypoint.angle
    response = keypoint.response
    octave = keypoint.octave
    descriptor = descriptors[i]

    # You can process or use keypoints and descriptors as needed
    print(descriptor)