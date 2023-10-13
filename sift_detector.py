import time

import cv2
import os
import numpy as np

# Load your image here
ground_truth = 'ground_truth.txt'
image_path = os.path.join(os.getcwd(), "test_image.jpg")
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


# Create an SIFT detector
start_time = time.time()
sift = cv2.SIFT_create()

# Detect keypoints and compute descriptors
keypoints, _ = sift.detectAndCompute(image, None)

end_time = time.time()

execution_time = end_time - start_time

print(f"Execution Time: {execution_time} seconds" )

# Convert keypoints to a list and sort by response
keypoints = sorted(keypoints, key=lambda x: x.response, reverse=True)

# Get the top 500 keypoints
top_500_keypoints = keypoints[:500]

# Create an image with keypoints drawn as circles
result_image = cv2.drawKeypoints(image, top_500_keypoints, None)

# Create a response map as a grayscale image
response_map = np.zeros_like(image, dtype=np.float32)

# Set pixel values in the response map based on keypoints' responses
for kp in keypoints:
    x, y = int(kp.pt[0]), int(kp.pt[1])
    response_map[y, x] = kp.response

with open(ground_truth, 'w') as f:
    for keypoint in keypoints:
        x,y = keypoint.pt
        f.write(f"{x} {y}\n")


# Normalize the response map for visualization
response_map = cv2.normalize(response_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the response map and the image with the strongest keypoints
height,width = image.shape;
cv2.namedWindow('SIFT Response Map', cv2.WINDOW_NORMAL)
cv2.resizeWindow('SIFT Response Map', width, height)
cv2.imshow('SIFT Response Map', response_map, )
cv2.imshow('SIFT Keypoints', result_image)
# cv2.imwrite('strongest_250.png', result_image)

cv2.waitKey(0)
cv2.destroyAllWindows()