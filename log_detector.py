import time
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import convolve


def gaussian_kernel(sigma):
    size = int(6 * sigma)
    if size % 2 == 0:
        size += 1
    x = np.arange(-size // 2, size // 2 + 1)
    kernel = np.exp(-x ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def compute_log_response(image, sigma=1.0):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Compute the LoG response
    kernel_size = int(6 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = gaussian_kernel(sigma)
    smoothed_image = convolve(image, np.outer(kernel, kernel))
    laplacian = convolve(smoothed_image, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))

    return laplacian


def find_interest_points(response, threshold):
    points = np.argwhere(response > threshold)
    sorted_points = points[np.argsort(response[points[:, 0], points[:, 1]])[::-1]]
    with open('log_points.txt', 'w') as f:
        for keypoint in sorted_points:
            f.write(f"{keypoint[0]} {keypoint[1]}\n")
    return sorted_points[:250],sorted_points[:500]


# Load your image here
image_path = os.path.join(os.getcwd(), "test_image.jpg")
image = plt.imread(image_path)

# Compute the LoG response
sigma = 2.0
start_time = time.time()
log_response = compute_log_response(image, sigma)
end_time = time.time()

execution_time = end_time - start_time

print(f"Execution Time: {execution_time} seconds" )


# Set threshold and maximum number of interest points
threshold = 0.03

# Find the strongest interest points
less_interest_points,interest_points = find_interest_points(log_response, threshold)


# Display the response map
plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(log_response, cmap='gray')
plt.title('LoG Response Map')

# Display the image with the strongest interest points
plt.subplot(1, 3, 2)
plt.imshow(image, cmap='gray')
plt.scatter(less_interest_points[:, 1], less_interest_points[:, 0], c='r', s=5)
plt.title('Strongest 250 points')

plt.subplot(1, 3, 3)
plt.imshow(image, cmap='gray')
plt.scatter(interest_points[:, 1], interest_points[:, 0], c='r', s=5)
plt.title('Strongest 500 Points')
plt.show()