import time
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d
from load_ground_truth import ground_loader;
from evaluator import evaluate_performance_metric;


def harris_corner_detector(image, k=0.04, threshold=0.01):
    # Convert the image to grayscale if it's not already
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)

    # Compute image gradients using Sobel operators
    Ix = convolve2d(image, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], mode='same', boundary='symm')
    Iy = convolve2d(image, [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], mode='same', boundary='symm')

    # Compute products of derivatives
    Ixx = Ix * Ix
    Ixy = Ix * Iy
    Iyy = Iy * Iy

    # Gaussian smoothing of the products of derivatives
    kernel_size = 5
    sigma = 1
    Ixx = convolve2d(Ixx, np.outer(np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * sigma**2)), 
                                   np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * sigma**2))), mode='same', boundary='symm')
    Ixy = convolve2d(Ixy, np.outer(np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * sigma**2)), 
                                   np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * sigma**2))), mode='same', boundary='symm')
    Iyy = convolve2d(Iyy, np.outer(np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * sigma**2)), 
                                   np.exp(-np.arange(-kernel_size, kernel_size+1)**2 / (2 * sigma**2))), mode='same', boundary='symm')

    # Harris corner response function
    det_M = Ixx * Iyy - Ixy**2
    trace_M = Ixx + Iyy
    R = det_M - k * (trace_M**2)

    # Find corners above the threshold
    corners = np.argwhere(R > threshold * R.max())

    # Sort the corners by Harris response
    sorted_corners = corners[np.argsort(R[corners[:, 0], corners[:, 1]])[::-1]]

    with open('harris_points.txt', 'w') as f:
        for keypoint in sorted_corners:
            f.write(f"{keypoint[0]} {keypoint[1]}\n")

    # Get the top 250 and top 500 corners
    top_250_corners = sorted_corners[:250]
    top_500_corners = sorted_corners[:500]

    # Create an image for visualization
    response_map = np.zeros_like(image)
    response_map[corners[:, 0], corners[:, 1]] = 255

    return response_map, top_250_corners, top_500_corners


image_path = os.path.join(os.getcwd(), "test_image.jpg")
image = plt.imread(image_path)


start_time = time.time()
response_map, top_250_corners, top_500_corners = harris_corner_detector(image)
end_time = time.time()

execution_time = end_time - start_time

print(f"Execution Time: {execution_time} seconds" )


plt.subplot(131)
plt.imshow(response_map, cmap='gray')
plt.title('Harris Response Map')

plt.subplot(132)
plt.imshow(image, cmap='gray')
plt.scatter(top_250_corners[:, 1], top_250_corners[:, 0], c='r', s=5)
plt.title('Strongest 250 Corners')


plt.subplot(133)
plt.imshow(image, cmap='gray')
plt.scatter(top_500_corners[:, 1], top_500_corners[:, 0], c='r', s=5)
plt.title('Strongest 500 Corners')
plt.show()




