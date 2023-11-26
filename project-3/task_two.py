import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema

print('# Load the group photo and convert it to grayscale')
img = cv2.imread('group.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the size of the sub-window
window_size = 64

print('# Extract all possible sub-windows from the image')
sub_windows = []
for i in range(0, gray.shape[0] - window_size + 1, window_size):
    for j in range(0, gray.shape[1] - window_size + 1, window_size):
        sub_window = gray[i:i + window_size, j:j + window_size]
        sub_windows.append(sub_window.flatten())

print('# Convert the list of sub-windows to a numpy array')
sub_windows = np.array(sub_windows)

print('# Load the face dataset')
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
n_samples, h, w = lfw_people.images.shape
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print('# Reshape each face image to match the size of the sub-window')
X = X.reshape((n_samples, h*w))
X = np.array([cv2.resize(x, (window_size, window_size)) for x in X])
X = X.reshape((n_samples, window_size * window_size))

print('# Compute the mean face vector and subtract it from each face vector')
mean_face = X.mean(axis=0)
X = X - mean_face

print('# Perform PCA on the mean-centered face vectors')
pca = PCA(n_components=50)
pca.fit(X)

print('# Project each sub-window vector onto the face subspace')
sub_windows_pca = pca.transform(sub_windows)

print('# Compute the distance between each sub-window vector and its projection')
sub_windows_dist = np.linalg.norm(sub_windows - pca.inverse_transform(sub_windows_pca), axis=1)

print('# Reshape the distance vector to a 2D array')
dist_map = sub_windows_dist.reshape((gray.shape[0] // window_size, gray.shape[1] // window_size))

# Plot the distance map
plt.imshow(dist_map, cmap='jet')
plt.colorbar()
plt.title('Distance map of the image')
plt.show()

# Implement multi-scale face detection
scales = [32, 48, 64]  # The sizes of the sub-windows
threshold = 1000  # The threshold for the distance
face_locations = []  # The list of face locations

for scale in scales:
    # Resize the image and the sub-windows
    resized_img = cv2.resize(gray, (gray.shape[1] // scale, gray.shape[0] // scale))
    resized_sub_windows = np.array([cv2.resize(x, (scale, scale)) for x in sub_windows])

    # Project each resized sub-window vector onto the face subspace
    resized_sub_windows_pca = pca.transform(resized_sub_windows)

    # Compute the distance between each resized sub-window vector and its projection
    resized_sub_windows_dist = np.linalg.norm(
        resized_sub_windows - pca.inverse_transform(resized_sub_windows_pca), axis=1)

    # Reshape the distance vector to a 2D array
    resized_dist_map = resized_sub_windows_dist.reshape(
        (resized_img.shape[0] // scale, resized_img.shape[1] // scale))

    # Find the local minima of the distance map that are below the threshold
    local_minima = argrelextrema(resized_dist_map, np.less, order=1)
    local_minima = np.array(local_minima).T
    local_minima = local_minima[resized_dist_map[local_minima[:, 0], local_minima[:, 1]] < threshold]

    # Convert the local minima to face locations
    local_minima = local_minima * scale
    face_locations.extend([(x, y, scale, scale) for x, y in local_minima])

# Merge the face locations from different scales and eliminate the overlapping ones
face_locations, weights = cv2.groupRectangles(face_locations, groupThreshold=1, eps=0.2)

# Draw the bounding boxes of the final face locations on the original image
for x, y, w, h in face_locations:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save and show the detection result
# cv2.imwrite('group_scale.png', img)
cv2.imshow('Face detection result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()