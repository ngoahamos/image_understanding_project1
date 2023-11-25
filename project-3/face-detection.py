import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the Haar cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Read the input image
image = cv2.imread('group.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the window size and step size for sliding window
window_size = 64
step_size = 8

# Function to compute the covariance matrix
def compute_covariance_matrix(data):
    mean_face = np.mean(data, axis=0)
    centered_data = data - mean_face
    covariance_matrix = np.dot(centered_data.T, centered_data) / (data.shape[0] - 1)
    return covariance_matrix, mean_face

# Function to perform eigen decomposition
def eigen_decomposition(covariance_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # Sort eigenvalues and corresponding eigenvectors in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    return eigenvalues, eigenvectors

# Function to project data onto the principal components
def project_data(data, eigenvectors, mean_face):
    centered_data = data - mean_face
    return np.dot(centered_data, eigenvectors)

# Function to reconstruct data from the principal components
def reconstruct_data(projected_data, eigenvectors, mean_face):
    return np.dot(projected_data, eigenvectors.T) + mean_face

# Function to compute the face distance for a sub-window
def face_distance(sub_window, mean_face, eigenvectors):
    # Project the sub-window to the face subspace
    projected_sub_window = project_data(sub_window.flatten(), eigenvectors, mean_face.flatten())
    # Reconstruct the sub-window from the projection
    reconstructed_sub_window = reconstruct_data(projected_sub_window, eigenvectors, mean_face.flatten())
    # Reshape the reconstructed sub-window
    reconstructed_sub_window = reconstructed_sub_window.reshape(sub_window.shape)
    # Compute the mean squared error between the sub-window and the reconstruction
    mse = np.mean((sub_window - reconstructed_sub_window) ** 2)
    # Return the inverse of the mse as the face distance
    return 1 / mse

# Function for sliding window and computing face distances
def sliding_window(image, scale_factor, mean_face, eigenvectors):
    # Get the height and width of the image
    height, width = image.shape
    # Initialize an empty list to store the face distances
    face_distances = []
    # Loop over the image with a sliding window at different scales
    for scale in scale_factor:
        resized_image = cv2.resize(image, (int(width*scale), int(height*scale)))
        for y in range(0, resized_image.shape[0] - window_size + 1, step_size):
            for x in range(0, resized_image.shape[1] - window_size + 1, step_size):
                # Extract the sub-window from the image
                sub_window = resized_image[y:y + window_size, x:x + window_size]
                # Compute the face distance for the sub-window
                distance = face_distance(sub_window, mean_face, eigenvectors)
                # Append the face distance and the window coordinates to the list
                face_distances.append((distance, int(x / scale), int(y / scale), int(window_size / scale)))
    # Return the list of face distances
    return face_distances

# Apply the sliding window function to the gray image with different scales
scale_factors = [0.8, 1.0, 1.2]  # You can adjust the scale factors as needed

# Detect faces and compute PCA on detected faces
faces = []  # List to store the face images
for (x, y, w, h) in face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5):
    face = gray[y:y + h, x:x + w]  # Crop the face from the image
    face = cv2.resize(face, (window_size, window_size))  # Resize the face to the window size
    faces.append(face.flatten())  # Flatten the face and append to the list

faces = np.array(faces)  # Convert the list to a numpy array

# Compute the covariance matrix, mean face, and perform eigen decomposition
covariance_matrix, mean_face = compute_covariance_matrix(faces)
eigenvalues, eigenvectors = eigen_decomposition(covariance_matrix)

# Apply the sliding window function
face_distances = sliding_window(gray, scale_factors, mean_face, eigenvectors)

# Create a faceness map for visualization
faceness_map = np.zeros_like(gray)

# Define a threshold for face detection
threshold = 0.01

# Draw a rectangle around each detected face above the threshold
for (face_distance, x, y, size) in face_distances:
    if face_distance > threshold:
        cv2.rectangle(image, (x, y), (x + size, y + size), (255, 0, 0), 2)
        faceness_map[y:y + size, x:x + size] += face_distance  # Add face distance to faceness map

# Show the original image with rectangles around detected faces
cv2.imshow('Face detection using PCA', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Show the faceness map
plt.imshow(faceness_map, cmap='hot', interpolation='nearest')
plt.title('Faceness Map')
plt.colorbar()
plt.show()