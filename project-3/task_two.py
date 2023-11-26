import cv2
import numpy as np

# Load the image
image_path = 'group.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Resize the image for multi-scale detection
resized_image = cv2.resize(gray_image, (0, 0), fx=0.8, fy=0.8)

# Load pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(resized_image, scaleFactor=1.3, minNeighbors=5)

# Extract face regions and resize them
face_regions = [cv2.resize(resized_image[y:y+h, x:x+w], (100, 100)) for (x, y, w, h) in faces]

# Perform PCA
data = np.array([face.flatten() for face in face_regions], dtype=np.float32)

# Compute mean and eigenvectors
mean, eigenvectors = cv2.PCACompute(data, mean=None, maxComponents=10)

# Project faces to PCA subspace
projected_faces = cv2.PCAProject(data, mean, eigenvectors)

# Display the faceness map
faceness_map = np.linalg.norm(data - projected_faces, axis=1).reshape(resized_image.shape)
faceness_map = cv2.normalize(faceness_map, None, 0, 255, cv2.NORM_MINMAX)
faceness_map = cv2.resize(faceness_map, (gray_image.shape[1], gray_image.shape[0]))

cv2.imshow('Faceness Map', faceness_map.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()