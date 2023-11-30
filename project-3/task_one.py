import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from PIL import Image
import os

# Load the training data
X_train = []  # List to store the training images
image_dir = 'dataset/CelebA_Train'
for filename in os.listdir(image_dir):  # Loop through the folder containing images
    img = Image.open(os.path.join(image_dir, filename))  # Open the image file
    img = img.resize((64, 64))  # Resize the image to 64x64 pixels
    img = np.array(img)  # Convert the image to a numpy array
    img = img.flatten()  # Flatten the array to a one-dimensional vector
    X_train.append(img)  # Append the vector to the list

X_train = np.array(X_train)  # Convert the list to a numpy array
n_samples, n_features = X_train.shape

# Normalize the data
X_train = X_train / 255.0

# Compute the mean face
mean_face = np.mean(X_train, axis=0)

# Subtract the mean face from the data
X_train = X_train - mean_face

# Perform PCA to compute eigenfaces
pca = PCA(n_components=30)  # Change this to use a different number of eigenfaces
pca.fit(X_train)
eigenfaces = pca.components_  # Shape: (n_components, n_features)

# Show the eigenface images
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(eigenfaces[i].reshape(64, 64, 3), cmap='gray')
    ax.set_title(f'Eigenface {i+1}')
plt.show()

# Project a face image into the face space
face = X_train[0]  # Change this to use a different face image
face_projected = pca.transform(face.reshape(1, -1))  # Shape: (1, n_components)

# Show the reconstructed face
face_reconstructed = pca.inverse_transform(face_projected)  # Shape: (1, n_features)
face_reconstructed = face_reconstructed + mean_face  # Add back the mean face
plt.imshow(face_reconstructed.reshape(64, 64, 3), cmap='gray')
plt.title('Reconstructed face')
plt.show()