import os
import numpy as np
from sklearn.decomposition import PCA
import cv2
import matplotlib.pyplot as plt

# Function to load images from a folder
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Load images from CelebA_Train folder
folder_path = 'dataset/CelebA_Train'  
images = load_images(folder_path)

# Convert images to NumPy array
data = np.array(images)

# Flatten the images
flattened_data = data.reshape(data.shape[0], -1)

# Number of eigenfaces to consider (adjust as needed)
num_eigenfaces = 20

# Perform PCA
pca = PCA(n_components=num_eigenfaces)
pca.fit(flattened_data)

# Display eigenfaces
eigenfaces = pca.components_.reshape((num_eigenfaces, data.shape[1], data.shape[2]))

for i in range(num_eigenfaces):
    plt.subplot(4, 5, i + 1)  # Change the layout based on your preference
    plt.imshow(eigenfaces[i], cmap='gray')
    plt.axis('off')

plt.show()

# Project a face image into the face space and show the reconstructed face
# Pick an image from the dataset
test_image = flattened_data[0]

# Project the image onto the eigenfaces space
projected = pca.transform([test_image])

# Reconstruct the image
reconstructed = pca.inverse_transform(projected)

# Reshape the reconstructed image to its original shape
reconstructed_image = reconstructed.reshape(data.shape[1], data.shape[2])

# Display the original and reconstructed images
plt.subplot(1, 2, 1)
plt.imshow(data[0], cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

plt.show()