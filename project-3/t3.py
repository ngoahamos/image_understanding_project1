import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import cv2

# Function to load images from a given directory
def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, target_size)  # Resize the image
            img = img.flatten()[:np.prod(target_size)]  # Flatten and truncate to the target size
            images.append(img)  # Add to the list
    return images

# Load celebrity faces dataset
celebrities_folder = 'dataset/LFW_Test'
celebrities = os.listdir(celebrities_folder)

X = []  # List to store the flattened images
y = []  # List to store the corresponding labels

for i, celebrity in enumerate(celebrities):
    celebrity_folder = os.path.join(celebrities_folder, celebrity)
    images = load_images_from_folder(celebrity_folder)
    
    X.extend(images)
    y.extend([i] * len(images))  # Assign a label to each celebrity

X = np.array(X)
y = np.array(y)
print(len(X))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Apply PCA
n_components = 200
pca = PCA(n_components=n_components, whiten=True).fit(X_train)

# Transform data using PCA
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a KNN classifier on the PCA-transformed data
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train_pca, y_train)

# Predict the labels of the test data
y_pred = knn.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Display some results
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(64, 64), cmap='gray')
    ax.set_title(f'Predicted: {celebrities[y_pred[i]]}')
    ax.axis('off')

plt.show()
