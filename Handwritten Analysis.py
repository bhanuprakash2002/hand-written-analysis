#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from PIL import Image

# Load the MNIST dataset
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)

# Preprocess the data
X = X / 255.0  # Normalize pixel values between 0 and 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction using Histogram of Oriented Gradients (HOG)
def extract_hog_features(X):
    features = []
    for image in X:
        # Extract HOG features
        hog_features = hog(image.reshape((28, 28)), orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
        
        features.append(hog_features)
    
    return np.array(features)

# Extract HOG features from the training and testing sets
X_train_features = extract_hog_features(X_train)
X_test_features = extract_hog_features(X_test)

# Scale the feature vectors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_features)
X_test_scaled = scaler.transform(X_test_features)

# Train the SVM model
svm_classifier = svm.SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Predict the labels for the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Prepare your own grayscale digit image
custom_image = Image.open('3-Figure2-1.png')
grayscale_image = custom_image.convert('L')
resized_image = grayscale_image.resize((28, 28))
normalized_image = np.array(resized_image) / 255.0

# Extract HOG features from the input image
input_features = extract_hog_features(np.array([normalized_image]))

# Scale the feature vector
input_scaled = scaler.transform(input_features)

# Make a prediction
predicted_label = svm_classifier.predict(input_scaled)

print("Predicted digit: {}".format(predicted_label[0]))

