import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split

# Path to dataset
data_path = "dataset"

categories = ["with_mask", "without_mask"]
data = []
labels = []

# Load images
for category in categories:
    path = os.path.join(data_path, category)
    label = categories.index(category)

    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (100, 100))
            data.append(image)
            labels.append(label)
        except:
            pass

# Convert to numpy arrays
data = np.array(data) / 255.0
labels = np.array(labels)

# One-hot encoding
labels = to_categorical(labels, 2)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Build CNN model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("model/mask_detector.h5")

print("Model trained and saved!")