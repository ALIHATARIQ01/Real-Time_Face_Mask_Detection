import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import pickle

DATASET_PATH = 'data'
IMG_SIZE = 224

data = []
labels = []

for category in ['with_mask', 'without_mask']:
    folder = os.path.join(DATASET_PATH, category)
    label = 0 if category == 'with_mask' else 1
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            data.append(img)
            labels.append(label)
        except:
            continue

X = np.array(data) / 255.0
y = to_categorical(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("✅ Data preprocessing completed and saved.")
