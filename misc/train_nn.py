import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from tensorflow.keras.callbacks import EarlyStopping


# Load dataset
df = pd.read_csv("landmarks.csv")

# Separate features & labels
X = df.iloc[:, 1:].values  # All landmark coordinates
y = df.iloc[:, 0].values   # Class labels (strings)

# Convert string labels to numeric categories
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Converts labels to [0,1,2,3]

# One-hot encode labels
y_onehot = to_categorical(y_encoded, num_classes=len(label_encoder.classes_))

# Normalize features (Standard Scaling)
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y_encoded)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Build Neural Network Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),  # Prevent overfitting
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer (4 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(X_train, y_train, epochs=1, batch_size=8, validation_data=(X_test, y_test), callbacks=[early_stop])
print(X_train)
# Evaluate
train_loss, train_acc = model.evaluate(X_train, y_train)
test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Train Accuracy: {train_acc:.2f}, Test Accuracy: {test_acc:.2f}")

model.save("gesture_classifier.h5")
# np.save("scaler.npy", scaler.mean_)  # Save scaler mean

# Save Label Encoder for Decoding Predictions Later
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
