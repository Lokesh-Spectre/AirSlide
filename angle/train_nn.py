import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Check for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is available and memory growth is set!")
    except RuntimeError as e:
        print(f"⚠️ GPU setup error: {e}")
else:
    print("❌ No GPU detected! Running on CPU.")

# Load CSV
df = pd.read_csv("angle/angles.csv")

# Split features (angles) and labels
X = df.iloc[:, 1:].values  # Angles (ignoring the Label column)
y = df.iloc[:, 0].values   # Gesture Labels

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
with open("angle_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Scale the angles (normalize)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use the same scaler for test data

# Save the scaler
with open("angle_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Reshape for LSTM (samples, timesteps, features)
X_train_scaled = np.expand_dims(X_train_scaled, axis=-1)
X_test_scaled = np.expand_dims(X_test_scaled, axis=-1)

# Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1], )),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

# Compile with GPU-optimized settings
optimizer = tf.keras.optimizers.AdamW(learning_rate=0.003, weight_decay=1e-4)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train on GPU
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=8, validation_data=(X_test_scaled, y_test))

# Evaluate on test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the model
model.save("angle_gesture_classifier.h5")

print("🎉 Training complete! Model, scaler, and label encoder saved.")
