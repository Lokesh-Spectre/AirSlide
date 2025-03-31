import pickle
import numpy as np

# Load the trained model & label encoder
model = tf.keras.models.load_model("your_model.h5")
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Example new data point (same preprocessing as training)
new_data = np.array([[0.1, 0.2, -0.3, ..., 0.4, -0.1, 0.2]])  # Shape (1, 63)
new_data = scaler.transform(new_data)  # Standardize

# Predict & decode label
predicted_probs = model.predict(new_data)
predicted_class = np.argmax(predicted_probs)
decoded_label = label_encoder.inverse_transform([predicted_class])

print("Predicted Class:", decoded_label[0])
