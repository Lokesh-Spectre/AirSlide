import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model
model = tf.keras.models.load_model("gesture_classifier.h5")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the same scaler used in training
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # ✅ Correctly loads the trained scaler

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize OpenCV
cap = cv2.VideoCapture(0)

def extract_landmarks(image, frame):
    """Extracts and normalizes hand landmarks using MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z  # Base point
            
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x - base_x)  # Normalize relative to first point
                landmarks.append(landmark.y - base_y)
                landmarks.append(landmark.z - base_z)
            
            # Convert to NumPy and normalize (rescale between -1 and 1)
            landmarks = np.array(landmarks)
            max_value = np.max(np.abs(landmarks))  # Find max absolute value for scaling
            if max_value > 0:
                landmarks = landmarks / max_value  # Normalize in range [-1, 1]

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            return landmarks  # Return properly normalized landmarks
    return None  # No hand detected

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    landmarks = extract_landmarks(frame, frame)  # Pass frame twice for drawing

    if landmarks is not None:
        # Convert to NumPy array and reshape
        landmarks = np.array(landmarks).reshape(1, -1)

        # Normalize using the trained scaler
        # landmarks = scaler.transform(landmarks)  # ✅ Scaled correctly
        # print("Normalized & Scaled Landmarks:", landmarks)

        # Predict gesture
        predicted_probs = model.predict(landmarks)
        predicted_class = np.argmax(predicted_probs)

        # Decode label correctly
        gesture_name = label_encoder.inverse_transform([predicted_class])[0]

        # Display gesture on frame
        cv2.putText(frame, f'Gesture: {gesture_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
