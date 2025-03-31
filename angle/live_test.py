import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
from math import atan2, degrees

# Load trained model
model = tf.keras.models.load_model("angle_gesture_classifier.h5")

# Load label encoder
with open("angle_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the scaler
with open("angle_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Initialize OpenCV
cap = cv2.VideoCapture(0)

# Function to calculate angles between three points
def calculate_angle(a, b, c):
    ang = degrees(atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang)

# Function to extract angles from landmarks
def extract_angles(image, frame):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            
            if len(points) >= 21:
                angles = [
                    calculate_angle(points[0], points[1], points[2]),  # Thumb angle
                    calculate_angle(points[1], points[2], points[3]),  # Index finger angle
                    calculate_angle(points[2], points[3], points[4]),  # Middle finger angle
                    calculate_angle(points[5], points[6], points[7]),  # Ring finger angle
                    calculate_angle(points[6], points[7], points[8]),  # Pinky finger angle
                ]

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

                return np.array(angles).reshape(1, -1)  # Reshape for model input
    return None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    angles = extract_angles(frame, frame)

    if angles is not None:
        # Normalize using the same scaler from training
        angles_scaled = scaler.transform(angles)

        # Predict gesture
        predicted_probs = model.predict(angles_scaled)
        predicted_class = np.argmax(predicted_probs)

        # Decode label correctly
        gesture_name = label_encoder.inverse_transform([predicted_class])[0]

        # Display gesture on frame
        cv2.putText(frame, f'Gesture: {gesture_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
