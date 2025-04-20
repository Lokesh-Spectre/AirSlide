import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyautogui
from math import atan2, degrees, sqrt
import zmq
import time


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://127.0.0.1:5555")

import math
radius = 200
center_x, center_y = 960, 540  # Example: screen center

# Load trained model
model = tf.keras.models.load_model("angle_gesture_classifier.h5")

# Load label encoder
with open("angle_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the scaler
with open("angle_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize MediaPipe Hands with support for 2 hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Initialize OpenCV capture
cap = cv2.VideoCapture(0)

# Function to calculate angle between three points (for classification)
def calculate_angle(a, b, c):
    ang = degrees(atan2(c[1] - b[1], c[0] - b[0]) -
                   atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang)

# Function to extract angles from a hand's landmarks (expects at least 21 points)
def extract_angles_from_landmarks(landmarks):
    points = [(lm.x, lm.y) for lm in landmarks.landmark]
    if len(points) >= 21:
        angles = [
            calculate_angle(points[0], points[1], points[2]),  # Thumb angle
            calculate_angle(points[1], points[2], points[3]),  # Index finger angle
            calculate_angle(points[2], points[3], points[4]),  # Middle finger angle
            calculate_angle(points[5], points[6], points[7]),  # Ring finger angle
            calculate_angle(points[6], points[7], points[8])   # Pinky finger angle
        ]
        return np.array(angles).reshape(1, -1)
    return None

# Function to process both hands and draw landmarks on the frame.
# Returns left hand landmarks and right hand landmarks (if detected).
def process_hands(image, frame):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    left_hand = None
    right_hand = None
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label  # 'Left' or 'Right'
            # Draw the landmarks on frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            if label.lower() == "left":
                left_hand = hand_landmarks
            else:
                right_hand = hand_landmarks
    return left_hand, right_hand

# Simple heuristic to determine if the left hand is in a fist.
def is_fist(landmarks, threshold=0.1):
    # We'll consider the distances from the wrist (landmark 0) to all fingertips.
    wrist = landmarks.landmark[0]
    tip_ids = [4, 8, 12, 16, 20]  # Thumb tip, Index tip, etc.
    closed_count = 0
    for tip_id in tip_ids:
        tip = landmarks.landmark[tip_id]
        # Euclidean distance in normalized coordinates.
        dist = sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        if dist < threshold:
            closed_count += 1
    # If at least 4 out of 5 fingers are close, consider it a fist.
    return closed_count >= 4

# Variables for additional gesture logic
gd_buff = 10
current_gesture = ""
last_gesture = ""

# Swipe detection variables (using right-hand as reference)
swipe_start_position = None   # Record the start position when swipe gesture begins.
swipe_detected = False        # Indicates if a swipe has been detected.
SWIPE_DISTANCE_THRESHOLD = 50 # Minimum horizontal movement in screen pixels.

frame_flag = 3

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process both hands on the current frame.
    right_hand, left_hand  = process_hands(frame, frame)

    # Default: No gesture predicted.
    gesture_name = "None"

    # Only if left hand is detected and is in a fist, proceed with right hand classification.
    if left_hand is not None and is_fist(left_hand):
        if right_hand is not None:
            angles = extract_angles_from_landmarks(right_hand)
            if angles is not None:
                # Normalize using the same scaler from training.
                angles_scaled = scaler.transform(angles)
                # Predict gesture
                predicted_probs = model.predict(angles_scaled)
                predicted_class = np.argmax(predicted_probs)
                gesture_name = label_encoder.inverse_transform([predicted_class])[0]
                # Decode label correctly.
        else:
            # Right hand not detected, so gesture remains "None"
            gesture_name = "Right hand missing"
    else:
        # Left hand is not in a fist; reset gesture classification and swipe detection.
        gesture_name = "Left fist not active"
        swipe_start_position = None
        swipe_detected = False

    if gesture_name != "None":
        current_gesture = gesture_name
    # Display gesture on frame.
    cv2.putText(frame, f'Gesture: {gesture_name}', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if current_gesture == last_gesture:
        print(f"GESTURE: {current_gesture}!")
    elif gd_buff == 0:
        gd_buff = 10
        print(f"GESTURE CHANGE: {current_gesture}!")
    else:
        gd_buff -= 1
    last_gesture = current_gesture

    # Process pointer and swipe only if the predicted gesture from right hand is valid.
    if left_hand is not None and is_fist(left_hand) and right_hand is not None:
        # Get frame dimensions.
        h, w, _ = frame.shape
        # Get right hand's landmarks for pointer/swipe processing.
        wrist = right_hand.landmark[0]
        index_tip = right_hand.landmark[8]
        wrist_coords = (int(wrist.x * w), int(wrist.y * h))
        index_tip_coords = (int(index_tip.x * w), int(index_tip.y * h))
        
        # Pointer: if gesture is "pointer", move the mouse pointer.
        if current_gesture.lower() == "pointer":
            angle = degrees(atan2(index_tip_coords[1] - wrist_coords[1],
                                   index_tip_coords[0] - wrist_coords[0]))
            cv2.putText(frame, f'Angle: {int(angle)}', (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            screen_w, screen_h = pyautogui.size()
            screen_x = int(index_tip_coords[0] / w * screen_w)
            screen_y = int(index_tip_coords[1] / h * screen_h)
            # Optionally flipping X-axis if needed:
            socket.send_string(f"{screen_x},{screen_y}")

            # pyautogui.moveTo(screen_w - screen_x, screen_y)

        # Swipe detection: if gesture is "swipe" (from right hand).
        if gesture_name.lower() == "swipe":
            screen_w, screen_h = pyautogui.size()
            current_position = (int(index_tip.x * screen_w), int(index_tip.y * screen_h))
            if swipe_start_position is None:
                swipe_start_position = current_position
                swipe_detected = False
                print("Swipe started at:", swipe_start_position)
            else:
                delta_x = current_position[0] - swipe_start_position[0]
                if not swipe_detected and abs(delta_x) > SWIPE_DISTANCE_THRESHOLD:
                    if delta_x > 0:
                        pyautogui.press('right')
                        print("Detected a swipe to the right!" +"RIGHT"*100)
                    else:
                        pyautogui.press('left')
                        print("Detected a swipe to the left!")
                    swipe_detected = True  # Prevent further detections until gesture changes.
            cv2.putText(frame, f'Swipe Pos: {current_position}', (50, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            # Reset swipe state if gesture is no longer "swipe".
            swipe_start_position = None
            swipe_detected = False

    # Show the processed video frame.
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
