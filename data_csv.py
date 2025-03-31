import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from math import atan2, degrees

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    ang = degrees(atan2(c[1] - b[1], c[0] - b[0]) - atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang)

# Specify folder containing videos
folder_path = "/home/lokesh/Videos/AirSlide/export_raw_3"

if not os.path.exists(folder_path):
    print("Folder not found. Exiting.")
    exit()

video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
landmarks_data = []
angles_data = []

for video in video_files:
    video_path = os.path.join(folder_path, video)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        print(f"\r{video}:\t{current_frame}/{total_frames}", end="")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame)
        Label = video.split("-")[-1].replace(".mp4", "")
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for idx, classification in enumerate(results.multi_handedness):
                handedness = classification.classification[0].label
                if handedness == "Left":  # Process only right hand
                    hand_landmarks = results.multi_hand_landmarks[idx]
                    row = [Label]
                    base_x, base_y, base_z = hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z
                    
                    for lm in hand_landmarks.landmark:
                        row.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])  # Normalize landmarks
                    landmarks_data.append(row)
                    
                    # Compute angles (example: thumb to index, index to middle, etc.)
                    angles_row = [Label]
                    points = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                    if len(points) >= 21:
                        angles_row.extend([
                            calculate_angle(points[0], points[1], points[2]),  # Thumb angle
                            calculate_angle(points[1], points[2], points[3]),  # Index finger angle
                            calculate_angle(points[2], points[3], points[4]),  # Middle finger angle
                            calculate_angle(points[5], points[6], points[7]),  # Ring finger angle
                            calculate_angle(points[6], points[7], points[8]),  # Pinky finger angle
                        ])
                        angles_data.append(angles_row)
    
    cap.release()
    print(" File Processed")

# Save landmarks to CSV
landmark_columns = ["Label"] + [f"{X}{i}" for i in range(21) for X in "XYZ"]
pd.DataFrame(landmarks_data, columns=landmark_columns).to_csv("landmarks.csv", index=False)

# Save angles to CSV
angle_columns = ["Label", "Thumb Angle", "Index Angle", "Middle Angle", "Ring Angle", "Pinky Angle"]
pd.DataFrame(angles_data, columns=angle_columns).to_csv("angles.csv", index=False)

print("Processing complete. Data saved to landmarks.csv and angles.csv")
