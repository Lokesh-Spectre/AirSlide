import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Load Images from Folder
image_folder = "hand"  # Folder containing images
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

for image_path in image_files:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for hand_landmarks in results.multi_hand_landmarks:
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='red')
            connections = mp_hands.HAND_CONNECTIONS
            segments = [(points[start], points[end]) for start, end in connections]
            line_collection = Line3DCollection(segments, color='blue', linewidths=2)
            ax.add_collection3d(line_collection)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=10, azim=120)
        plt.show()
    else:
        print(f"No hands detected in {image_path}.")
