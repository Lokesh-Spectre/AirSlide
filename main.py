import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize Pose model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with Pose model
        results = pose.process(rgb_frame)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the frame
        cv2.imshow('Full-Body Pose Estimation', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
