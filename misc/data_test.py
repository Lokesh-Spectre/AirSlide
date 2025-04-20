import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog

# Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open file dialog to select a video
def select_video():
    # root = tk.Tk()
    # root.withdraw()  # Hide the main window
    # file_path = filedialog.askopenfilename(title="Select a video file", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    # return file_path
    return "/home/lokesh/Videos/AirSlide/export_raw_4/AirSlide_dataset_4-no_gesture.mp4"

video_path = select_video()
if not video_path:
    print("No file selected. Exiting.")
    exit()

cap = cv2.VideoCapture(video_path)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # if results.multi_handedness[idx].classification[0].label == "Left":  # Only process right hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            print(results.multi_handedness[idx].classification[0].label)
    cv2.imshow("Right Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
