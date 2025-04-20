import sys
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle
import pyautogui
from math import atan2, degrees, sqrt

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                               QFormLayout, QVBoxLayout, QLabel, QLineEdit, QGroupBox)
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QImage

# --------------------- Global Setup (Load Models, Scaler, etc.) --------------------- #
# Load trained model
model = tf.keras.models.load_model("angle_gesture_classifier.h5")

# Load label encoder
with open("angle_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load the scaler
with open("angle_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --------------------- Utility Functions --------------------- #
def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three points.
    """
    ang = degrees(atan2(c[1] - b[1], c[0] - b[0]) -
                   atan2(a[1] - b[1], a[0] - b[0]))
    return abs(ang)

def extract_angles_from_landmarks(landmarks):
    """
    Given a MediaPipe hand landmarks object, extract a feature array of angles.
    Expects at least 21 points.
    """
    points = [(lm.x, lm.y) for lm in landmarks.landmark]
    if len(points) >= 21:
        angles = [
            calculate_angle(points[0], points[1], points[2]),
            calculate_angle(points[1], points[2], points[3]),
            calculate_angle(points[2], points[3], points[4]),
            calculate_angle(points[5], points[6], points[7]),
            calculate_angle(points[6], points[7], points[8]),
        ]
        return np.array(angles).reshape(1, -1)
    return None

def is_fist(landmarks, threshold=0.1):
    """
    A simple heuristic to determine if a hand (assumed left hand)
    is making a fist by checking the distance from the wrist (landmark 0)
    to the fingertips.
    """
    wrist = landmarks.landmark[0]
    tip_ids = [4, 8, 12, 16, 20]
    closed_count = 0
    for tip_id in tip_ids:
        tip = landmarks.landmark[tip_id]
        dist = sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
        if dist < threshold:
            closed_count += 1
    return closed_count >= 4

# --------------------- QThread Classes --------------------- #
class MediaPipeThread(QThread):
    # This thread will capture frames, process MediaPipe, and if left hand is a fist,
    # it will extract right-hand angles and emit them.
    angles_signal = Signal(object)         # emits extracted angles (numpy array)
    display_frame_signal = Signal(object)    # emits the current frame (optional for display)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Initialize MediaPipe for two-hand detection.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=True,
                                          max_num_hands=2,
                                          min_detection_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with MediaPipe.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            left_hand = None
            right_hand = None

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    label = handedness.classification[0].label  # 'Left' or 'Right'
                    # Draw landmarks on frame.
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    if label.lower() == "left":
                        left_hand = hand_landmarks
                    else:
                        right_hand = hand_landmarks

            # Only proceed if left hand is detected and is making a fist.
            if left_hand is not None and is_fist(left_hand):
                if right_hand is not None:
                    angles = extract_angles_from_landmarks(right_hand)
                    if angles is not None:
                        # Send the angles to be classified.
                        self.angles_signal.emit(angles)
            # Optionally, emit the frame for display purposes in the main window.
            self.display_frame_signal.emit(frame)
        cap.release()

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

class GestureClassificationThread(QThread):
    # This thread receives angles from MediaPipeThread and classifies the gesture.
    result_signal = Signal(str)   # emits the predicted gesture label

    @Slot(object)
    def process_angles(self, angles):
        # Normalize the angles.
        angles_scaled = scaler.transform(angles)
        # Predict with your ML model.
        predicted_probs = model.predict(angles_scaled)
        predicted_class = int(np.argmax(predicted_probs))
        gesture = label_encoder.inverse_transform([predicted_class])[0]
        print(gesture)
        self.result_signal.emit(gesture)

    def run(self):
        # Nothing to do in run as this thread works via slots.
        self.exec()

# --------------------- Main Application Window --------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Presenter Settings")
        self.setup_ui()

        # Create worker threads (but don’t start them yet).
        self.mp_thread = MediaPipeThread()
        self.class_thread = GestureClassificationThread()

        # Connect signals: media pipe thread sends angles to classification thread.
        self.mp_thread.angles_signal.connect(self.class_thread.process_angles)
        self.class_thread.result_signal.connect(self.update_gesture_label)
        # Optionally, display processed frame from the MediaPipe thread.
        # self.mp_thread.display_frame_signal.connect(self.update_display)

    def setup_ui(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Settings form (add your settings as needed)
        settings_group = QGroupBox("Settings")
        form_layout = QFormLayout()
        # Example settings
        self.staticModeEdit = QLineEdit("True")
        self.detectionConfidenceEdit = QLineEdit("0.5")
        form_layout.addRow("Static Image Mode:", self.staticModeEdit)
        form_layout.addRow("Detection Confidence:", self.detectionConfidenceEdit)
        settings_group.setLayout(form_layout)

        # Gesture result display.
        self.gesture_label = QLabel("Gesture: None")
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)

        # Start button to launch threads.
        self.start_button = QPushButton("Start Gesture Detection")
        self.start_button.clicked.connect(self.start_detection)

        # Layout all widgets.
        layout = QVBoxLayout(central_widget)
        layout.addWidget(settings_group)
        layout.addWidget(self.gesture_label)
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)

    @Slot()
    def start_detection(self):
        # Here you can update parameters from settings if needed.
        # For instance, update the MediaPipe initialization based on user input.
        # Restart or start the threads.
        if not self.mp_thread.isRunning():
            self.mp_thread.start()
        if not self.class_thread.isRunning():
            self.class_thread.start()
        self.start_button.setEnabled(False)
        self.gesture_label.setText("Gesture: Waiting for input...")

    @Slot(str)
    def update_gesture_label(self, gesture):
        self.gesture_label.setText(f"Gesture: {gesture}")


    @Slot(object)
    def update_display(self, frame):
        # Convert frame (BGR cv2 image) to QImage.
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        # Convert QImage to QPixmap and scale it.
        pixmap = QPixmap.fromImage(q_image).scaled(self.video_label.width(), self.video_label.height())
        self.video_label.setPixmap(pixmap)
    def closeEvent(self, event):
        # Make sure to stop threads when closing the application.
        self.mp_thread.stop()
        self.class_thread.quit()
        self.class_thread.wait()
        event.accept()

# --------------------- Run the Application --------------------- #
if __name__ == "__main__":
    from PySide6.QtGui import QImage, QPixmap
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
