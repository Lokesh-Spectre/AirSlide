# gesture_detector.py
from PySide6.QtCore import QObject, Signal
import time
import random

def detect_gesture():
    """
    Mock function to simulate gesture detection.
    Replace this with your trained AI model's detection logic later.
    """
    time.sleep(1)  # Simulate processing delay
    gestures = ['swipe_left', 'swipe_right', 'pinch', 'zoom']
    return random.choice(gestures)

class GestureDetector(QObject):
    gestureDetected = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = True

    def process(self):
        """Continuously detect gestures and emit events."""
        while self._running:
            gesture = detect_gesture()
            self.gestureDetected.emit(gesture)

    def stop(self):
        """Stop the detection loop."""
        self._running = False
