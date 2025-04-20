# integration_module.py
from PySide6.QtCore import QThread, QObject
from gesture_detector import GestureDetector
from presentation_module import process_gesture

class Controller(QObject):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

        # Create a QThread for gesture detection
        self.thread = QThread()
        self.gesture_detector = GestureDetector()
        self.gesture_detector.moveToThread(self.thread)

        # Connect the gesture detected signal to the handler
        self.gesture_detector.gestureDetected.connect(self.handle_gesture)
        # When thread starts, call the process method of our detector
        self.thread.started.connect(self.gesture_detector.process)

        # Connect UI start/stop signals
        self.ui.start_signal.connect(self.start_detection)
        self.ui.stop_signal.connect(self.stop_detection)

    def start_detection(self):
        if not self.thread.isRunning():
            self.thread.start()
            self.ui.status_label.setText("Gesture detection started!")
        else:
            self.ui.status_label.setText("Gesture detection already running.")

    def stop_detection(self):
        if self.thread.isRunning():
            self.gesture_detector.stop()  # Signal the loop to exit
            self.thread.quit()
            self.thread.wait()
            self.ui.status_label.setText("Gesture detection stopped.")
        else:
            self.ui.status_label.setText("Gesture detection is not running.")

    def handle_gesture(self, gesture):
        # Update the UI and process the gesture event
        self.ui.status_label.setText(f"Detected: {gesture}")
        process_gesture(gesture)
