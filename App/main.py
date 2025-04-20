import sys
import zmq
import threading
import time
from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QCheckBox
)
from PySide6.QtGui import QPainter, QColor, QRadialGradient, QBrush
from PySide6.QtCore import Qt, QTimer, QPoint

class PointerWindow(QWidget):

    listener_thread_shutdown=False
    def __init__(self):
        super().__init__()

        # Use flags to enforce an always-on-top, input-transparent overlay.
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |    # Always on top
            Qt.Tool |
            Qt.WindowTransparentForInput  # Makes window transparent to all input events
        )
        # Set attributes for transparency
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_NoSystemBackground)

        self.showFullScreen()

        self.pointer_pos = QPoint(0, 0)  # Initial pointer position

        # Setup the ZMQ listener for pointer movement
        self.setup_zmq_listener()
        # Setup a separate listener for shutdown messages (used by external shutdown signals)
        # self.setup_shutdown_listener()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

    def setup_zmq_listener(self):
        def listen():
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect("tcp://127.0.0.1:5555")
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            while not self.listener_thread_shutdown:
                message = socket.recv_string()
                try:
                    x, y = map(int, message.split(","))
                    self.pointer_pos = QPoint(x, y)
                except Exception:
                    continue  # Ignore malformed messages
            
        thread = threading.Thread(target=listen, daemon=False)
        thread.start()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QRadialGradient(self.pointer_pos, 20)
        gradient.setColorAt(0, QColor(255, 0, 0, 255))
        gradient.setColorAt(0.8, QColor(255, 0, 0, 100))
        gradient.setColorAt(1, QColor(255, 0, 0, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.pointer_pos.x() - 20, self.pointer_pos.y() - 20, 40, 40)

    def closeEvent(self, event):
        self.listener_thread_shutdown = True
        event.accept()



class GestureControlUI(QWidget):

    def __init__(self, pointerWindow):
        super().__init__()
        self.pointerWindow = pointerWindow
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setFixedSize(250, 200)
        self.setWindowTitle("Gesture Control Panel")

        self.gesture_label = QLabel("Gesture: None")
        self.start_button = QPushButton("Start Gesture Detection")
        self.test_button = QPushButton("Test Gesture")
        self.keyboard_input_checkbox = QCheckBox("Enable Keyboard Input")

        layout = QVBoxLayout()
        layout.addWidget(self.start_button)
        layout.addWidget(self.test_button)
        layout.addWidget(self.keyboard_input_checkbox)
        layout.addWidget(self.gesture_label)
        # self.setLayout(layout)

        self.test_button.clicked.connect(self.test_gesture)
        self.start_button.clicked.connect(self.start_gesture_detection)

    def test_gesture(self):
        # Simulate gesture detection result
        simulated_gesture = "Swipe Right"
        self.gesture_label.setText(f"Gesture: {simulated_gesture}")

        if self.keyboard_input_checkbox.isChecked():
            # Perform action if keyboard input is enabled
            print(f"[Gesture Triggered] Executing action for: {simulated_gesture}")
        else:
            print(f"[Gesture Detected] {simulated_gesture} (action disabled due to keyboard input toggle)")

    def start_gesture_detection(self):
        print("Gesture detection started (placeholder)")

    def closeEvent(self, event):
        # When closing this UI, also send a shutdown signal to the micro service.
        self.pointerWindow.close()
        # send_shutdown_signal()
        self.close()
        event.accept()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    pointer = PointerWindow()
    ui = GestureControlUI(pointer)
    ui.show()
    sys.exit(app.exec())
