# ui_module.py
from PySide6.QtWidgets import QMainWindow, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtCore import Qt, Signal

class MainWindow(QMainWindow):
    # Signals to trigger start/stop detection in the integration module
    start_signal = Signal()
    stop_signal = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture-Driven Presentation Controller")
        self.resize(400, 300)
        self._setup_ui()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Status label to show system state or detected gestures
        self.status_label = QLabel("System Idle...")
        self.status_label.setAlignment(Qt.AlignCenter)

        # Start/Stop buttons to control gesture detection
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.start_button.clicked.connect(self.on_start)
        self.stop_button.clicked.connect(self.on_stop)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(button_layout)
        central_widget.setLayout(main_layout)

    def on_start(self):
        self.status_label.setText("Starting gesture detection...")
        self.start_signal.emit()

    def on_stop(self):
        self.status_label.setText("Stopping gesture detection...")
        self.stop_signal.emit()
