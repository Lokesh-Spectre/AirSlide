from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QColor, QRadialGradient, QBrush, QKeyEvent, QCursor
from PySide6.QtCore import Qt, QTimer, QPoint
import sys

class PointerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.showFullScreen()  # Make the window take the entire screen
        
        self.pointer_pos = QCursor.pos()  # Start at current mouse position
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_position)
        self.timer.start(10)
    
    def update_position(self):
        self.pointer_pos = QCursor.pos()  # Use Qt's built-in method for mouse position
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Create a radial gradient without a distinct border
        gradient = QRadialGradient(self.pointer_pos, 20)
        gradient.setColorAt(0, QColor(255, 0, 0, 255))  # Solid red center
        gradient.setColorAt(0.8, QColor(255, 0, 0, 100))  # Softer red mid-tone
        gradient.setColorAt(1, QColor(255, 0, 0, 0))   # Fully transparent outer edge
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)  # Remove any border
        painter.drawEllipse(self.pointer_pos.x() - 20, self.pointer_pos.y() - 20, 40, 40)
    
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Q:
            QApplication.instance().quit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    pointer = PointerWindow()
    pointer.show()
    sys.exit(app.exec())
