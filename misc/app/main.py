# main.py
import sys
from PySide6.QtWidgets import QApplication
from ui_module import MainWindow
from integration_module import Controller

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Initialize the controller which integrates all modules
    controller = Controller(window)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
