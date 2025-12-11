from PyQt6.QtWidgets import QApplication
import sys

# Import the MainWindow from GUI.IHM
try:
    from GUI.IHM import MainWindow
except Exception as e:
    # Provide a friendly error so the developer can debug import problems
    print("Failed to import MainWindow from GUI.IHM:", e)
    raise


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
