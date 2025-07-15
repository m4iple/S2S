from PyQt6.QtWidgets import QApplication
from ui import StreamWindow

def main():
    app = QApplication([])
    win = StreamWindow()
    win.show()
    app.exec()

if __name__ == '__main__':
    main()