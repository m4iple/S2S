import os
os.environ["ORT_LOGGING_LEVEL"] = "3"

from PyQt6.QtWidgets import QApplication
from src.s2s import S2s
from ui.s2s import S2sWindow
from ui.subtitles import SubtitleWindow
from src.subtitles import SubtitleController

def main():
    app = QApplication([])

    subtitle_window = SubtitleWindow()
    
    subtitle_controller = SubtitleController(subtitle_window)
    
    s2s_instance = S2s(subtitle_window=subtitle_controller)
    
    win = S2sWindow(s2s_instance, subtitle_window, subtitle_controller)
    win.show()
    
    subtitle_window.show()
    
    app.exec()

if __name__ == '__main__':
    main()