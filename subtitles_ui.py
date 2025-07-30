from PyQt6.QtWidgets import QMainWindow, QLabel
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, QTimer


class SubtitleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Subtitles')
        self.resize(1920,330)
        self.setMinimumSize(1920, 330)
        self.setWindowIcon(QIcon("icon.ico"))
        self.label = QLabel('', self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setWordWrap(True)  # Enable word wrapping
        self.label.setContentsMargins(20, 10, 20, 10)  # Add some padding
        self.label.setStyleSheet("QLabel { padding: 10px; }")  # Additional padding via stylesheet
        
        # Load fonts from .fonts folder and set default
        self.load_custom_fonts()
        
        # Setup auto fade-out timer
        self.fade_timer = QTimer()
        self.fade_timer.setSingleShot(True)  # Timer runs only once
        self.fade_timer.timeout.connect(self.auto_clear_subtitle)
        self.fade_duration = 10000  # 10 seconds in milliseconds
        
        self.setCentralWidget(self.label)

    def load_custom_fonts(self):
        """Load fonts from .fonts folder and set the first one as default"""
        import os
        from PyQt6.QtGui import QFontDatabase
        
        fonts_folder = os.path.join(os.path.dirname(__file__), '.fonts')
        font_loaded = False
        
        if os.path.isdir(fonts_folder):
            for file in os.listdir(fonts_folder):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_path = os.path.join(fonts_folder, file)
                    ids = QFontDatabase.addApplicationFont(font_path)
                    if ids != -1:
                        loaded_fonts = QFontDatabase.applicationFontFamilies(ids)
                        if loaded_fonts and not font_loaded:
                            # Use the first loaded font as default
                            font = QFont(loaded_fonts[0])
                            font.setPointSize(16)
                            self.label.setFont(font)
                            font_loaded = True
        
        # If no custom fonts were loaded, use default font
        if not font_loaded:
            font = QFont()
            font.setPointSize(24)
            self.label.setFont(font)

    def format_subtitle_text(self, text):
        """Format text with intelligent line breaks for better readability"""
        if not text:
            return text
            
        # Split very long sentences at natural break points
        max_chars_per_line = 80  # Adjust based on your preference
        
        if len(text) <= max_chars_per_line:
            return text
            
        # Try to break at sentence boundaries first
        sentences = text.split('. ')
        if len(sentences) > 1:
            formatted_lines = []
            current_line = ""
            
            for i, sentence in enumerate(sentences):
                # Add the period back except for the last sentence
                if i < len(sentences) - 1:
                    sentence += ". "
                    
                if len(current_line + sentence) <= max_chars_per_line:
                    current_line += sentence
                else:
                    if current_line:
                        formatted_lines.append(current_line.strip())
                    current_line = sentence
                    
            if current_line:
                formatted_lines.append(current_line.strip())
                
            return '\n'.join(formatted_lines)
        
        # If no sentence breaks, break at word boundaries
        words = text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
                
        if current_line:
            lines.append(current_line)
            
        return '\n'.join(lines)

    def set_subtitle(self, text):
        print(text)
        formatted_text = self.format_subtitle_text(text)
        self.label.setText(formatted_text)
        
        # Reset and start the auto fade-out timer
        self.fade_timer.stop()  # Stop any existing timer
        if text.strip():  # Only start timer if there's actual text
            self.fade_timer.start(self.fade_duration)

    def auto_clear_subtitle(self):
        """Automatically clear the subtitle after the fade duration"""
        self.label.setText('')

    def clear_subtitle(self):
        self.fade_timer.stop()  # Stop the timer when manually clearing
        self.label.setText('')