from PyQt6.QtWidgets import QMainWindow, QLabel, QSystemTrayIcon, QWidget, QVBoxLayout, QGraphicsOpacityEffect, QSizePolicy
from PyQt6.QtGui import QIcon, QFont, QFontDatabase, QScreen
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup, QParallelAnimationGroup
import os
from collections import deque

class SubtitleWindow(QMainWindow):
    # Signals for thread-safe subtitle updates
    subtitle_update_signal = pyqtSignal(str)
    subtitle_clear_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        # --- Window Creation (Restored to original spec) ---
        self.setWindowTitle('Subtitles')
        self.resize(1920, 330)
        self.setMinimumSize(1920, 330)
        self.setWindowIcon(QIcon("icon.ico"))
        #self.setStyleSheet("background-color: green;")
        #self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # --- Position window at the top of the screen ---
        # screen = QScreen.primaryScreen()
        # screen_geometry = screen.geometry()
        # Center the window horizontally at the top of the primary screen
        screen = self.screen() or self.windowHandle().screen()
        if screen:
            screen_geometry = screen.geometry()
            x = (screen_geometry.width() - self.width()) // 2
            self.move(x, 0)
        else:
            self.move(0, 0)


        # --- System Tray Icon (Restored) ---
        self.tray_icon = QSystemTrayIcon(QIcon("icon.ico"), self)
        self.tray_icon.setToolTip("S2S Audio Stream")
        self.tray_icon.show()

        # --- Main Widget and Layout for multiple lines ---
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        # --- Alignment changed to AlignTop and AlignHCenter ---
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        self.layout.setSpacing(5) # Added a small spacing between lines
        self.layout.setContentsMargins(0, 0, 0, 0)  # Remove margins for full-width text

        # --- Subtitle Management ---
        self.subtitle_lines = []
        self.subtitle_queue = deque()
        self.max_lines = 3
        self.fade_duration = 5000  # 5 seconds per line

        # --- Font Management ---
        self.current_font = self.load_custom_fonts()

        # --- Animation Configurations (Easily Replaceable) ---
        self.animation_config = {
            'appear': {
                'duration': 300,
                'easing': QEasingCurve.Type.OutCubic,
                'property': b'opacity', # Changed to opacity
                'start_value': 0.0,
                'end_value': 1.0,
            },
            'disappear': {
                'duration': 200,
                'easing': QEasingCurve.Type.InCubic,
                'property': b'opacity', # Changed to opacity
                'start_value': 1.0,
                'end_value': 0.0,
            },
            'move_up': {
                'duration': 400,
                'easing': QEasingCurve.Type.InOutQuart,
                'property': b'geometry',
            }
        }

        # Connect signals to slots for thread-safe updates
        self.subtitle_update_signal.connect(self.add_subtitle_line)
        self.subtitle_clear_signal.connect(self.clear_subtitle)

    def load_custom_fonts(self):
        """Load fonts from .fonts folder and return the first one."""
        fonts_folder = os.path.join(os.path.dirname(__file__), '.fonts')
        if os.path.isdir(fonts_folder):
            for file in os.listdir(fonts_folder):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_path = os.path.join(fonts_folder, file)
                    font_id = QFontDatabase.addApplicationFont(font_path)
                    if font_id != -1:
                        font_families = QFontDatabase.applicationFontFamilies(font_id)
                        if font_families:
                            font = QFont(font_families[0], 24)
                            return font
        # Default font if no custom fonts are found
        return QFont("Arial", 24)

    def create_animation(self, widget, anim_type):
        """Creates an animation based on the configuration."""
        config = self.animation_config[anim_type]
        
        # Target the opacity effect for fade animations, otherwise the widget itself
        if config['property'] == b'opacity':
            # Ensure the widget has an opacity effect
            if not widget.graphicsEffect():
                opacity_effect = QGraphicsOpacityEffect(widget)
                widget.setGraphicsEffect(opacity_effect)
            anim_target = widget.graphicsEffect()
        else:
            anim_target = widget

        anim = QPropertyAnimation(anim_target, config['property'])
        anim.setDuration(config['duration'])
        anim.setEasingCurve(config['easing'])
        if 'start_value' in config:
            anim.setStartValue(config['start_value'])
        if 'end_value' in config:
            anim.setEndValue(config['end_value'])
        return anim

    def set_subtitle(self, text):
        """Public method to add text to the subtitle queue via a signal."""
        self.subtitle_update_signal.emit(text)

    def add_subtitle_line(self, text):
        """Adds a new line of text to the subtitles."""
        if len(self.subtitle_lines) >= self.max_lines:
            self._remove_first_line(and_add_new=True)
        else:
            self._create_new_label(text)

    def _create_new_label(self, text):
        """Creates and animates a new label."""
        label = QLabel(text, self.central_widget)
        label.setFont(self.current_font)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(False)  # Disable word wrapping to prevent splitting before exceeding width
        label.setStyleSheet("color: white;")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)  # Allow horizontal expansion
        
        # Set up opacity effect for fading
        opacity_effect = QGraphicsOpacityEffect(label)
        opacity_effect.setOpacity(0.0)  # Start fully transparent
        label.setGraphicsEffect(opacity_effect)
        
        self.layout.addWidget(label)
        
        timer = QTimer(self)
        timer.setSingleShot(True)
        
        line_data = {'label': label, 'timer': timer, 'animation': None}
        self.subtitle_lines.append(line_data)

        # Start appear animation
        appear_anim = self.create_animation(label, 'appear')
        appear_anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        line_data['animation'] = appear_anim
        
        # Start fade-out timer for this specific line
        timer.timeout.connect(lambda: self._start_fade_out(line_data))
        timer.start(self.fade_duration)

    def _start_fade_out(self, line_data):
        """Initiates the fade-out animation for a specific line."""
        if line_data in self.subtitle_lines:
            self._remove_line(self.subtitle_lines.index(line_data))

    def _remove_first_line(self, and_add_new=False):
        """Removes the oldest subtitle line and optionally adds a new one."""
        if self.subtitle_lines:
            text_to_add = self.subtitle_queue.popleft() if and_add_new and self.subtitle_queue else None
            self._remove_line(0, text_to_add)

    def _remove_line(self, index, text_to_add=None):
        """Removes a subtitle line at a specific index with animation."""
        if not (0 <= index < len(self.subtitle_lines)):
            return

        line_to_remove = self.subtitle_lines.pop(index)
        label_to_remove = line_to_remove['label']
        line_to_remove['timer'].stop()

        # Disappear animation
        disappear_anim = self.create_animation(label_to_remove, 'disappear')
        disappear_anim.finished.connect(label_to_remove.deleteLater)

        # Group animations
        animation_group = QParallelAnimationGroup(self)
        animation_group.addAnimation(disappear_anim)

        # Move-up animation for remaining lines
        if index == 0 and self.subtitle_lines:
            for line_data in self.subtitle_lines:
                label = line_data['label']
                move_anim = self.create_animation(label, 'move_up')
                start_geo = label.geometry()
                end_geo = start_geo.translated(0, -(label_to_remove.height() + self.layout.spacing()))
                move_anim.setStartValue(start_geo)
                move_anim.setEndValue(end_geo)
                animation_group.addAnimation(move_anim)
        
        if text_to_add:
            animation_group.finished.connect(lambda: self._create_new_label(text_to_add))

        animation_group.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


    def clear_subtitle(self):
        """Clears all subtitles immediately."""
        for line_data in self.subtitle_lines:
            line_data['timer'].stop()
            line_data['label'].deleteLater()
        self.subtitle_lines.clear()
        self.subtitle_queue.clear()

    def change_font(self, font_name):
        """Changes the font for all current and future subtitle labels."""
        if font_name:
            self.current_font = QFont(font_name, 24)
            for line_data in self.subtitle_lines:
                line_data['label'].setFont(self.current_font)
