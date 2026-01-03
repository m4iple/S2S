from PyQt6.QtWidgets import QMainWindow, QLabel, QSystemTrayIcon, QWidget, QVBoxLayout, QGraphicsOpacityEffect, QSizePolicy
from PyQt6.QtGui import QIcon, QFont, QFontDatabase
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QParallelAnimationGroup
import os
from collections import deque
from utils.config import load_config

EASING_MAP = {
    'Linear': QEasingCurve.Type.Linear,
    'InQuad': QEasingCurve.Type.InQuad,
    'OutQuad': QEasingCurve.Type.OutQuad,
    'InOutQuad': QEasingCurve.Type.InOutQuad,
    'InCubic': QEasingCurve.Type.InCubic,
    'OutCubic': QEasingCurve.Type.OutCubic,
    'InOutCubic': QEasingCurve.Type.InOutCubic,
    'InQuart': QEasingCurve.Type.InQuart,
    'OutQuart': QEasingCurve.Type.OutQuart,
    'InOutQuart': QEasingCurve.Type.InOutQuart,
}


class SubtitleWindow(QMainWindow):
    """Main window for displaying subtitles with animations"""
    
    subtitle_update_signal = pyqtSignal(str)
    subtitle_clear_signal = pyqtSignal()
    subtitle_styled_signal = pyqtSignal(str, str, str)  # text, font, color

    def __init__(self, config_path="configs/subtitles.toml"):
        super().__init__()
        
        self.cfg = load_config(config_path)
        
        self.setWindowTitle('Subtitles')
        window_width = self.cfg['window']['width']
        window_height = self.cfg['window']['height']
        self.resize(window_width, window_height)
        self.setMinimumSize(window_width, window_height)
        self.setWindowIcon(QIcon("icon.png"))
        
        if self.cfg['window']['frameless']:
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        if self.cfg['window']['transparent_background']:
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._position_window()

        self.tray_icon = QSystemTrayIcon(QIcon("icon.png"), self)
        self.tray_icon.setToolTip("S2S Audio Stream")
        self.tray_icon.show()

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        alignment = self._get_alignment()
        self.layout.setAlignment(alignment)
        self.layout.setSpacing(self.cfg['display']['line_spacing'])
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.subtitle_lines = []
        self.subtitle_queue = deque()
        self.max_lines = self.cfg['display']['max_lines']
        self.fade_duration = self.cfg['display']['fade_duration']

        self.current_font = self.load_custom_fonts()
        
        self.animation_config = self._build_animation_config()

        self.subtitle_update_signal.connect(self.add_subtitle_line)
        self.subtitle_clear_signal.connect(self.clear_subtitle)
        self.subtitle_styled_signal.connect(lambda t, f, c: self.add_subtitle_line(t, f, c))
    
    def _position_window(self):
        """Position window based on config"""
        position = self.cfg['window']['position']
        screen = self.screen() or self.windowHandle().screen()
        
        if screen:
            screen_geometry = screen.geometry()
            x = (screen_geometry.width() - self.width()) // 2
            
            if position == 'top-center':
                self.move(x, 0)
            elif position == 'bottom-center':
                y = screen_geometry.height() - self.height()
                self.move(x, y)
        else:
            self.move(0, 0)
    
    def _get_alignment(self):
        """Get Qt alignment from config"""
        vertical = Qt.AlignmentFlag.AlignTop
        
        alignment_str = self.cfg['display']['alignment']
        if alignment_str == 'left':
            horizontal = Qt.AlignmentFlag.AlignLeft
        elif alignment_str == 'right':
            horizontal = Qt.AlignmentFlag.AlignRight
        else:  # center
            horizontal = Qt.AlignmentFlag.AlignHCenter
        
        return vertical | horizontal
    
    def _build_animation_config(self):
        """Build animation config dict from loaded TOML config"""
        config = {}
        
        if self.cfg['animation']['appear']['enabled']:
            config['appear'] = {
                'duration': self.cfg['animation']['appear']['duration'],
                'easing': EASING_MAP.get(self.cfg['animation']['appear']['easing'], QEasingCurve.Type.OutCubic),
                'property': b'opacity',
                'start_value': self.cfg['animation']['appear']['start_opacity'],
                'end_value': self.cfg['animation']['appear']['end_opacity'],
            }
        
        if self.cfg['animation']['disappear']['enabled']:
            config['disappear'] = {
                'duration': self.cfg['animation']['disappear']['duration'],
                'easing': EASING_MAP.get(self.cfg['animation']['disappear']['easing'], QEasingCurve.Type.InCubic),
                'property': b'opacity',
                'start_value': self.cfg['animation']['disappear']['start_opacity'],
                'end_value': self.cfg['animation']['disappear']['end_opacity'],
            }
        
        if self.cfg['animation']['move_up']['enabled']:
            config['move_up'] = {
                'duration': self.cfg['animation']['move_up']['duration'],
                'easing': EASING_MAP.get(self.cfg['animation']['move_up']['easing'], QEasingCurve.Type.InOutQuart),
                'property': b'geometry',
            }
        
        return config

    def load_custom_fonts(self):
        """Load fonts from .fonts folder and return the configured font."""
        font_size = self.cfg['font']['size']
        font_family = self.cfg['font']['family']
        
        if font_family != 'auto':
            font = QFont(font_family, font_size)
            self._apply_font_style(font)
            return font
        
        fonts_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.fonts')
        if os.path.isdir(fonts_folder):
            for file in os.listdir(fonts_folder):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_path = os.path.join(fonts_folder, file)
                    font_id = QFontDatabase.addApplicationFont(font_path)
                    if font_id != -1:
                        font_families = QFontDatabase.applicationFontFamilies(font_id)
                        if font_families:
                            font = QFont(font_families[0], font_size)
                            self._apply_font_style(font)
                            return font
        
        font = QFont("Arial", font_size)
        self._apply_font_style(font)
        return font
    
    def _apply_font_style(self, font):
        """Apply font weight and style from config"""
        if self.cfg['font']['weight'] == 'bold':
            font.setBold(True)
        if self.cfg['font']['style'] == 'italic':
            font.setItalic(True)

    def create_animation(self, widget, anim_type):
        """Creates an animation based on the configuration."""
        config = self.animation_config[anim_type]
        
        if config['property'] == b'opacity':
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

    def set_subtitle(self, text, font_name=None, color=None):
        """Public method to add text to the subtitle queue via a signal."""
        if font_name or color:
            self.subtitle_styled_signal.emit(text, font_name if font_name else "", color if color else "")
        else:
            self.subtitle_update_signal.emit(text)

    def add_subtitle_line(self, text, font_name=None, color=None):
        """Adds a new line of text to the subtitles with optional custom font and color."""
        if len(self.subtitle_lines) >= self.max_lines:
            self._remove_first_line(and_add_new=True)
        else:
            self._create_new_label(text, font_name, color)

    def _create_new_label(self, text, font_name=None, color=None):
        """Creates and animates a new label with optional custom font and color."""
        label = QLabel(text, self.central_widget)
        
        # Load custom font if specified
        custom_font = None
        if font_name:
            custom_font = self._load_font_from_file(font_name)
        
        label.setFont(custom_font if custom_font else self.current_font)
        
        alignment = self._get_text_alignment()
        label.setAlignment(alignment)
        label.setWordWrap(self.cfg['display']['word_wrap'])
        
        # Use custom color if provided, otherwise use configured color
        font_color = color if color else self.cfg['font']['color']
        label.setStyleSheet(f"color: {font_color};")
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        
        opacity_effect = QGraphicsOpacityEffect(label)
        opacity_effect.setOpacity(0.0)
        label.setGraphicsEffect(opacity_effect)
        
        self.layout.addWidget(label)
        
        timer = QTimer(self)
        timer.setSingleShot(True)
        
        line_data = {'label': label, 'timer': timer, 'animation': None}
        self.subtitle_lines.append(line_data)

        appear_anim = self.create_animation(label, 'appear')
        line_data['animation'] = appear_anim
        appear_anim.finished.connect(lambda: self._on_animation_finished(line_data))
        appear_anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        
        timer.timeout.connect(lambda: self._start_fade_out(line_data))
        timer.start(self.fade_duration)

    def _on_animation_finished(self, line_data):
        """Called when an animation finishes to clean up the reference."""
        if line_data in self.subtitle_lines:
            line_data['animation'] = None
    
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
        
        # Stop any existing animation
        if line_to_remove['animation']:
            line_to_remove['animation'].stop()
            line_to_remove['animation'] = None

        disappear_anim = self.create_animation(label_to_remove, 'disappear')
        disappear_anim.finished.connect(label_to_remove.deleteLater)

        animation_group = QParallelAnimationGroup(self)
        animation_group.addAnimation(disappear_anim)

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
            if line_data['animation']:
                line_data['animation'].stop()
            line_data['label'].deleteLater()
        self.subtitle_lines.clear()
        self.subtitle_queue.clear()

    def _get_text_alignment(self):
        """Get Qt text alignment from config"""
        alignment_str = self.cfg['display']['alignment']
        if alignment_str == 'left':
            return Qt.AlignmentFlag.AlignLeft
        elif alignment_str == 'right':
            return Qt.AlignmentFlag.AlignRight
        else:  # center
            return Qt.AlignmentFlag.AlignCenter
    
    def change_font(self, font_name):
        """Changes the font for all current and future subtitle labels."""
        if font_name:
            font_size = self.cfg['font']['size']
            self.current_font = QFont(font_name, font_size)
            self._apply_font_style(self.current_font)
            for line_data in self.subtitle_lines:
                line_data['label'].setFont(self.current_font)
    
    def _load_font_from_file(self, font_name):
        """Load a font from the .fonts folder by name."""
        if not font_name:
            return None
            
        fonts_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.fonts')
        
        if not os.path.isdir(fonts_folder):
            return None
        
        # Find the font file
        for file in os.listdir(fonts_folder):
            if file.lower().endswith(('.ttf', '.otf')):
                # Match font name (with or without extension)
                file_name_no_ext = os.path.splitext(file)[0]
                if file_name_no_ext == font_name or file == font_name:
                    font_path = os.path.join(fonts_folder, file)
                    
                    # Load the font
                    if os.path.exists(font_path):
                        font_id = QFontDatabase.addApplicationFont(font_path)
                        if font_id != -1:
                            font_families = QFontDatabase.applicationFontFamilies(font_id)
                            if font_families:
                                custom_font = QFont(font_families[0], self.cfg['font']['size'])
                                self._apply_font_style(custom_font)
                                return custom_font
        
        return None
