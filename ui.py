from PyQt6.QtWidgets import QMainWindow, QPushButton, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QSystemTrayIcon, QTextEdit, QDoubleSpinBox
from PyQt6.QtGui import QIcon, QFont, QFontDatabase
from s2s import S2S
from subtitles_ui import SubtitleWindow
from model_functions import get_all_models
import threading
import keyboard
import os

class StreamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('S2S Audio Stream')
        self.resize(400, 400)
        self.setMinimumSize(400, 400)

        # --- Window Icon ---
        self.setWindowIcon(QIcon("icon.png"))

        # Create subtitle window but don't show it yet
        self.subtitle_window = SubtitleWindow()
        
        # Initialize S2S with the subtitle window
        self.s2s = S2S(subtitle_window=self.subtitle_window)

        self.init_ui()
        
        # Show subtitle window after main UI is fully initialized
        self.subtitle_window.show()

        # --- Global Hotkey Setup ---
        try:
            self._hotkey_thread = threading.Thread(target=self._setup_global_hotkey, daemon=True)
            self._hotkey_thread.start()
        except Exception as e:
            print(f"Error setting up global hotkey: {e}")

    def _setup_global_hotkey(self):
        
        # Right Ctrl + F24 (keyboard uses 'right ctrl+f24')
        hotkey = 'right ctrl+f24'
        print(f"Registering global hotkey: {hotkey}")
        keyboard.add_hotkey(hotkey, self._global_toggle_stream)
        keyboard.wait()

    def _global_toggle_stream(self):
        self.stream_btn.click()

    def init_ui(self):
        stream_layout = QHBoxLayout()
        self.stream_btn = QPushButton('Start Stream')
        self.stream_btn.setCheckable(True)
        self.stream_btn.setMinimumHeight(40)
        self.stream_btn.clicked.connect(self.toggle_stream)
        stream_layout = (self.stream_btn)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(30)
        self.populate_models()
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

        # Voice Modification Configs
        voice_layout = QHBoxLayout()
        # Soft voice
        self.voice_soft_btn = QPushButton('Soft Voice: True')
        self.voice_soft_btn.setCheckable(True)
        self.voice_soft_btn.setChecked(True)
        self.voice_soft_btn.setMinimumHeight(35)
        self.voice_soft_btn.clicked.connect(self.toggle_voice_soft)
        voice_layout.addWidget(self.voice_soft_btn)
        # TTS speed
        self.voice_speed_spin = QDoubleSpinBox()
        self.voice_speed_spin.setMinimum(0.5)
        self.voice_speed_spin.setMaximum(2.0)
        self.voice_speed_spin.setSingleStep(0.01)
        self.voice_speed_spin.setValue(1.0)
        self.voice_speed_spin.setDecimals(2)
        self.voice_speed_spin.setSuffix("x")
        self.voice_speed_spin.setMinimumHeight(35)
        self.voice_speed_spin.valueChanged.connect(self.change_voice_speed)
        voice_layout.addWidget(self.voice_speed_spin)

        # Font selection
        font_layout = QHBoxLayout()
        font_label = QLabel('Subtitle Font:')
        self.font_combo = QComboBox()
        self.font_combo.setMinimumHeight(30)
        self.populate_fonts()
        self.font_combo.currentTextChanged.connect(self.change_font)
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_combo)

        # Manual text input
        text_layout = QVBoxLayout()
        text_label = QLabel('Manual Text Input:')
        self.text_input = QTextEdit()
        self.text_input.setMinimumHeight(80)
        self.text_input.setMaximumHeight(120)
        self.text_input.setPlaceholderText('Enter text to synthesize...')
        self.synthesize_btn = QPushButton('Synthesize')
        self.synthesize_btn.setMinimumHeight(35)
        self.synthesize_btn.clicked.connect(self.synthesize_text)
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_input)
        text_layout.addWidget(self.synthesize_btn)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addLayout(model_layout)
        layout.addLayout(voice_layout)
        layout.addLayout(font_layout)
        layout.addLayout(text_layout)
        layout.addWidget(stream_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def populate_fonts(self):
        """Populate the font dropdown with fonts from .fonts folder only"""

        self.font_combo.clear()
        current_font_name = self.subtitle_window.current_font.family()

        fonts_folder = os.path.join(os.path.dirname(__file__), '.fonts')
        if os.path.isdir(fonts_folder):
            font_index_to_select = 0
            for i, file in enumerate(os.listdir(fonts_folder)):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_path = os.path.join(fonts_folder, file)
                    ids = QFontDatabase.addApplicationFont(font_path)
                    if ids != -1:
                        loaded_fonts = QFontDatabase.applicationFontFamilies(ids)
                        for loaded_font in loaded_fonts:
                            self.font_combo.addItem(loaded_font)
                            # If this matches the current font, remember its index
                            if loaded_font == current_font_name:
                                font_index_to_select = self.font_combo.count() - 1
            
            # Set the combo box to the current font being used
            if self.font_combo.count() > 0:
                self.font_combo.setCurrentIndex(font_index_to_select)

    def change_font(self):
        """Change the font of the subtitle window"""
        font_name = self.font_combo.currentText()
        if font_name:
            self.subtitle_window.change_font(font_name)


    def toggle_voice_soft(self):
        """Toggle the soft voice"""
        is_checked = self.voice_soft_btn.isChecked()
        self.voice_soft_btn.setText(f'Soft Voice: {str(is_checked)}')
        try:
            self.s2s.chage_voice_soft(is_checked)
        except Exception as e:
            print(f"Error changing soft voice:: {e}")
    
    def change_voice_speed(self):
        value = self.voice_speed_spin.value()
        try:
            self.s2s.change_voice_speed(value)
        except Exception as e:
            print(f"Error changing voice speed:: {e}")

    def populate_models(self):
        """Populate the model dropdown with available models"""
        models = get_all_models()
        self.model_combo.clear()
        
        if not models:
            self.model_combo.addItem("No models available", None)
            return

        if models and isinstance(models[0], dict):
            for model in models:
                display_name = model['display_name']
                key = model['key']
                self.model_combo.addItem(display_name, key)

            current_model_key = 'en_US-hfc_female-medium'
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == current_model_key:
                    self.model_combo.setCurrentIndex(i)
                    break
        else:
            for model in models:
                display_name = model.replace('.onnx', '') if isinstance(model, str) else str(model)
                self.model_combo.addItem(display_name, model)

            current_model = 'en_US-hfc_female-medium'
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == current_model:
                    self.model_combo.setCurrentIndex(i)
                    break

    def change_model(self):
        """Handle model selection change"""
        if self.model_combo.currentData():
            try:
                model_key = self.model_combo.currentData()
                self.s2s.set_model(model_key)
            except Exception as e:
                print(f"Error changing model: {e}")

    def synthesize_text(self):
        """Synthesize the text from the input field"""
        text = self.text_input.toPlainText().strip()
        if text:
            try:
                self.s2s.synthesize_text(text)
                self.text_input.clear()
            except Exception as e:
                print(f"Error synthesizing text: {e}")

    def toggle_stream(self):
        """Toggles the Voice Stream"""
        if self.stream_btn.isChecked():
            self.stream_btn.setText('Stop Stream')
            self.s2s.start_stream()
        else:
            self.stream_btn.setText('Start Stream')
            self.s2s.stop_stream()
            self.subtitle_window.clear_subtitle()

    def closeEvent(self, event):
        """Handle window close event - close subtitle window and clean up"""
        try:
            # Stop the stream if it's running
            if hasattr(self, 's2s'):
                self.s2s.stop_stream()
            
            # Close the subtitle window
            if hasattr(self, 'subtitle_window'):
                self.subtitle_window.close()
            
            # Hide the system tray icon
            if hasattr(self, 'tray_icon'):
                self.tray_icon.hide()
            
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        # Accept the close event
        event.accept()
