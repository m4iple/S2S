from PyQt6.QtWidgets import QMainWindow, QPushButton, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QSystemTrayIcon, QTextEdit
from PyQt6.QtGui import QIcon
from s2s import S2S
import threading
import keyboard
from model_functions import get_all_models

class StreamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('S2S Audio Stream')
        self.resize(400, 350)
        self.setMinimumSize(400, 350)
        self.s2s = S2S()
        
        # --- Window Icon ---
        self.setWindowIcon(QIcon("icon.ico"))

        # --- System Tray Icon ---
        self.tray_icon = QSystemTrayIcon(QIcon("icon.ico"), self)
        self.tray_icon.setToolTip("S2S Audio Stream")
        self.tray_icon.show()

        self.init_ui()

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
        self.toggle_btn.click()

    def init_ui(self):
        self.toggle_btn = QPushButton('Start Stream')
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setMinimumHeight(40)
        self.toggle_btn.clicked.connect(self.toggle_stream)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.model_combo.setMinimumHeight(30)
        self.populate_models()
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

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
        layout.addLayout(text_layout)

        # Add softmod toggle button
        self.softmod_btn = QPushButton('Soft Voice: True')
        self.softmod_btn.setCheckable(True)
        self.softmod_btn.setChecked(True)
        self.softmod_btn.setMinimumHeight(35)
        self.softmod_btn.clicked.connect(self.toggle_soft_voice)
        layout.addWidget(self.softmod_btn)
        layout.addWidget(self.toggle_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_soft_voice(self):
        is_checked = self.softmod_btn.isChecked()
        self.softmod_btn.setText(f'Soft Voice: {str(is_checked)}')
        try:
            self.s2s.chage_soft_voice(is_checked)
        except Exception as e:
            print(f"Error changing soft voice:: {e}")

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
        if self.toggle_btn.isChecked():
            self.toggle_btn.setText('Stop Stream')
            self.s2s.start_stream()
        else:
            self.toggle_btn.setText('Start Stream')
            self.s2s.stop_stream()