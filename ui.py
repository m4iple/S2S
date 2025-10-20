from PyQt6.QtWidgets import QMainWindow, QPushButton, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QTextEdit, QDoubleSpinBox, QFrame, QSpinBox, QCheckBox
from PyQt6.QtGui import QIcon, QFont, QFontDatabase
from s2s import S2S
from subtitles_ui import SubtitleWindow
from model_functions import get_all_models
from debug import set_capture_training_data
import threading
import keyboard
import os

# ============================================================================
# Helper Functions for UI Element Creation
# ============================================================================

def create_spinbox(min_val, max_val, step, value, suffix="", decimals=0, callback=None, height=35):
    """Factory function for creating and configuring spinboxes
    
    Args:
        min_val: Minimum value
        max_val: Maximum value
        step: Single step increment
        value: Initial value
        suffix: Optional suffix text (e.g., " Hz")
        decimals: Number of decimal places
        callback: Optional slot to connect valueChanged signal to
        height: Minimum height in pixels
        
    Returns:
        QDoubleSpinBox: Configured spinbox widget
    """
    spin = QDoubleSpinBox()
    spin.setMinimum(min_val)
    spin.setMaximum(max_val)
    spin.setSingleStep(step)
    spin.setValue(value)
    spin.setDecimals(decimals)
    if suffix:
        spin.setSuffix(suffix)
    spin.setMinimumHeight(height)
    if callback:
        spin.valueChanged.connect(callback)
    return spin

def create_button(text, callback=None, checkable=False, checked=False, height=35):
    """Factory function for creating and configuring buttons
    
    Args:
        text: Button text
        callback: Optional slot to connect clicked signal to
        checkable: Whether button is checkable/togglable
        checked: Initial checked state
        height: Minimum height in pixels
        
    Returns:
        QPushButton: Configured button widget
    """
    btn = QPushButton(text)
    btn.setCheckable(checkable)
    btn.setChecked(checked)
    btn.setMinimumHeight(height)
    if callback:
        btn.clicked.connect(callback)
    return btn

def create_combo(items=None, callback=None, height=30):
    """Factory function for creating and configuring comboboxes
    
    Args:
        items: Optional list of items to add
        callback: Optional slot to connect currentTextChanged signal to
        height: Minimum height in pixels
        
    Returns:
        QComboBox: Configured combobox widget
    """
    combo = QComboBox()
    combo.setMinimumHeight(height)
    if items:
        combo.addItems(items)
    if callback:
        combo.currentTextChanged.connect(callback)
    return combo

# ============================================================================
# Main Window Class
# ============================================================================

class StreamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('S2S Audio Stream')
        self.resize(0, 0)
        self.setMinimumSize(0, 0)

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

    def create_horizontal_line(self):
        """Create a horizontal line separator"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _setup_global_hotkey(self):
        
        # Right Ctrl + F24 (keyboard uses 'right ctrl+f24')
        hotkey = 'right ctrl+f24'
        print(f"Registering global hotkey: {hotkey}")
        keyboard.add_hotkey(hotkey, self._global_toggle_stream)
        keyboard.wait()

    def _global_toggle_stream(self):
        self.stream_btn.click()

    def init_ui(self):
        # Stream button
        stream_layout = QHBoxLayout()
        self.stream_btn = create_button('Start Stream', self.toggle_stream, checkable=True, height=40)
        stream_layout.addWidget(self.stream_btn, 1) 

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel('Model:')
        self.model_combo = create_combo(callback=self.change_model)
        self.populate_models()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

        # STT model selection
        stt_model_layout = QHBoxLayout()
        stt_model_label = QLabel('STT Model:')
        self.stt_model_combo = create_combo(callback=self.change_stt_model)
        self.populate_stt_models()
        stt_model_layout.addWidget(stt_model_label)
        stt_model_layout.addWidget(self.stt_model_combo)

        # EQ Filters section (Lowpass and Highpass side by side)
        eq_buttons_layout = QHBoxLayout()
        self.lowpass_btn = create_button('Lowpass Filter: True', self.toggle_lowpass_filter, checkable=True, checked=True)
        self.highpass_btn = create_button('Highpass Filter: False', self.toggle_highpass_filter, checkable=True, checked=False)
        eq_buttons_layout.addWidget(self.lowpass_btn)
        eq_buttons_layout.addWidget(self.highpass_btn)

        # Parameters row
        eq_params_layout = QHBoxLayout()
        self.lowpass_cutoff_spin = create_spinbox(0, 10000, 100, 6000, " Hz", 0, self.change_lowpass_cutoff)
        self.lowpass_order_spin = create_spinbox(1, 10, 1, 2, "", 0, self.change_lowpass_order)
        self.highpass_cutoff_spin = create_spinbox(0, 10000, 10, 80, " Hz", 0, self.change_highpass_cutoff)
        self.highpass_order_spin = create_spinbox(1, 10, 1, 2, "", 0, self.change_highpass_order)
        eq_params_layout.addWidget(self.lowpass_cutoff_spin)
        eq_params_layout.addWidget(self.lowpass_order_spin)
        eq_params_layout.addWidget(self.highpass_cutoff_spin)
        eq_params_layout.addWidget(self.highpass_order_spin)

        # Voice basic settings
        voice_basic_layout = QHBoxLayout()
        self.voice_speed_spin = create_spinbox(0.0, 2.0, 0.01, 1.0, "x (speed)", 2, self.change_voice_speed)
        self.voice_pitch_spin = create_spinbox(-20.0, 20.0, 0.1, 0.0, " st (pitch)", 1, self.change_voice_pitch)
        voice_basic_layout.addWidget(self.voice_speed_spin)
        voice_basic_layout.addWidget(self.voice_pitch_spin)

        # Font selection
        font_layout = QHBoxLayout()
        font_label = QLabel('Subtitle Font:')
        self.font_combo = create_combo(callback=self.change_font)
        self.populate_fonts()
        font_layout.addWidget(font_label)
        font_layout.addWidget(self.font_combo)

        # Manual text input
        text_layout = QVBoxLayout()
        text_label = QLabel('Manual Text Input:')
        self.text_input = QTextEdit()
        self.text_input.setMinimumHeight(80)
        self.text_input.setMaximumHeight(120)
        self.text_input.setPlaceholderText('Enter text to synthesize...')
        self.synthesize_btn = create_button('Synthesize', self.synthesize_text)
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_input)
        text_layout.addWidget(self.synthesize_btn)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Model selection section
        layout.addLayout(model_layout)
        layout.addLayout(stt_model_layout)
        layout.addWidget(self.create_horizontal_line())
        
        # Font section
        layout.addLayout(font_layout)
        layout.addWidget(self.create_horizontal_line())
        
        # EQ Filters section (Lowpass and Highpass)
        layout.addLayout(eq_buttons_layout)
        layout.addLayout(eq_params_layout)
        layout.addWidget(self.create_horizontal_line())
        
        # Voice basic settings section
        layout.addLayout(voice_basic_layout)
        layout.addWidget(self.create_horizontal_line())
        
        # Manual text input section
        layout.addLayout(text_layout)
        layout.addWidget(self.create_horizontal_line())

        # Debug/Training Data Section
        debug_layout = QHBoxLayout()
        self.capture_training_data_checkbox = QCheckBox("Capture Training Data")
        self.capture_training_data_checkbox.stateChanged.connect(self.toggle_capture_training_data)
        debug_layout.addWidget(self.capture_training_data_checkbox)
        layout.addLayout(debug_layout)
        layout.addWidget(self.create_horizontal_line())

        layout.addLayout(stream_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_capture_training_data(self, state):
        """Toggle the capturing of training data."""
        enabled = state == 2  # 2 means checked
        set_capture_training_data(enabled)
        print(f"Training data capture {'enabled' if enabled else 'disabled'}.")
    
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


    def toggle_lowpass_filter(self):
        """Toggle the lowpass filter"""
        is_checked = self.lowpass_btn.isChecked()
        self.lowpass_btn.setText(f'Lowpass Filter: {str(is_checked)}')
        try:
            self.s2s.toggle_lowpass_filter(is_checked)
        except Exception as e:
            print(f"Error toggling lowpass filter: {e}")

    def change_lowpass_cutoff(self):
        """Change the lowpass cutoff frequency"""
        value = self.lowpass_cutoff_spin.value()
        try:
            self.s2s.change_lowpass_cutoff(value)
        except Exception as e:
            print(f"Error changing lowpass cutoff: {e}")

    def change_lowpass_order(self):
        """Change the lowpass filter order"""
        value = self.lowpass_order_spin.value()
        try:
            self.s2s.change_lowpass_order(value)
        except Exception as e:
            print(f"Error changing lowpass order: {e}")

    def toggle_highpass_filter(self):
        """Toggle the highpass filter"""
        is_checked = self.highpass_btn.isChecked()
        self.highpass_btn.setText(f'Highpass Filter: {str(is_checked)}')
        try:
            self.s2s.toggle_highpass_filter(is_checked)
        except Exception as e:
            print(f"Error toggling highpass filter: {e}")

    def change_highpass_cutoff(self):
        """Change the highpass cutoff frequency"""
        value = self.highpass_cutoff_spin.value()
        try:
            self.s2s.change_highpass_cutoff(value)
        except Exception as e:
            print(f"Error changing highpass cutoff: {e}")

    def change_highpass_order(self):
        """Change the highpass filter order"""
        value = self.highpass_order_spin.value()
        try:
            self.s2s.change_highpass_order(value)
        except Exception as e:
            print(f"Error changing highpass order: {e}")

    def change_voice_speed(self):
        value = self.voice_speed_spin.value()
        try:
            self.s2s.change_voice_speed(value)
        except Exception as e:
            print(f"Error changing voice speed:: {e}")
    
    def change_voice_pitch(self):
        value = self.voice_pitch_spin.value()
        try:
            self.s2s.change_voice_pitch(value)
        except Exception as e:
            print(f"Error changing voice pitch:: {e}")

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

    def populate_stt_models(self):
        """Populate the STT model dropdown with available models"""
        self.stt_model_combo.clear()
        for display_name, key in self.s2s.stt_models.items():
            self.stt_model_combo.addItem(display_name, key)
        
        # Set default active STT model
        active_model_value = self.s2s.active_stt
        # Find the key corresponding to the active model value
        active_model_display_name = [k for k, v in self.s2s.stt_models.items() if v == active_model_value]
        if active_model_display_name:
            for i in range(self.stt_model_combo.count()):
                if self.stt_model_combo.currentText() == active_model_display_name[0]:
                    self.stt_model_combo.setCurrentIndex(i)
                    break

    def change_stt_model(self):
        """Handle STT model selection change"""
        if self.stt_model_combo.currentText():
            try:
                model_display_name = self.stt_model_combo.currentText()
                self.s2s.set_stt_model(model_display_name)
            except Exception as e:
                print(f"Error changing STT model: {e}")
