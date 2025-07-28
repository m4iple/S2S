from PyQt6.QtWidgets import QMainWindow, QPushButton, QComboBox, QLabel, QDoubleSpinBox, QHBoxLayout, QVBoxLayout, QWidget, QLineEdit, QTextEdit
from s2s import S2S

class StreamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('S2S Audio Stream')
        self.s2s = S2S()
        self.init_ui()

    def init_ui(self):
        self.toggle_btn = QPushButton('Start Stream')
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.toggle_stream)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel('Model:')
        self.model_combo = QComboBox()
        self.populate_models()
        self.model_combo.currentTextChanged.connect(self.change_model)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)

        # Manual text input
        text_layout = QVBoxLayout()
        text_label = QLabel('Manual Text Input:')
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText('Enter text to synthesize...')
        self.text_input.returnPressed.connect(self.synthesize_text)  # Enter key support
        self.synthesize_btn = QPushButton('Synthesize')
        self.synthesize_btn.clicked.connect(self.synthesize_text)
        text_layout.addWidget(text_label)
        text_layout.addWidget(self.text_input)
        text_layout.addWidget(self.synthesize_btn)

        layout = QVBoxLayout()
        layout.addLayout(model_layout)
        layout.addLayout(text_layout)
        layout.addWidget(self.toggle_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def populate_models(self):
        """Populate the model dropdown with available models"""
        models = self.s2s.get_available_models()
        self.model_combo.clear()
        
        if not models:
            self.model_combo.addItem("No models available", None)
            return
        
        # Check if we have the new format (list of dicts) or old format (list of strings)
        if models and isinstance(models[0], dict):
            # New format with voices.json
            for model in models:
                display_name = model['display_name']
                key = model['key']
                self.model_combo.addItem(display_name, key)
            
            # Try to set current model as selected
            current_model_key = 'en_US-lessac-medium'  # Default
            for i in range(self.model_combo.count()):
                if self.model_combo.itemData(i) == current_model_key:
                    self.model_combo.setCurrentIndex(i)
                    break
        else:
            # Old format (fallback)
            for model in models:
                # Remove .onnx extension for display
                display_name = model.replace('.onnx', '') if isinstance(model, str) else str(model)
                self.model_combo.addItem(display_name, model)
            
            # Set current model as selected (default is the first one loaded)
            current_model = 'en_US-lessac-medium.onnx'
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
        text = self.text_input.text().strip()
        if text:
            try:
                self.s2s.synthesize_text(text)
                self.text_input.clear()  # Clear the input after synthesis
            except Exception as e:
                print(f"Error synthesizing text: {e}")

    def change_input(self):
        index = self.input_combo.currentIndex()
        self.s2s.set_input(index)

    def change_output(self):
        index = self.output_combo.currentIndex()
        self.s2s.set_output(index)

    def change_latency(self):
        index = self.latency_combo.currentIndex()
        self.s2s.set_latency(index)

    def change_samplerate(self):
        value = int(self.samplerate_spin.value())
        self.s2s.set_samplerate(value)

    def change_blocksize(self):
        value = int(self.blocksize_spin.value())
        self.s2s.set_blocksize(value)

    def change_dtype(self):
        dtype = self.dtype_combo.currentText()
        self.s2s.set_dtype(dtype)

    def change_channels(self):
        value = int(self.channels_spin.value())
        self.s2s.set_channels(value)

    def toggle_stream(self):
        if self.toggle_btn.isChecked():
            self.toggle_btn.setText('Stop Stream')
            self.s2s.start_stream()
        else:
            self.toggle_btn.setText('Start Stream')
            self.s2s.stop_stream()