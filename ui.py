from PyQt6.QtWidgets import QMainWindow, QPushButton, QComboBox, QLabel, QDoubleSpinBox, QHBoxLayout, QVBoxLayout, QWidget
from s2s import S2S

class StreamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('S2S Audio Stream')
        self.s2s = S2S()
        self.init_ui()

    def init_ui(self):
        devices = self.s2s.get_audio_devices()
        default_input = self.s2s.get_current_input()
        default_output = self.s2s.get_current_output()

        self.input_devices = [d['name'] for d in devices]
        self.output_devices = [d['name'] for d in devices]
        self.input_combo = QComboBox()
        self.input_combo.addItems(self.input_devices)
        self.input_combo.setCurrentIndex(default_input)
        self.input_combo.currentIndexChanged.connect(self.change_input)
        self.output_combo = QComboBox()
        self.output_combo.addItems(self.output_devices)
        self.output_combo.setCurrentIndex(default_output)
        self.output_combo.currentIndexChanged.connect(self.change_output)

        self.latency_combo = QComboBox()
        self.latency_combo.addItems(['Low', 'High'])
        self.latency_combo.setCurrentIndex(0)
        self.latency_combo.currentIndexChanged.connect(self.change_latency)

        # Samplerate
        self.samplerate_spin = QDoubleSpinBox()
        self.samplerate_spin.setRange(8000, 192000)
        self.samplerate_spin.setValue(self.s2s.get_current_samplerate())
        self.samplerate_spin.setSingleStep(1000)
        self.samplerate_spin.valueChanged.connect(self.change_samplerate)

        # Blocksize
        self.blocksize_spin = QDoubleSpinBox()
        self.blocksize_spin.setRange(16, 8192)
        self.blocksize_spin.setValue(self.s2s.get_current_blocksize())
        self.blocksize_spin.setSingleStep(16)
        self.blocksize_spin.valueChanged.connect(self.change_blocksize)

        # Dtype
        self.dtype_combo = QComboBox()
        dtypes = ['float32', 'int16', 'int32', 'uint8']
        self.dtype_combo.addItems(dtypes)
        current_dtype = str(self.s2s.get_current_dtype())
        if current_dtype in dtypes:
            self.dtype_combo.setCurrentText(current_dtype)
        self.dtype_combo.currentTextChanged.connect(self.change_dtype)

        # Channels
        self.channels_spin = QDoubleSpinBox()
        self.channels_spin.setRange(1, 16)
        self.channels_spin.setValue(self.s2s.get_current_channels())
        self.channels_spin.setSingleStep(1)
        self.channels_spin.valueChanged.connect(self.change_channels)

        self.toggle_btn = QPushButton('Start Stream')
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.clicked.connect(self.toggle_stream)

        layout = QVBoxLayout()
        row1 = QHBoxLayout()
        row1.addWidget(QLabel('Input Device:'))
        row1.addWidget(self.input_combo)
        row1.addWidget(QLabel('Output Device:'))
        row1.addWidget(self.output_combo)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel('Latency:'))
        row2.addWidget(self.latency_combo)
        row2.addWidget(QLabel('Samplerate:'))
        row2.addWidget(self.samplerate_spin)
        row2.addWidget(QLabel('Blocksize:'))
        row2.addWidget(self.blocksize_spin)
        layout.addLayout(row2)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel('Dtype:'))
        row4.addWidget(self.dtype_combo)
        row4.addWidget(QLabel('Channels:'))
        row4.addWidget(self.channels_spin)
        layout.addLayout(row4)

        layout.addWidget(self.toggle_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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
            self.input_combo.setDisabled(True)
            self.output_combo.setDisabled(True)
            self.latency_combo.setDisabled(True)
            self.samplerate_spin.setDisabled(True)
            self.blocksize_spin.setDisabled(True)
            self.dtype_combo.setDisabled(True)
            self.channels_spin.setDisabled(True)
            self.s2s.start_stream()
        else:
            self.toggle_btn.setText('Start Stream')
            self.input_combo.setDisabled(False)
            self.output_combo.setDisabled(False)
            self.latency_combo.setDisabled(False)
            self.samplerate_spin.setDisabled(False)
            self.blocksize_spin.setDisabled(False)
            self.dtype_combo.setDisabled(False)
            self.channels_spin.setDisabled(False)
            self.s2s.end_stream()