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


        self.latency_spin = QDoubleSpinBox()
        self.latency_spin.setDecimals(2)
        self.latency_spin.setSingleStep(0.01)
        self.latency_spin.setMinimum(0.01)
        self.latency_spin.setMaximum(1.0)
        self.latency_spin.setValue(float(self.s2s.get_current_latency()))
        self.latency_spin.valueChanged.connect(self.change_latency)

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
        row2.addWidget(self.latency_spin)
        layout.addLayout(row2)

        layout.addWidget(self.toggle_btn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def change_input(self):
        index = self.input_combo.currentIndex()
        self.s2s.set_new_input(index)

    def change_output(self):
        index = self.output_combo.currentIndex()
        self.s2s.set_new_output(index)

    def change_latency(self):
        latency = self.latency_spin.value()
        self.s2s.set_latency(latency)
        

    def toggle_stream(self):
        if self.toggle_btn.isChecked():
            self.toggle_btn.setText('Stop Stream')
            self.input_combo.setDisabled(True)
            self.output_combo.setDisabled(True)
            self.s2s.start_stream()
        else:
            self.toggle_btn.setText('Start Stream')
            self.input_combo.setDisabled(False)
            self.output_combo.setDisabled(False)
            self.s2s.end_stream()