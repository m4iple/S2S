from PyQt6.QtWidgets import QMainWindow, QPushButton, QComboBox, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QTextEdit, QDoubleSpinBox, QFrame, QSpinBox, QCheckBox
from PyQt6.QtGui import QIcon, QFont, QFontDatabase
from PyQt6.QtCore import Qt
import os
import sounddevice as sd
from utils.config import load_config


class S2sWindow(QMainWindow):
    """Main window for S2S audio stream application"""
    
    FIELD_DEFINITIONS = {
        'model': {
            'type': 'combo',
            'label': 'Model:',
            'handler': 'set_model',
            'population_method': 'populate_tts_models'
        },
        'lowpass_btn': {
            'type': 'button',
            'label': 'Lowpass Filter:',
            'config_id': 'filters.lowpass.enabled',
            'checkable': True,
            'checked': True,
            'handler': 'toggle_filter',
            'filter_type': 'lowpass'
        },
        'lowpass_cutoff_spin': {
            'type': 'spinbox',
            'min': 0,
            'max': 10000,
            'step': 100,
            'value': 6000,
            'suffix': ' Hz',
            'decimals': 0,
            'config_id': 'filters.lowpass.cutoff',
            'handler': 'change_conf'
        },
        'lowpass_order_spin': {
            'type': 'spinbox',
            'min': 1,
            'max': 10,
            'step': 1,
            'value': 2,
            'suffix': '',
            'decimals': 0,
            'config_id': 'filters.lowpass.order',
            'handler': 'change_conf'
        },
        'highpass_btn': {
            'type': 'button',
            'label': 'Highpass Filter:',
            'config_id': 'filters.highpass.enabled',
            'checkable': True,
            'checked': False,
            'handler': 'toggle_filter',
            'filter_type': 'highpass'
        },
        'highpass_cutoff_spin': {
            'type': 'spinbox',
            'min': 0,
            'max': 10000,
            'step': 10,
            'value': 80,
            'suffix': ' Hz',
            'decimals': 0,
            'config_id': 'filters.highpass.cutoff',
            'handler': 'change_conf'
        },
        'highpass_order_spin': {
            'type': 'spinbox',
            'min': 1,
            'max': 10,
            'step': 1,
            'value': 2,
            'suffix': '',
            'decimals': 0,
            'config_id': 'filters.highpass.order',
            'handler': 'change_conf'
        },
        'voice_speed_spin': {
            'type': 'spinbox',
            'min': 0.0,
            'max': 2.0,
            'step': 0.01,
            'value': 1.0,
            'suffix': 'x (speed)',
            'decimals': 2,
            'config_id': 'voice_processing.speed',
            'handler': 'change_conf'
        },
        'voice_pitch_spin': {
            'type': 'spinbox',
            'min': -20.0,
            'max': 20.0,
            'step': 0.1,
            'value': 0.0,
            'suffix': ' st (pitch)',
            'decimals': 1,
            'config_id': 'voice_processing.pitch',
            'handler': 'change_conf'
        },
        'font_combo': {
            'type': 'combo',
            'label': 'Subtitle Font:',
            'handler': 'change_font',
            'population_method': 'populate_fonts'
        },
        'text_input': {
            'type': 'textedit',
            'label': 'Manual Text Input:',
            'placeholder': 'Enter text to synthesize...',
        },
        'synthesize_btn': {
            'type': 'button',
            'label': 'Synthesize',
            'handler': 'synthesize_text'
        },
        'capture_training_data_checkbox': {
            'type': 'checkbox',
            'label': 'Capture Training Data',
            'config_id': 'training.capture',
            'handler': 'change_conf'
        },
        'api_enabled_checkbox': {
            'type': 'checkbox',
            'label': 'Enable API Server',
            'handler': 'toggle_api_server'
        },
        'audio_devices_btn': {
            'type': 'button',
            'label': 'List Audio Devices',
            'handler': 'print_audio_devices'
        },
        'stream_btn': {
            'type': 'button',
            'label': 'Start Stream',
            'checkable': True,
            'checked': False,
            'handler': 'toggle_stream',
            'height': 40
        },
    }
    
    SECTIONS = [
        {
            'title': None,
            'fields': ['audio_devices_btn', 'model'],
            'divider': True
        },
        {
            'title': None,
            'fields': ['font_combo'],
            'divider': True
        },
        {
            'title': None,
            'layout': 'rows',
            'rows': [
                ['lowpass_btn', 'highpass_btn'],
                ['lowpass_cutoff_spin', 'lowpass_order_spin', 'highpass_cutoff_spin', 'highpass_order_spin']
            ],
            'divider': True
        },
        {
            'title': None,
            'fields': ['voice_speed_spin', 'voice_pitch_spin'],
            'divider': True
        },
        {
            'title': 'Manual Text Input:',
            'fields': ['text_input', 'synthesize_btn'],
            'divider': True
        },
        {
            'title': None,
            'fields': ['capture_training_data_checkbox'],
            'divider': True
        },
        {
            'title': None,
            'fields': ['api_enabled_checkbox'],
            'divider': True
        },
        {
            'title': None,
            'fields': ['stream_btn'],
            'divider': False
        }
    ]
    
    HANDLERS = {}
    
    def __init__(self, s2s_instance, subtitle_window=None, subtitle_controller=None):
        super().__init__()
        
        # Load configuration
        self.cfg = load_config("configs/s2s.toml")
        
        # Apply window configuration
        window_cfg = self.cfg.get('window', {})
        self.setWindowTitle(window_cfg.get('title', 'S2S Audio Stream'))
        
        width = window_cfg.get('width', 0)
        height = window_cfg.get('height', 0)
        self.resize(width, height)
        
        min_width = window_cfg.get('min_width', 0)
        min_height = window_cfg.get('min_height', 0)
        self.setMinimumSize(min_width, min_height)
        
        self.setWindowIcon(QIcon("icon.png"))
        
        # Set window flags
        if window_cfg.get('always_on_top', False):
            self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        
        # Position window
        self._position_window(window_cfg.get('position', 'default'))
        
        self.s2s = s2s_instance
        self.subtitle_window = subtitle_window
        self.subtitle_controller = subtitle_controller
        self.ui_elements = {}
        
        self._init_handlers()
        self.init_ui()
    
    def _init_handlers(self):
        """Register all handlers"""
        self.HANDLERS = {
            'change_conf': self._handle_change_conf,
            'set_model': self._handle_set_model,
            'set_stt_model': self._handle_set_stt_model,
            'toggle_filter': self._handle_toggle_filter,
            'change_font': self._handle_change_font,
            'synthesize_text': self._handle_synthesize_text,
            'toggle_stream': self._handle_toggle_stream,
            'toggle_capture_training_data': self._handle_toggle_training_data,
            'toggle_api_server': self._handle_toggle_api_server,
            'print_audio_devices': self._handle_print_audio_devices,
        }
    
    def _position_window(self, position):
        """Position window based on config"""
        if position == 'default':
            return  # Let OS decide
        
        screen = self.screen()
        if not screen:
            return
        
        screen_geometry = screen.geometry()
        window_width = self.width() if self.width() > 0 else 400
        window_height = self.height() if self.height() > 0 else 400
        
        if position == 'center':
            x = (screen_geometry.width() - window_width) // 2
            y = (screen_geometry.height() - window_height) // 2
            self.move(x, y)
        elif position == 'top-left':
            self.move(0, 0)
        elif position == 'top-right':
            self.move(screen_geometry.width() - window_width, 0)
        elif position == 'bottom-left':
            self.move(0, screen_geometry.height() - window_height)
        elif position == 'bottom-right':
            self.move(screen_geometry.width() - window_width, 
                     screen_geometry.height() - window_height)
    
    def create_horizontal_line(self):
        """Create a horizontal line separator"""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line
    
    def _create_spinbox(self, field_def):
        """Create a spinbox from field definition"""
        spin = QDoubleSpinBox()
        spin.setMinimum(field_def.get('min', 0))
        spin.setMaximum(field_def.get('max', 100))
        spin.setSingleStep(field_def.get('step', 1))
        spin.setValue(field_def.get('value', 0))
        spin.setDecimals(field_def.get('decimals', 0))
        
        suffix = field_def.get('suffix', '')
        if suffix:
            spin.setSuffix(suffix)
        
        spin.setMinimumHeight(35)
        return spin
    
    def _create_button(self, field_def):
        """Create a button from field definition"""
        btn = QPushButton(field_def.get('label', 'Button'))
        btn.setCheckable(field_def.get('checkable', False))
        btn.setChecked(field_def.get('checked', False))
        btn.setMinimumHeight(field_def.get('height', 35))
        return btn
    
    def _create_combo(self, field_def):
        """Create a combobox from field definition"""
        combo = QComboBox()
        combo.setMinimumHeight(30)
        return combo
    
    def _create_textedit(self, field_def):
        """Create a text edit from field definition"""
        text_edit = QTextEdit()
        text_edit.setMinimumHeight(80)
        text_edit.setMaximumHeight(120)
        placeholder = field_def.get('placeholder', '')
        if placeholder:
            text_edit.setPlaceholderText(placeholder)
        return text_edit
    
    def _create_checkbox(self, field_def):
        """Create a checkbox from field definition"""
        return QCheckBox(field_def.get('label', 'Checkbox'))
    
    def _create_ui_element(self, field_def):
        """Factory method to create UI element based on type"""
        element_type = field_def.get('type')
        
        creators = {
            'spinbox': self._create_spinbox,
            'button': self._create_button,
            'combo': self._create_combo,
            'textedit': self._create_textedit,
            'checkbox': self._create_checkbox,
        }
        
        creator = creators.get(element_type)
        if not creator:
            raise ValueError(f"Unknown field type: {element_type}")
        
        return creator(field_def)
    
    def _connect_handler(self, field_name, field_def, element):
        """Connect handler based on field definition and element type"""
        handler_name = field_def.get('handler')
        if not handler_name or handler_name not in self.HANDLERS:
            return
        
        handler_func = self.HANDLERS[handler_name]
        element_type = field_def.get('type')
        
        if element_type == 'spinbox':
            element.valueChanged.connect(lambda value: handler_func(field_name, field_def, value))
        elif element_type == 'combo':
            element.currentTextChanged.connect(lambda text: handler_func(field_name, field_def, text))
        elif element_type == 'button':
            element.clicked.connect(lambda checked: handler_func(field_name, field_def, element.isChecked() if field_def.get('checkable') else None))
        elif element_type == 'checkbox':
            element.stateChanged.connect(lambda state: handler_func(field_name, field_def, state))
    
    def _handle_change_conf(self, field_name, field_def, value):
        """Handle generic config changes"""
        config_id = field_def.get('config_id')
        if config_id and hasattr(self.s2s, 'change_conf'):
            self.s2s.change_conf((config_id, value))
    
    def _handle_set_model(self, field_name, field_def, value):
        """Handle TTS model changes"""
        element = self.ui_elements[field_name]
        if element.currentData() and hasattr(self.s2s, 'set_tts_model'):
            self.s2s.set_tts_model(element.currentData())
    
    def _handle_set_stt_model(self, field_name, field_def, value):
        """Handle STT model changes"""
        if value and hasattr(self.s2s, 'set_stt_model'):
            self.s2s.set_stt_model(value)
    
    def _handle_toggle_filter(self, field_name, field_def, is_checked):
        """Handle filter toggle"""
        filter_type = field_def.get('filter_type')
        config_id = field_def.get('config_id')
        element = self.ui_elements[field_name]
        
        element.setText(f'{filter_type.capitalize()} Filter: {str(is_checked)}')
        if config_id and hasattr(self.s2s, 'change_conf'):
            self.s2s.change_conf((config_id, is_checked))
    
    def _handle_change_font(self, field_name, field_def, value):
        """Handle font changes"""
        if value and self.subtitle_window:
            self.subtitle_window.change_font(value)
    
    def _handle_synthesize_text(self, field_name, field_def, value):
        """Handle synthesize button"""
        text = self.ui_elements['text_input'].toPlainText().strip()
        if text and hasattr(self.s2s, 'synthesize_text'):
            self.s2s.synthesize_text(text)
            self.ui_elements['text_input'].clear()
    
    def _handle_toggle_stream(self, field_name, field_def, is_checked):
        """Handle stream toggle"""
        stream_btn = self.ui_elements['stream_btn']
        if is_checked:
            stream_btn.setText('Stop Stream')
            if hasattr(self.s2s, 'start_stream'):
                self.s2s.start_stream()
        else:
            stream_btn.setText('Start Stream')
            if hasattr(self.s2s, 'stop_stream'):
                self.s2s.stop_stream()
            # Clear subtitles when stopping stream
            if self.subtitle_controller:
                self.subtitle_controller.clear_subtitle()
    
    def _handle_toggle_training_data(self, field_name, field_def, state):
        """Handle training data capture toggle"""
        enabled = state == 2  # 2 means checked
        if hasattr(self.s2s, 'toggle_capture_training_data'):
            self.s2s.toggle_capture_training_data(enabled)
    
    def _handle_toggle_api_server(self, field_name, field_def, state):
        """Handle API server toggle"""
        enabled = state == 2  # 2 means checked
        if hasattr(self.s2s, 'toggle_api'):
            self.s2s.toggle_api(enabled)

    def _handle_print_audio_devices(self, field_name, field_def, value):
        """Handle audio devices button click"""
        print("\n" + "="*80)
        print("AUDIO DEVICES")
        print("="*80)
        
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                print(f"\nDevice {i}: {device['name']}")
                print(f"  Input Channels: {device['max_input_channels']}")
                print(f"  Output Channels: {device['max_output_channels']}")
                print(f"  Sample Rate: {device['default_samplerate']} Hz")
            
            print("\n" + "="*80)
            print(f"Default Input Device: {sd.default.device[0]}")
            print(f"Default Output Device: {sd.default.device[1]}")
            print("="*80 + "\n")
        except Exception as e:
            print(f"Error querying audio devices: {e}")
    
    
    def populate_tts_models(self):
        """Populate TTS model dropdown"""
        model_combo = self.ui_elements.get('model')
        
        if model_combo is None:
            print("[ERROR] model_combo is None!")
            return
        
        if not hasattr(self.s2s, 'get_all_tts_models'):
            print("[ERROR] s2s doesn't have get_all_tts_models method!")
            return
        
        models = self.s2s.get_all_tts_models()
        
        model_combo.clear()
        
        if not models:
            model_combo.addItem("No models available", None)
            return
        
        if models and isinstance(models[0], dict):
            for model in models:
                display_name = model.get('display_name', 'Unknown')
                key = model.get('key')
                model_combo.addItem(display_name, key)
            
            current_model_key = 'en_US-hfc_female-medium'
            for i in range(model_combo.count()):
                if model_combo.itemData(i) == current_model_key:
                    model_combo.setCurrentIndex(i)
                    break
        else:
            for model in models:
                display_name = model.replace('.onnx', '') if isinstance(model, str) else str(model)
                model_combo.addItem(display_name, model)
            
            current_model = 'en_US-hfc_female-medium'
            for i in range(model_combo.count()):
                if model_combo.itemData(i) == current_model:
                    model_combo.setCurrentIndex(i)
                    break
    
    def populate_fonts(self):
        """Populate font dropdown from .fonts folder"""
        font_combo = self.ui_elements.get('font_combo')
        
        if font_combo is None:
            print("[ERROR] font_combo is None!")
            return
        
        font_combo.clear()
        # Go up one directory from ui/ to project root
        fonts_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.fonts')
        
        if os.path.isdir(fonts_folder):
            font_files = [f for f in os.listdir(fonts_folder) if f.lower().endswith(('.ttf', '.otf'))]
            
            for file in font_files:
                font_path = os.path.join(fonts_folder, file)
                ids = QFontDatabase.addApplicationFont(font_path)
                if ids != -1:
                    loaded_fonts = QFontDatabase.applicationFontFamilies(ids)
                    for loaded_font in loaded_fonts:
                        font_combo.addItem(loaded_font)
        else:
            font_combo.addItem("No fonts folder found")
    
    def init_ui(self):
        """Initialize the main UI layout from SECTIONS definition"""
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        for section in self.SECTIONS:
            if section.get('title'):
                layout.addWidget(QLabel(section['title']))
            
            if section.get('layout') == 'rows':
                for row in section.get('rows', []):
                    row_layout = QHBoxLayout()
                    for field_name in row:
                        element = self._create_and_setup_field(field_name)
                        row_layout.addWidget(element)
                    layout.addLayout(row_layout)

            else:
                fields = section.get('fields', [])
                
                has_label = section.get('title') is not None
                
                for field_name in fields:
                    field_def = self.FIELD_DEFINITIONS.get(field_name)
                    if not field_def:
                        continue
                    
                    element = self._create_and_setup_field(field_name)
                    
                    if field_def.get('type') == 'textedit' and has_label:
                        layout.addWidget(element)
                    
                    elif field_def.get('label') and field_def.get('type') != 'textedit':
                        field_layout = QHBoxLayout()
                        field_layout.addWidget(element)
                        layout.addLayout(field_layout)

                    else:
                        if field_def.get('type') == 'button' and field_name == 'stream_btn':
                            stream_layout = QHBoxLayout()
                            stream_layout.addWidget(element, 1)
                            layout.addLayout(stream_layout)
                        else:
                            layout.addWidget(element)
            
            if section.get('divider'):
                layout.addWidget(self.create_horizontal_line())
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def _create_and_setup_field(self, field_name):
        """Create a field, store it, and connect its handler"""
        field_def = self.FIELD_DEFINITIONS.get(field_name)
        if not field_def:
            return None
        
        element = self._create_ui_element(field_def)
        self.ui_elements[field_name] = element
        
        self._connect_handler(field_name, field_def, element)

        population_method = field_def.get('population_method')
        if population_method and hasattr(self, population_method):
            getattr(self, population_method)()
        
        return element
    
    def closeEvent(self, event):
        """Handle window close event - cleanup subtitle controller and window"""
        try:
            # Stop the stream if it's running
            if hasattr(self.s2s, 'stop_stream'):
                self.s2s.stop_stream()
            
            # Stop the subtitle controller thread
            if self.subtitle_controller:
                self.subtitle_controller.stop()
            
            # Close the subtitle window
            if self.subtitle_window:
                self.subtitle_window.close()
            
        except Exception as e:
            print(f"[ERROR] during cleanup: {e}")
        
        # Accept the close event
        event.accept()