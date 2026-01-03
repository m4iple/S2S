import numpy as np
from piper import PiperVoice
import torch
from utils.audio import resample_audio
import os
import json

class Tts:
    def __init__(self, config):
        self.cfg = config["tts"]
        self.model = None
        self.model_path = None
        self.path = '.models/tts'
    
    def load_model(self, model):
        """Loads tts model"""
        if not model:
            model = self.cfg["default_voice"]

        self.get_model_path(model)
        self.model = PiperVoice.load(self.model_path, use_cuda=self.cfg["use_cuda"])

    def synthesize(self, text):
        """Synthesizes given text"""
        wav_generator = self.model.synthesize(text)
        wav_bytes = b"".join(chunk.audio_int16_bytes for chunk in wav_generator)

        audio_output_np = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32, copy=False) / 32767.0

        resampled = resample_audio(torch.from_numpy(audio_output_np), self.model.config.sample_rate, self.cfg["samplerate"]).cpu().numpy()
        return resampled

    def get_model_path(self, model):
        """Gets the model path from an key"""
        voice_path = None
        voices_json_path = self.path + '/voices.json'
        if os.path.exists(voices_json_path):
            try:
                with open(voices_json_path, 'r', encoding='utf-8') as f:
                    voices_data = json.load(f)
                
                if model in voices_data:
                    voice_info = voices_data[model]
                    for file_path in voice_info.get('files', {}).keys():
                        if file_path.endswith('.onnx'):
                            test_path = os.path.join(self.path, file_path)
                            if os.path.exists(test_path):
                                voice_path = test_path
                                break
            except json.JSONDecodeError as e:
                print(f"[ERROR] reading voices.json: {e}")
        
        if not voice_path:
            raise FileNotFoundError(f"No TTS model found. Please ensure you have models in the .models directory.")
    
        self.model_path = voice_path

    def get_all_models(self):
        """Gets all the modles form the voices.json"""
        voices_json_path = self.path + '/voices.json'
        if not os.path.exists(voices_json_path):
            models_dir = self.path
            if not os.path.exists(models_dir):
                return []
        
        try:
            with open(voices_json_path, 'r', encoding='utf-8') as f:
                voices_data = json.load(f)
            
            available_models = []
            for voice_key, voice_info in voices_data.items():
                onnx_file_path = None
                for file_path in voice_info.get('files', {}).keys():
                    if file_path.endswith('.onnx'):
                        full_path = os.path.join(self.path, file_path)
                        if os.path.exists(full_path):
                            onnx_file_path = file_path
                            break
                
                if onnx_file_path:
                    language = voice_info.get('language', {})
                    name = voice_info.get('name', voice_key)
                    quality = voice_info.get('quality', '')
                    
                    display_name = f"{language.get('name_english', language.get('code', ''))} {language.get('region', '')} - {name}"
                    if quality:
                        display_name += f" ({quality})"
                    
                    available_models.append({
                        'key': voice_key,
                        'display_name': display_name,
                        'file_path': onnx_file_path,
                        'language': language,
                        'name': name,
                        'quality': quality
                    })

            available_models.sort(key=lambda x: (x['language'].get('name_english', ''), x['language'].get('region', ''), x['name'], x['quality']))
            return available_models
            
        except Exception as e:
            print(f"[ERROR] reading voices.json: {e}")
            return []