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
        """Gets the model path by autodetecting .onnx files in the tts models folder"""
        # gather all .onnx files under the models folder
        found = []
        for root, dirs, files in os.walk(self.path):
            for fname in files:
                if fname.lower().endswith('.onnx'):
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, self.path)
                    found.append((full_path, rel_path, fname))

        if not found:
            raise FileNotFoundError("No TTS model found. Please ensure you have .onnx models in the .models/tts directory.")

        voice_path = None
        if model:
            model_lower = model.lower()
            # exact basename (without extension) match
            for full_path, rel_path, fname in found:
                if os.path.splitext(fname)[0].lower() == model_lower:
                    voice_path = full_path
                    break
            # exact relative path match
            if not voice_path:
                for full_path, rel_path, fname in found:
                    if rel_path.lower() == model_lower or rel_path.replace('\\','/').lower() == model_lower:
                        voice_path = full_path
                        break
            # substring match
            if not voice_path:
                for full_path, rel_path, fname in found:
                    if model_lower in fname.lower() or model_lower in rel_path.lower():
                        voice_path = full_path
                        break
        else:
            # try default from config if present
            default = self.cfg.get('default_voice')
            if default:
                return self.get_model_path(default)

            # fallback to first available model
            voice_path = found[0][0]

        if not voice_path:
            raise FileNotFoundError(f"No TTS model matching '{model}' found in {self.path}.")

        self.model_path = voice_path

    def get_all_models(self):
        """Detect all .onnx models in the tts models folder and return metadata list"""
        models_dir = self.path
        if not os.path.exists(models_dir):
            return []

        available_models = []
        for root, dirs, files in os.walk(models_dir):
            for fname in files:
                if fname.lower().endswith('.onnx'):
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, models_dir).replace('\\','/')
                    key = os.path.splitext(os.path.basename(fname))[0]
                    available_models.append({
                        'key': key,
                        'display_name': key,
                        'file_path': rel_path,
                        'language': {},
                        'name': key,
                        'quality': ''
                    })

        available_models.sort(key=lambda x: (x['name'], x['key']))
        return available_models