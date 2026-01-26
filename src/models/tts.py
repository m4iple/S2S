import os
import numpy as np
from piper import PiperVoice
import torch
import torchaudio
import onnxruntime as ort


class Tts:
    def __init__(self, config):
        self.cfg = config["tts"]
        self.model = None
        self.model_path = None
        self.path = '.models/tts'
        
        # PERSISTENT RESAMPLER
        self.resampler = None
        self.target_rate = self.cfg["samplerate"]
        self.device = torch.device("cuda" if self.cfg["use_cuda"] and torch.cuda.is_available() else "cpu")
    
    def load_model(self, model):
        """Loads tts model"""
        if not model:
            model = self.cfg["default_voice"]

        self.get_model_path(model)
        
        # Silence ONNX Warnings
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        
        print(f"[INFO] Loading TTS model on {self.device}")
        self.model = PiperVoice.load(
            self.model_path, 
            use_cuda=(self.device.type == "cuda"),
        )

        # --- DEBUG PRINTS ---
        model_rate = self.model.config.sample_rate
        print(f"[DEBUG] Model Native Rate: {model_rate} Hz")
        print(f"[DEBUG] Target Output Rate: {self.target_rate} Hz")
        # --------------------
        
        # Initialize GPU Resampler
        model_rate = self.model.config.sample_rate
        if model_rate != self.target_rate:
            print(f"[INFO] Initializing CUDA Resampler: {model_rate}Hz -> {self.target_rate}Hz")
            
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=model_rate, 
                new_freq=self.target_rate,
                dtype=torch.float32
            ).to(self.device) # <--- MOVES RESAMPLER TO GPU
        else:
            self.resampler = None

    def synthesize(self, text):
        """Synthesizes given text"""
        if not self.model:
            return np.array([], dtype=np.float32)


        wav_generator = self.model.synthesize(text)
        wav_bytes = b"".join(chunk.audio_int16_bytes for chunk in wav_generator)

        audio_int16 = np.frombuffer(wav_bytes, dtype=np.int16).copy()

        if self.resampler is None:
            return audio_int16.astype(np.float32) / 32767.0

        audio_tensor = torch.from_numpy(audio_int16).to(self.device).float()

        audio_tensor /= 32767.0

        resampled_tensor = self.resampler(audio_tensor)

        return resampled_tensor.cpu().numpy()

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