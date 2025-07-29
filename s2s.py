import sounddevice as sd
import time
import torch
import torchaudio
from silero_vad import load_silero_vad
from piper import PiperVoice
import faster_whisper
import os
import numpy as np
import threading
import queue
import json
from scipy.signal import butter, lfilter

def load_whisper_model():
    """Loads the Faster Whisper model."""
    # base tiny.en
    whisper_model = faster_whisper.WhisperModel(
        'base', 
        device='cuda', 
        compute_type='float16'
    )
    return whisper_model

class S2S:
    def __init__(self):
        self.is_running = threading.Event()

        self.input_device_index = sd.default.device[0]
        self.output_device_index = sd.default.device[1]
        self.samplerate = 48000
        self.blocksize = 1024
        self.dtype = 'float32'
        self.channels = 1

        self.input_queue = queue.Queue()
        self.tts_output_buffer = np.array([], dtype=self.dtype)

        self.vad_model = load_silero_vad()
        self.vad_samplerate = 16000
        self.vad_speech_prob_threshold = 0.5
        self.vad_buffer_chunk_size = 512
        self.vad_audio_buffer = torch.empty(0)
        self.is_speaking = False
        self.speech_audio_buffer = torch.empty(0)
        self.silence_counter = 0
        self.silence_threshold_frames = 5

        self.device = "cuda"
        self.text_model = load_whisper_model()
        
        print("Loading Piper TTS model...")

        default_voice_key = 'en_US-hfc_female-medium'
        voice_path = None
        
        voices_json_path = '.models/voices.json'
        if os.path.exists(voices_json_path):
            try:
                with open(voices_json_path, 'r', encoding='utf-8') as f:
                    voices_data = json.load(f)
                
                if default_voice_key in voices_data:
                    voice_info = voices_data[default_voice_key]
                    for file_path in voice_info.get('files', {}).keys():
                        if file_path.endswith('.onnx'):
                            test_path = os.path.join('.models', file_path)
                            if os.path.exists(test_path):
                                voice_path = test_path
                                break
            except json.JSONDecodeError as e:
                print(f"Error reading voices.json: {e}")
        
        if not voice_path:
            raise FileNotFoundError(f"No TTS model found. Please ensure you have models in the .models directory.")
        
        print(f"Loading model from: {voice_path}")
        
        # Optimize ONNX Runtime for performance
        os.environ['ORT_ENABLE_CUDA_GRAPH'] = '1'  # Enable CUDA graph optimization
        os.environ['ORT_DISABLE_ALL_OPTIMIZATION'] = '0'  # Ensure optimizations are enabled
        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'  # Enable TensorRT cache if available
        
        self.tts_model = PiperVoice.load(voice_path, use_cuda=True)
        
        print("All models loaded.")

        # --- Threads ---
        self.stream = None
        self.processing_thread = None

        self.autotume = False
        self.softmod = True

    def chage_softmod(self, data):
        self.softmod = data
        
    def audio_callback(self, indata, outdata, frames, time, status):
        if status:
            print(status)
        
        self.input_queue.put(indata.copy())

        buffer_len = len(self.tts_output_buffer)
        if buffer_len >= frames:
            outdata[:] = self.tts_output_buffer[:frames].reshape(outdata.shape)
            self.tts_output_buffer = self.tts_output_buffer[frames:]
        else:
            outdata[:buffer_len] = self.tts_output_buffer.reshape(-1, 1)
            outdata[buffer_len:] = 0
            self.tts_output_buffer = np.array([], dtype=self.dtype)

    def _processing_loop(self):
        while self.is_running.is_set():
            try:
                indata = self.input_queue.get(timeout=1)
                
                audio_tensor = torch.from_numpy(indata[:, 0]).to(torch.float32)
                resampled_for_vad = self.resample_audio(audio_tensor, self.samplerate, self.vad_samplerate)
                self.vad_audio_buffer = torch.cat([self.vad_audio_buffer, resampled_for_vad])

                while self.vad_audio_buffer.shape[0] >= self.vad_buffer_chunk_size:
                    chunk = self.vad_audio_buffer[:self.vad_buffer_chunk_size]
                    self.vad_audio_buffer = self.vad_audio_buffer[self.vad_buffer_chunk_size:]
                    
                    speech_prob = self.vad_model(chunk, self.vad_samplerate).item()

                    if speech_prob > self.vad_speech_prob_threshold:
                        self.silence_counter = 0
                        if not self.is_speaking:
                            self.is_speaking = True
                        self.speech_audio_buffer = torch.cat([self.speech_audio_buffer, chunk])
                    else:
                        if self.is_speaking:
                            self.silence_counter += 1
                            if self.silence_counter > self.silence_threshold_frames:
                                self.is_speaking = False
                                complete_time_start = time.time()
                                audio_to_process = self.speech_audio_buffer.clone().cpu().numpy()
                                self.speech_audio_buffer = torch.empty(0)
                                self.silence_counter = 0
                                
                                text = self.whisper_transcribe(audio_to_process)
                                print(f"Transcribed: '{text.strip()}'")

                                if text.strip():
                                    self._synthesize_and_buffer_text(text)
                                    complete_time_end = time.time()
                                    print(f"Complete synthesis took: {(complete_time_end - complete_time_start) * 1000:.2f} ms")
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def whisper_transcribe(self, audio):
        stt_start_time = time.time()
        segments, _ = self.text_model.transcribe(audio, beam_size=5)
        stt_end_time = time.time()
        print(f"Whisper transcription took: {(stt_end_time - stt_start_time) * 1000:.2f} ms")
        return "".join(segment.text for segment in segments)

    def resample_audio(self, audio_tensor, original_rate, target_rate):
        if original_rate == target_rate:
            return audio_tensor

        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio_tensor)

    def start_stream(self):
        if self.stream is None:
            try:
                self.is_running.set()
                self.processing_thread = threading.Thread(target=self._processing_loop)
                self.processing_thread.start()

                self.stream = sd.Stream(
                    device=(self.input_device_index, self.output_device_index),
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    dtype=self.dtype,
                    channels=self.channels,
                    callback=self.audio_callback,
                )
                self.stream.start()
                print("Stream started.")
            except Exception as e:
                print(f"Error starting stream: {e}")

    def stop_stream(self):
        if self.stream is not None:
            self.is_running.clear()
            
            if self.processing_thread:
                self.processing_thread.join(timeout=2)
                self.processing_thread = None

            self.stream.stop()
            self.stream.close()
            self.stream = None

            self.tts_output_buffer = np.array([], dtype=self.dtype)
            
            print("Stream stopped.")

    def get_available_models(self):
        """Get list of available TTS models from voices.json"""
        voices_json_path = '.models/voices.json'
        if not os.path.exists(voices_json_path):
            models_dir = '.models'
            if not os.path.exists(models_dir):
                return []
            
            models = []
            for file in os.listdir(models_dir):
                if file.endswith('.onnx'):
                    models.append(file)
            return models
        
        try:
            with open(voices_json_path, 'r', encoding='utf-8') as f:
                voices_data = json.load(f)
            
            available_models = []
            for voice_key, voice_info in voices_data.items():
                onnx_file_path = None
                for file_path in voice_info.get('files', {}).keys():
                    if file_path.endswith('.onnx'):
                        full_path = os.path.join('.models', file_path)
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
            print(f"Error reading voices.json: {e}")
            return []

    def set_model(self, model_key_or_filename):
        """Switch to a different TTS model"""
        voice_path = None
        
        voices_json_path = '.models/voices.json'
        if os.path.exists(voices_json_path):
            try:
                with open(voices_json_path, 'r', encoding='utf-8') as f:
                    voices_data = json.load(f)
                
                if model_key_or_filename in voices_data:
                    voice_info = voices_data[model_key_or_filename]
                    for file_path in voice_info.get('files', {}).keys():
                        if file_path.endswith('.onnx'):
                            voice_path = os.path.join('.models', file_path)
                            if os.path.exists(voice_path):
                                break
                    
                    if not voice_path:
                        raise FileNotFoundError(f"ONNX file not found for voice: {model_key_or_filename}")
                else:
                    voice_path = os.path.join('.models', model_key_or_filename)
                    
            except json.JSONDecodeError as e:
                print(f"Error reading voices.json: {e}")
                voice_path = os.path.join('.models', model_key_or_filename)
        else:
            voice_path = os.path.join('.models', model_key_or_filename)
        
        if not voice_path or not os.path.exists(voice_path):
            raise FileNotFoundError(f"Model not found: {model_key_or_filename}")
        
        print(f"Loading model: {model_key_or_filename}")
        self.tts_model = PiperVoice.load(voice_path, use_cuda=True)
        
        self.tts_resampler = torchaudio.transforms.Resample(
            orig_freq=self.tts_model.config.sample_rate,
            new_freq=self.samplerate
        ) if self.tts_model.config.sample_rate != self.samplerate else None
        
        print(f"Model loaded: {model_key_or_filename}")

    def _synthesize_and_buffer_text(self, text):
        """Internal method to synthesize text and add to output buffer"""
        if not text.strip():
            return
        
        print(f"Synthesizing speech with Piper: '{text.strip()}'")
        tts_start_time = time.time()
        
        wav_generator = self.tts_model.synthesize(text)
        wav_bytes = b"".join(chunk.audio_int16_bytes for chunk in wav_generator)
        
        audio_output_np = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32, copy=False) / 32767.0

        tts_end_time = time.time()
        print(f"Piper TTS synthesis took: {(tts_end_time - tts_start_time) * 1000:.2f} ms")
        
        resampled_for_output = self.resample_audio(
            torch.from_numpy(audio_output_np), 
            self.tts_model.config.sample_rate, 
            self.samplerate
        ).cpu().numpy()
        output_ready = self.audio_modifications(resampled_for_output)
        self.tts_output_buffer = np.concatenate([self.tts_output_buffer, output_ready])

    def synthesize_text(self, text):
        """Synthesize text to speech and add to output buffer"""
        self._synthesize_and_buffer_text(text)

    def audio_modifications(self, audio):
        mod_start_time = time.time()
        if self.softmod:
            softmod_start_time = time.time()
            def butter_lowpass(cutoff, fs, order=4):
                nyq = 0.5 * fs
                normal_cutoff = cutoff / nyq
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a
            def lowpass_filter(data, cutoff, fs, order=4):
                b, a = butter_lowpass(cutoff, fs, order=order)
                y = lfilter(b, a, data)
                return y
            audio = lowpass_filter(audio, cutoff=4000, fs=self.samplerate, order=4)
            softmod_end_time = time.time()
            print(f"softmod took: {(softmod_end_time - softmod_start_time) * 1000:.2f} ms")
        if self.autotume:
            pass
            
        mod_end_time = time.time()
        print(f"Mod took: {(mod_end_time - mod_start_time) * 1000:.2f} ms")
        return audio
