import numpy as np
import torch
import torchaudio
import threading
import queue
from utils.config import load_config
from utils.timing import Timing
from utils.database import Database
from src.audio.stream import AudioStream
from src.audio import effects
from src.models.vad import Vad
from src.models.stt import Stt
from src.models.tts import Tts
from src.api import ApiServer

class S2s:

    def __init__(self, subtitle_window=None):
        print("[INFO] Loading config...")
        self.cfg = load_config()
        
        # Load API config separately and merge it
        try:
            api_cfg = load_config("configs/api.toml")
            self.cfg["api"] = api_cfg
        except FileNotFoundError:
            print("[WARN] No api.toml found, API features will be unavailable")

        print("[INFO] Loading utilities...")
        self.timing = Timing()
        self.database = Database()
        self.database.open()
        self.subtitle_window = subtitle_window

        print("[INFO] Loading VAD model...")
        self.vad  = Vad(self.cfg)

        print("[INFO] Loading STT model...")
        self.stt = Stt(self.cfg)

        print("[INFO] Loading TTS model...")
        self.tts = Tts(self.cfg)
        self.tts.load_model(None)

        print("[INFO] Setup misc stuff...")
        self.stream = AudioStream(self.cfg, self.stream_callback)

        self.processing_thread = None
        self.input_queue = queue.Queue()

        self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])
        
        print("[INFO] Initializing Resampler...")
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.cfg["audio"]["samplerate"], 
            new_freq=self.cfg["vad"]["samplerate"]
        )
        self.raw_input_buffer = torch.empty(0)

        print("[INFO] Setting up API server...")
        self.api = ApiServer(self)
        if self.cfg.get("api", {}).get("enabled", False):
            self.start_api()

        print("[INFO] Everything loaded.")


    def change_conf(self, conf):
        """Handles config changes from the UI"""
        conf_id, value = conf
        keys = conf_id.split('.')
        config_ref = self.cfg
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]

        config_ref[keys[-1]] = value
    

    def set_tts_model(self, model_key):
        """Change TTS model - special case of changes from the UI"""
        print(f"[INFO] Loading TTS model: {model_key}")
        self.tts.load_model(model_key)
        print(f"[INFO] TTS model loaded: {model_key}")
    
    def get_all_tts_models(self):
        """Gets all the TTS models for the UI"""
        return self.tts.get_all_models()

    def start_stream(self):
        """Starts the audio stream and processing thread"""
        if self.processing_thread is not None:
            return

        self.stream.start()
        
        if self.stream.stream is not None:
            self.timing.set_stream_latency(self.stream.stream.latency)
        
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        print("[INFO] Stream started.")

    def stop_stream(self):
        """Stopps the audio stream and processing thread, and clears the subtitles"""
        if self.processing_thread is None:
            return
        
        self.stream.stop()

        if self.processing_thread:
            self.processing_thread.join(timeout=2)
            self.processing_thread = None

        self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])
        self.vad.reset()
        
        if self.subtitle_window:
            self.subtitle_window.clear()

        print("[INFO] Stream stopped.")

    def stream_callback(self, indata, outdata, frames, time, status):
        """Callback function for the Audio stream - input/output"""
        self.input_queue.put(indata.copy())

        channels = outdata.shape[1]
        buffer_len = len(self.output_buffer)
        samples_needed = frames * channels
        
        if buffer_len >= samples_needed:
            samples = self.output_buffer[:samples_needed].reshape(frames, channels)
            outdata[:] = samples
            self.output_buffer = self.output_buffer[samples_needed:]
        elif buffer_len > 0:
            available_frames = buffer_len // channels
            remainder = buffer_len % channels
            
            if available_frames > 0:
                samples = self.output_buffer[:available_frames * channels].reshape(available_frames, channels)
                outdata[:available_frames] = samples
            
            outdata[available_frames:] = 0
            self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])
        else:
            outdata[:] = 0

    def _processing_loop(self):
        """Main processing loop - threaded"""

        while self.stream.is_running.is_set():
            try:
                indata = self.input_queue.get(timeout=0.1)

                self.timing.start('complete')

                self.timing.start('tensor_ops')
                mono_audio_tensor = torch.from_numpy(indata[:, 0]).to(torch.float32)
                
                self.raw_input_buffer = torch.cat([self.raw_input_buffer, mono_audio_tensor])
                self.timing.end('tensor_ops')

                if self.raw_input_buffer.shape[0] >= self.cfg["audio"]["resample_buffer_threshold"]:
                    
                    self.timing.start('resample')
                    resampled = self.resampler(self.raw_input_buffer)
                    self.raw_input_buffer = torch.empty(0)
                    self.timing.end('resample') 

                    self.timing.start('vad')
                    speech_audio, should_process = self.vad.process_chunk(resampled)
                    self.timing.end('vad')

                    text_for_summary = ""
                    if should_process and speech_audio is not None:
                        text_for_summary = self._process_speech_chunk(speech_audio) or ""

                    self.timing.end('complete')

                    if text_for_summary.strip():
                        self.timing.print_summary(text_for_summary.strip())

            except queue.Empty:
                continue

            except Exception as e:
                import traceback
                print(f"[ERROR] Error in processing loop: {e}")
                print(traceback.format_exc())

    def _process_speech_chunk(self, speech_audio):
        """Process detected speech - transcribe and synthesize"""
        if speech_audio.shape[0] == 0:
            return ""
        
        self.timing.start('cpu_ops')
        audio_np = speech_audio.cpu().numpy()
        self.timing.end('cpu_ops')

        self.timing.start('stt')
        text, _ = self.stt.transcribe(audio_np)
        self.timing.end('stt')

        if text.strip():
            
            self.timing.start('training')
            if self.cfg["training"]["capture"]:
                self.database.insert_training_data(text.strip(), audio_np)
            self.timing.end('training')

            self.timing.start('tts')
            synthesized = self.tts.synthesize(text)
            self.timing.end('tts')

            self.timing.start('audio_mod')
            modified = self._apply_audio_effects(synthesized)
            self.timing.end("audio_mod")

            self.timing.start('buffer_ops')
            self.output_buffer = np.concatenate([self.output_buffer, modified])
            self.timing.end('buffer_ops')

            if self.subtitle_window:
                self.subtitle_window.set(text.strip())
        
        return text

    def synthesize_text(self, text):
        """Manual synthesize text"""
        if not text.strip():
            return
        
        #self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])
        synthesized = self.tts.synthesize(text)
        modified = self._apply_audio_effects(synthesized)
        self.output_buffer = np.concatenate([self.output_buffer, modified])
        
        if self.subtitle_window:
            self.subtitle_window.set_subtitle(text.strip())
    
    def synthesize_via_api(self, text, model=None, font=None, color=None):
        """Synthesize text via API with optional temporary TTS model and subtitle customization"""
        if not text.strip():
            return
        
        interrupt_enabled = self.cfg.get("api", {}).get("interrupt_on_api_request", True)
        if interrupt_enabled:
            self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])
        
        if model:
            temp_tts = Tts(self.cfg)
            temp_tts.load_model(model)
            synthesized = temp_tts.synthesize(text)
        else:
            synthesized = self.tts.synthesize(text)
        
        modified = self._apply_audio_effects(synthesized)
        self.output_buffer = np.concatenate([self.output_buffer, modified])
        
        if self.subtitle_window:
            self.subtitle_window.set_subtitle(text.strip(), font, color)
    
    def start_api(self):
        """Start the API server"""
        host = self.cfg.get("api", {}).get("host", "127.0.0.1")
        port = self.cfg.get("api", {}).get("port", 5050)
        self.api.start(host, port)
    
    def stop_api(self):
        """Stop the API server"""
        self.api.stop()
    
    def toggle_api(self, enabled):
        """Toggle API server on/off"""
        if enabled:
            self.start_api()
        else:
            self.stop_api()

    def _apply_audio_effects(self, audio):
        """Apply effects to audio"""
        modified = audio

        if self.cfg["voice_processing"]["pitch"] != 0.0:
            modified = effects.pitch(modified, self.cfg["audio"]["samplerate"], self.cfg["voice_processing"]["pitch"])

        if self.cfg["voice_processing"]["speed"] != 1.0:
            modified = effects.speed(modified, self.cfg["audio"]["samplerate"], self.cfg["voice_processing"]["speed"])

        if self.cfg["filters"]["lowpass"]["enabled"]:
            modified = effects.lowpass_filter(modified, self.cfg["audio"]["samplerate"], self.cfg["filters"]["lowpass"])

        if self.cfg["filters"]["highpass"]["enabled"]:
            modified = effects.highpass_filter(modified, self.cfg["audio"]["samplerate"], self.cfg["filters"]["highpass"])

        return modified
