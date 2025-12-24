import numpy as np
import torch
import threading
import queue
from utils.config import load_config
from utils.audio import resample_audio
from utils.timing import Timing
from utils.database import Database
from src.audio.stream import AudioStream
from src.audio import effects
from src.models.vad import Vad
from src.models.stt import Stt
from src.models.tts import Tts

class S2s:

    def __init__(self, subtitle_window=None):
        print("Loading config...")
        self.cfg = load_config()

        print("Loading utilities...")
        self.timing = Timing()
        self.database = Database()
        self.subtitle_window = subtitle_window

        print("Loading VAD model...")
        self.vad  = Vad(self.cfg)

        print("Loading STT model...")
        self.stt = Stt(self.cfg)

        print("Loading TTS model...")
        self.tts = Tts(self.cfg)
        self.tts.load_model(None)

        print("Setup misc stuff...")
        self.stream = AudioStream(self.cfg, self.stream_callback)

        self.processing_thread = None
        self.input_queue = queue.Queue()

        self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])

        print("Everything loaded.")


    def change_conf(self, conf):
        """Handles config changes from the UI"""
        for category, values in conf.items():
            if category not in self.cfg:
                self.cfg[category] = {}
            for key, value in values.items():
                self.cfg[category][key] = value

    def set_tts_model(self, model_key):
        """Change TTS model - special case of changes from the UI"""
        print(f"Loading TTS model: {model_key}")
        self.tts.load_model(model_key)
        print(f"TTS model loaded: {model_key}")
    
    def get_all_tts_models(self):
        """Gets all the TTS models for the UI"""
        return self.tts.get_all_models()

    def start_stream(self):
        """Starts the audio stream and processing thread"""
        if self.processing_thread is not None:
            return
        
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()

        self.timing.set_stream_latency(self.stream.stream.latency)

        self.stream.start()
        print("Stream started.")

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

        print("Stream stopped.")

    def stream_callback(self, indata, outdata, frames, time, status):
        """Callback function for the Audio stream - input/output"""
        self.input_queue.put(indata.copy())

        buffer_len = len(self.output_buffer)
        if buffer_len >= frames:
            outdata[:] = self.output_buffer[:frames].reshape(outdata.shape)
            self.output_buffer = self.output_buffer[frames:]
        elif buffer_len > 0:
            outdata[:buffer_len] = self.output_buffer.reshape(-1, outdata.shape[1])
            outdata[buffer_len:] = 0
            self.output_buffer = np.array([], dtype=self.cfg["audio"]["dtype"])
        else:
            outdata[:] = 0

    def _processing_loop(self):
        """Main processing loop - threaded"""
        while self.stream.is_running.is_set():
            try:
                indata = self.input_queue.get(timeout=1)

                self.timing.start('complete')

                mono_audio_tensor = torch.from_numpy(indata[:, 0]).to(torch.float32)

                resampled = resample_audio(mono_audio_tensor, self.cfg["audio"]["samplerate"], self.cfg["vad"]["samplerate"]) 

                self.timing.start('vad')
                speech_audio, should_process = self.vad.process_chunk(resampled)
                self.timing.end('vad')

                text_for_summary = ""
                if should_process and speech_audio is not None:
                    resampled_spech = resample_audio(speech_audio, self.cfg["vad"]["samplerate"], self.cfg["audio"]["samplerate"])
                    text_for_summary = self._process_speech_chunk(resampled_spech) or ""

                self.timing.end('complete')
                self.timing.print_summary(text_for_summary.strip())

            except queue.Empty:
                continue

            except Exception as e:
                print(f"Error in processing loop: {e}")

    def _process_speech_chunk(self, speech_audio):
        """Process detected speech - transcribe and synthesize"""
        if speech_audio.shape[0] == 0:
            return ""
        
        audio_np = speech_audio.cpu.numpy()

        self.timing.start('stt')
        text, _ = self.stt.transcribe(audio_np)
        self.timing.end('stt')

        if text.strip():
            self.timing.start('training')
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
        
        synthesized = self.tts.synthesize(text)
        modified = self._apply_audio_effects(synthesized)
        self.output_buffer = np.concatenate([self.output_buffer, modified])
        
        if self.subtitle_window:
            self.subtitle_window.set_subtitle(text.strip())

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
