import sounddevice as sd
import time
import torch
import torchaudio
from silero_vad import load_silero_vad
from piper import PiperVoice
import faster_whisper
import numpy as np
import threading
import queue
import os
from model_functions import get_model_path
from debug import start_timer, end_timer, print_timing_summary

from scipy.signal import resample

def load_whisper_model():
    """Loads the Faster Whisper model."""
    # base tiny.en
    whisper_model = faster_whisper.WhisperModel(
        'distil-large-v3', 
        device = 'cuda', 
        compute_type = 'float16',
        num_workers = 4
    )
    return whisper_model

class S2S:
    def __init__(self, subtitle_window=None):
        # ---  Onix Settings ---
        os.environ['ORT_ENABLE_CUDA_GRAPH'] = '1'  # Enable CUDA graph optimization
        os.environ['ORT_DISABLE_ALL_OPTIMIZATION'] = '0'  # Ensure optimizations are enabled
        os.environ['ORT_TENSORRT_ENGINE_CACHE_ENABLE'] = '1'  # Enable TensorRT cache if available
        
        # -- Global ---
        self.is_running = threading.Event()
        
        # --- Input/Output Device Info ---
        self.input_device_index = sd.default.device[0]
        self.output_device_index = sd.default.device[1]
        self.samplerate = 48000
        self.blocksize = 1024
        self.dtype = 'float32'
        self.channels = 1

        # --- Audio queues / buffers ---
        self.input_queue = queue.Queue()
        self.tts_output_buffer = np.array([], dtype=self.dtype)

        # -- VAD settings ---
        self.vad_model = load_silero_vad()
        self.vad_samplerate = 16000
        self.vad_speech_prob_threshold = 0.5
        self.vad_buffer_chunk_size = 512
        self.vad_audio_buffer = torch.empty(0)
        self.is_speaking = False
        self.speech_audio_buffer = torch.empty(0)
        self.silence_counter = 0
        self.silence_threshold_frames = 3 # x * 512 / 16000 = 96ms

        # --- stt settings ---
        self.text_model = load_whisper_model()
        self.whisper_prompt = "The quick brown fox jumps over the lazy dog. She sells seashells by the seashore. Charles and Philip watched the game with zeal."

        # --- Piper settings ---
        print("Loading Piper TTS model...")
        default_voice = 'en_US-hfc_female-medium'
        voice_path = get_model_path(default_voice)
        print(f"Loading model from: {voice_path}")
        self.tts_model = PiperVoice.load(voice_path, use_cuda=True)
        print("All models loaded.")

        # --- Threads ---
        self.stream = None
        self.processing_thread = None
        
        # --- Monitoring device ---
        self.monitoring_device_index = 22
        self.monitoring_stream = None

        # --- Voice modification settings --
        self.voice_soft = True
        self.voice_speed = 1.0
        self.voice_tune = False

        # --- Subtitle UI ---
        self.subtitle_window = subtitle_window

    def chage_voice_soft(self, data):
        """Toggle the Soft Voice"""
        self.voice_soft = data

    def change_voice_speed(self, data):
        """Change the Speed of the TTS Voice"""
        self.voice_speed = data

    def set_debug_mode(self, enabled):
        """Enable or disable debug timing"""
        from debug import set_debug
        set_debug(enabled)

    def start_stream(self):
        """Start the audio Stream"""
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
                
                # Initialize monitoring stream for device 22
                try:
                    self.monitoring_stream = sd.OutputStream(
                        device=self.monitoring_device_index,
                        samplerate=self.samplerate,
                        blocksize=self.blocksize,
                        dtype=self.dtype,
                        channels=self.channels,
                    )
                    self.monitoring_stream.start()
                    print(f"Monitoring stream started on device {self.monitoring_device_index}.")
                except Exception as e:
                    print(f"Warning: Could not start monitoring stream on device {self.monitoring_device_index}: {e}")
                    self.monitoring_stream = None
                
                print("Stream started.")
            except Exception as e:
                print(f"Error starting stream: {e}")

    def stop_stream(self):
        """Ends the audio Stream"""
        if self.stream is not None:
            self.is_running.clear()
            
            if self.processing_thread:
                self.processing_thread.join(timeout=2)
                self.processing_thread = None

            self.stream.stop()
            self.stream.close()
            self.stream = None
            
            # Stop and close monitoring stream
            if self.monitoring_stream is not None:
                try:
                    self.monitoring_stream.stop()
                    self.monitoring_stream.close()
                    self.monitoring_stream = None
                    print("Monitoring stream stopped.")
                except Exception as e:
                    print(f"Error stopping monitoring stream: {e}")

            self.tts_output_buffer = np.array([], dtype=self.dtype)
            
            print("Stream stopped.")

            # Clear subtitle when stream ends
            if self.subtitle_window:
                 self.subtitle_window.clear_subtitle()
        
    def audio_callback(self, indata, outdata, frames, time, status):
        """Stream audio callback"""
        # copy stream
        self.input_queue.put(indata.copy())

        # Set the output buffer lengh to the stream lenght (to avoid mismatch error error)
        buffer_len = len(self.tts_output_buffer)
        if buffer_len >= frames:
            outdata[:] = self.tts_output_buffer[:frames].reshape(outdata.shape)
            self.tts_output_buffer = self.tts_output_buffer[frames:]
        elif buffer_len > 0:
            outdata[:buffer_len] = self.tts_output_buffer.reshape(-1, outdata.shape[1])
            outdata[buffer_len:] = 0
            self.tts_output_buffer = np.array([], dtype=self.dtype)
        else:
            outdata[:] = 0

        # duplicate output to device 22 for monitoring
        if hasattr(self, 'monitoring_stream') and self.monitoring_stream is not None:
            try:
                # Send the same audio data to device 22 for monitoring
                self.monitoring_stream.write(outdata.copy())
            except Exception as e:
                print(f"Error writing to monitoring device: {e}")


    def _processing_loop(self):
        """Processing thread"""
        while self.is_running.is_set():
            try:
                # This line retrieves the next chunk of audio data from the input queue, waiting up to one second for new data to arrive.
                indata = self.input_queue.get(timeout=1)
                
                # convers a 2D Numpy array to an torch tensor
                audio_tensor = torch.from_numpy(indata[:, 0]).to(torch.float32)

                # Resample for the VAD
                resampled_for_vad = self.resample_audio(audio_tensor, self.samplerate, self.vad_samplerate)
                # Combine old and new data
                self.vad_audio_buffer = torch.cat([self.vad_audio_buffer, resampled_for_vad])
                # Checks if the audio is long enogh for the VAD
                while self.vad_audio_buffer.shape[0] >= self.vad_buffer_chunk_size:
                    # Cut out a Chuck of data
                    chunk = self.vad_audio_buffer[:self.vad_buffer_chunk_size]
                    self.vad_audio_buffer = self.vad_audio_buffer[self.vad_buffer_chunk_size:]
                    
                    # Runs the VAD
                    speech_prob = self.vad_model(chunk, self.vad_samplerate).item()

                    # Check if Speaking
                    if speech_prob > self.vad_speech_prob_threshold:
                        self.silence_counter = 0
                        if not self.is_speaking:
                            self.is_speaking = True
                        # Combine only the speech chunks
                        self.speech_audio_buffer = torch.cat([self.speech_audio_buffer, chunk])
                    else:
                        if self.is_speaking:
                            self.silence_counter += 1
                            # Waits for the Silence to be longer than the threshold
                            if self.silence_counter > self.silence_threshold_frames:
                                self.is_speaking = False
                                start_timer('complete')
                                
                                # reset the speach buffer and counter
                                start_timer('buffer_prep')
                                audio_to_process = self.speech_audio_buffer.clone().cpu().numpy()
                                self.speech_audio_buffer = torch.empty(0)
                                self.silence_counter = 0
                                end_timer('buffer_prep')
                                
                                text = self.whisper_transcribe(audio_to_process)

                                if text.strip():
                                    start_timer('synthesis_total')
                                    self._synthesize_and_buffer_text(text)
                                    end_timer('synthesis_total')
                                
                                end_timer('complete')
                                print_timing_summary(text.strip())
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def whisper_transcribe(self, audio):
        start_timer('stt')

        segments, _ = self.text_model.transcribe(
            audio, 
            language = "en", 
            beam_size = 1,
            word_timestamps = True,
            initial_prompt = self.whisper_prompt
        )

        full_text = []

        for segment in segments:
            segment_text = segment.text
            
            full_text.append(segment_text)
        
        result = "".join(full_text).strip()
        
        end_timer('stt')
        return result

    def resample_audio(self, audio_tensor, original_rate, target_rate):
        if original_rate == target_rate:
            return audio_tensor

        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio_tensor)

    def set_model(self, model):
        """Switch to a different TTS model"""
        voice_path = get_model_path(model)
        
        print(f"Loading model: {model}")
        self.tts_model = PiperVoice.load(voice_path, use_cuda=True)
        
        print(f"Model loaded: {model}")

    def _synthesize_and_buffer_text(self, text):
        """Internal method to synthesize text and add to output buffer"""
        if not text.strip():
            return

        start_timer('tts')
        
        wav_generator = self.tts_model.synthesize(text)

        # extract and combine the raw int16 bytes form the tts output
        wav_bytes = b"".join(chunk.audio_int16_bytes for chunk in wav_generator)
        
        audio_output_np = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32, copy=False) / 32767.0

        end_timer('tts')
        
        # Resample the audio back to the Stream sample rate
        start_timer('resample')
        resampled_for_output = self.resample_audio(torch.from_numpy(audio_output_np), self.tts_model.config.sample_rate, self.samplerate).cpu().numpy()
        end_timer('resample')

        output_ready = self.audio_modifications(resampled_for_output)
       
        start_timer('buffer_ops')
        self.tts_output_buffer = np.concatenate([self.tts_output_buffer, output_ready])
        end_timer('buffer_ops')

        if self.subtitle_window:
            self.subtitle_window.set_subtitle(text.strip())

    def synthesize_text(self, text):
        """Synthesize text to speech and add to output buffer"""
        self._synthesize_and_buffer_text(text)

    def audio_modifications(self, audio):
        """modifies audio data"""
        start_timer('audio_mod')
        if self.voice_soft:
            pass
        
        if self.voice_tune:
            pass

        new_length = int(len(audio) / self.voice_speed)
        audio = resample(audio, new_length)
            
        end_timer('audio_mod')
        return audio
