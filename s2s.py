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

from scipy.signal import butter, lfilter

def load_whisper_model():
    """Loads the Faster Whisper model."""
    # base tiny.en
    whisper_model = faster_whisper.WhisperModel(
        'large-v3-turbo', 
        device='cuda', 
        compute_type='float16'
    )
    return whisper_model

class S2S:
    def __init__(self):
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
        self.silence_threshold_frames = 5

        # --- Whisper settings ---
        self.text_model = load_whisper_model()
        print("Loading Piper TTS model...")
        default_voice = 'en_US-hfc_female-medium'
        voice_path = get_model_path(default_voice)
        print(f"Loading model from: {voice_path}")
        self.tts_model = PiperVoice.load(voice_path, use_cuda=True)
        print("All models loaded.")

        # --- Threads ---
        self.stream = None
        self.processing_thread = None

        # --- Voice modification settings --
        self.soft_voice = True
        self.auto_tume = False

    def chage_soft_voice(self, data):
        """Toggle the Soft Voice"""
        self.soft_voice = data
        
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
                                complete_time_start = time.time() # DEBUG Start Complete Timer
                                # reset the speach buffer and counter
                                audio_to_process = self.speech_audio_buffer.clone().cpu().numpy()
                                self.speech_audio_buffer = torch.empty(0)
                                self.silence_counter = 0
                                
                                text = self.whisper_transcribe(audio_to_process)
                                print(f"Transcribed: '{text.strip()}'")

                                if text.strip():
                                    self._synthesize_and_buffer_text(text)
                                    complete_time_end = time.time() # DEBUG End Complete Timer
                                    print(f"Complete synthesis took: {(complete_time_end - complete_time_start) * 1000:.2f} ms")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def whisper_transcribe(self, audio):
        stt_start_time = time.time() # DEBUG Start Whisper Timer
        segments, _ = self.text_model.transcribe(audio, beam_size=5) # TODO  DEBUG it seems the model is getting bad audio
        stt_end_time = time.time() # DEBUG End Whisper Timer
        print(f"Whisper transcription took: {(stt_end_time - stt_start_time) * 1000:.2f} ms")
        return "".join(segment.text for segment in segments)

    def resample_audio(self, audio_tensor, original_rate, target_rate):
        if original_rate == target_rate:
            return audio_tensor

        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio_tensor)

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

            self.tts_output_buffer = np.array([], dtype=self.dtype)
            
            print("Stream stopped.")

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
        
        print(f"Synthesizing speech with Piper: '{text.strip()}'")
        tts_start_time = time.time() # DEBUG start tts Timer
        
        wav_generator = self.tts_model.synthesize(text)

        # extract and combine the raw int16 bytes form the tts output
        wav_bytes = b"".join(chunk.audio_int16_bytes for chunk in wav_generator)
        
        audio_output_np = np.frombuffer(wav_bytes, dtype=np.int16).astype(np.float32, copy=False) / 32767.0

        tts_end_time = time.time() # DEBUG End tts Timer
        print(f"Piper TTS synthesis took: {(tts_end_time - tts_start_time) * 1000:.2f} ms")
        
        # Resample the audio back to the Stream sample rate
        resampled_for_output = self.resample_audio(torch.from_numpy(audio_output_np), self.tts_model.config.sample_rate, self.samplerate).cpu().numpy()

        output_ready = self.audio_modifications(resampled_for_output)

        self.tts_output_buffer = np.concatenate([self.tts_output_buffer, output_ready])

    def synthesize_text(self, text):
        """Synthesize text to speech and add to output buffer"""
        self._synthesize_and_buffer_text(text)

    def audio_modifications(self, audio):
        """modifies audio data"""
        mod_start_time = time.time() # DEBUG start mod Timer
        if self.soft_voice:
            soft_voice_start_time = time.time() # DEBUG start soft_voice Timer

            # Define a Butterworth low-pass filter
            def butter_lowpass(cutoff, fs, order=4):
                nyq = 0.5 * fs  # Nyquist frequency
                normal_cutoff = cutoff / nyq  # Normalized cutoff frequency
                b, a = butter(order, normal_cutoff, btype='low', analog=False)
                return b, a

            # Apply the low-pass filter to the audio data
            def lowpass_filter(data, cutoff, fs, order=4):
                b, a = butter_lowpass(cutoff, fs, order=order)
                y = lfilter(b, a, data)
                return y

            # Filter the audio to soften the voice (reduce high frequencies)
            audio = lowpass_filter(audio, cutoff=4000, fs=self.samplerate, order=4)

            soft_voice_end_time = time.time() # DEBUG End soft_voice Timer
            print(f"softmod took: {(soft_voice_end_time - soft_voice_start_time) * 1000:.2f} ms")
        if self.autotume:
            pass
            
        mod_end_time = time.time() # DEBUG End mod Timer
        print(f"Mod took: {(mod_end_time - mod_start_time) * 1000:.2f} ms")
        return audio
