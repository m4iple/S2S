import sounddevice as sd
import torch
import torchaudio
from silero_vad import load_silero_vad
from piper import PiperVoice
import faster_whisper
import numpy as np
import threading
import queue
import pyrubberband as rb
import onnxruntime
from scipy.signal import butter, lfilter
from model_functions import get_model_path
from debug import start_timer, end_timer, print_timing_summary, save_training_data

class S2S:
    def __init__(self, subtitle_window=None):
        # ---  Onix Settings ---
        sess_options = onnxruntime.SessionOptions()

        # Set optimization level
        # ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3

        
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
        self.pre_speech_buffer = torch.empty(0)
        self.pre_speech_frames = 5  # Number of frames to keep before speech detection
        self.post_speech_silence_frames = 2  # Number of silence frames to add after speech ends
        # Maximum audio buffer size in frames before forcing transcription. 1.5 seconds * 16000 Hz.
        self.max_buffer_frames = int(1.5 * self.vad_samplerate)
        self.process_now = False

        self.stt_models = {
            "Whisper (Distilled)": "whisper",
        }
        self.active_stt = "whisper"
        # --- stt settings ---
        print("Loading the ASR Model ...")
        self.text_model_whisper = self.load_whisper_model()
        # The initial prompt helps the model recognize specific words or names.
        self.whisper_prompt = ""

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
        self.monitoring_device_index = 22 # <- hardcoded device change! to see all devices (python -m sounddevice)
        self.monitoring_stream = None

        # --- Voice lowpass filter settings ---
        self.lowpass_enabled = True
        self.lowpass_cutoff = 6000
        self.lowpass_order = 2

        # --- Voice highpass filter settings ---
        self.highpass_enabled = False
        self.highpass_cutoff = 80
        self.highpass_order = 2

        # --- Voice Basic settings ---
        self.voice_speed = 1.0
        self.voice_pitch = 0.0

        # --- Subtitle UI ---
        self.subtitle_window = subtitle_window

        # --- Pay around ---
        self.audio_copy = None

    def toggle_lowpass_filter(self, data):
        """Toggle the Lowpass Filter"""
        self.lowpass_enabled = data

    def change_lowpass_cutoff(self, data):
        """Change the Lowpass Filter Cutoff"""
        self.lowpass_cutoff = data
        
    def change_lowpass_order(self, data):
        """Change the Lowpass Filter Order"""
        self.lowpass_order = data

    def toggle_highpass_filter(self, data):
        """Toggle the Highpass Filter"""
        self.highpass_enabled = data

    def change_highpass_cutoff(self, data):
        """Change the Highpass Filter Cutoff"""
        self.highpass_cutoff = data
        
    def change_highpass_order(self, data):
        """Change the Highpass Filter Order"""
        self.highpass_order = data

    def change_voice_speed(self, data):
        """Change the Speed of the TTS Voice"""
        self.voice_speed = data
    
    def change_voice_pitch(self, data):
        """Change the pitch of the TTS Voice"""
        self.voice_pitch = data

    def load_whisper_model(self):
        """Loads the Faster Whisper model. For cpu set device to 'cpu' and set compute_type to 'int8'."""
        whisper_model = faster_whisper.WhisperModel(
            'distil-small.en', 
            device = 'cuda', 
            compute_type = 'float16'
        )
        return whisper_model

        

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

    def _process_speech_chunk(self, is_final_chunk=False):
        """
        Transcribes the speech audio buffer, synthesizes the text, and updates the buffer.
        It intelligently handles chunking for long sentences to maintain low latency.
        """
        if self.speech_audio_buffer.shape[0] == 0:
            return

        audio_to_process = self.speech_audio_buffer.clone().cpu().numpy()
        self.audio_copy = self.speech_audio_buffer.clone().cpu().numpy()

        start_timer('complete')
        if self.active_stt == "whisper":
            text, last_word_end_time = self.whisper_transcribe(audio_to_process)
        else:
            # Handle cases where the STT model is not whisper or is unset
            print(f"Warning: Active STT model '{self.active_stt}' is not supported. No transcription will be performed.")
            text = ""
            last_word_end_time = 0

        if text.strip():
            start_timer('training')
            save_training_data(text.strip(), audio_to_process)
            end_timer('training')
            start_timer('synthesis_total')
            self._synthesize_and_buffer_text(text)
            end_timer('synthesis_total')
        
        end_timer('complete')
        print_timing_summary(text.strip(), True, self.tts_output_buffer)

        self.speech_audio_buffer = torch.empty(0)

        if self.active_stt == "whisper":
            if is_final_chunk:
               self.speech_audio_buffer = torch.empty(0)
            else:
                # Convert the end time of the last word to a frame index
                last_frame = int(last_word_end_time * self.vad_samplerate)
                # Trim the buffer to keep only the audio that hasn't been transcribed
                if last_frame < self.speech_audio_buffer.shape[0]:
                    self.speech_audio_buffer = self.speech_audio_buffer[last_frame:]
                else:
                    # This can happen if whisper processes the whole chunk
                    self.speech_audio_buffer = torch.empty(0)

    def _processing_loop(self):
        """Processing thread"""
        while self.is_running.is_set():
            try:
                # Get audio data from input queue
                indata = self.input_queue.get(timeout=1)
                # Convert to tensor and extract mono channel
                audio_tensor = torch.from_numpy(indata[:, 0]).to(torch.float32)
                # Resample audio to VAD model's expected sample rate
                resampled_for_vad = self.resample_audio(audio_tensor, self.samplerate, self.vad_samplerate)
                # Add new audio to VAD buffer
                self.vad_audio_buffer = torch.cat([self.vad_audio_buffer, resampled_for_vad])

                # Process VAD buffer in chunks
                while self.vad_audio_buffer.shape[0] >= self.vad_buffer_chunk_size:
                    # Extract chunk for VAD analysis
                    chunk = self.vad_audio_buffer[:self.vad_buffer_chunk_size]
                    self.vad_audio_buffer = self.vad_audio_buffer[self.vad_buffer_chunk_size:]
                    # Get speech probability from VAD model
                    speech_prob = self.vad_model(chunk, self.vad_samplerate).item()

                    # Maintain pre-speech buffer for context
                    self.pre_speech_buffer = torch.cat([self.pre_speech_buffer, chunk])
                    if self.pre_speech_buffer.shape[0] > self.pre_speech_frames * self.vad_buffer_chunk_size:
                        frames_to_remove = self.pre_speech_buffer.shape[0] - (self.pre_speech_frames * self.vad_buffer_chunk_size)
                        self.pre_speech_buffer = self.pre_speech_buffer[frames_to_remove:]

                    # Handle speech detection
                    if speech_prob > self.vad_speech_prob_threshold:
                        self.silence_counter = 0
                        # Start of speech detection
                        if not self.is_speaking:
                            self.is_speaking = True
                            # Add pre-speech context to speech buffer
                            self.speech_audio_buffer = torch.cat([self.speech_audio_buffer, self.pre_speech_buffer])
                        # Add current chunk to speech buffer
                        self.speech_audio_buffer = torch.cat([self.speech_audio_buffer, chunk])

                        # If buffer is too long, process a chunk of it without waiting for silence
                        if self.speech_audio_buffer.shape[0] > self.max_buffer_frames:
                            self.process_now = True

                    # Handle silence detection
                    else:
                        if self.is_speaking:
                            # Add silence frames after speech for context
                            if self.silence_counter < self.post_speech_silence_frames:
                                self.speech_audio_buffer = torch.cat([self.speech_audio_buffer, chunk])
                            
                            # Count consecutive silence frames
                            self.silence_counter += 1
                            # End of speech detection
                            if self.silence_counter > self.silence_threshold_frames + self.post_speech_silence_frames or self.process_now:
                                self.is_speaking = False
                                self.process_now = False
                                # Process the final chunk of speech
                                self._process_speech_chunk(is_final_chunk=True)
                                self.silence_counter = 0
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")

    def whisper_transcribe(self, audio):
        """Transcribes audio and returns the text and the end time of the last word."""
        start_timer('stt')
        segments, _ = self.text_model_whisper.transcribe(
            audio, 
            language="en", 
            beam_size=1,
            word_timestamps=True,
            vad_filter=False,
            initial_prompt=self.whisper_prompt
        )
        end_timer('stt')

        start_timer('stt_text')
        full_text = []
        last_word_end_time = 0
        for segment in segments:
            full_text.append(segment.text)
            if segment.words:
                last_word_end_time = segment.words[-1].end
        end_timer('stt_text')
        
        result = "".join(full_text).strip()
        
        return result, last_word_end_time
    

    def resample_audio(self, audio_tensor, original_rate, target_rate):
        if original_rate == target_rate:
            return audio_tensor

        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio_tensor)

    def set_stt_model(self, model_key):
        """Switch to a different STT model"""
        if model_key in self.stt_models:
            self.active_stt = self.stt_models[model_key]
            print(f"STT model switched to: {self.active_stt}")
        else:
            print(f"Warning: STT model key '{model_key}' not found.")

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
        modified_audio = audio

        if self.voice_pitch != 0.0:
            modified_audio = rb.pitch_shift(modified_audio, self.samplerate, self.voice_pitch)

        if self.voice_speed != 1.0:
            modified_audio = rb.time_stretch(modified_audio, self.samplerate, self.voice_speed)

        if self.lowpass_enabled:
            modified_audio = self.lowpass_filter(modified_audio)
        
        if self.highpass_enabled:
            modified_audio = self.highpass_filter(modified_audio)
            
        end_timer('audio_mod')
        return modified_audio
    
    def lowpass_filter(self, audio):
        """Apply a lowpass filter to the audio"""
        # Nyquist frequency is the highest possible frequency that can be accurately captured and reproduced at a given sample rate.
        nyquist = 0.5 * self.samplerate
        normal_cutoff = self.lowpass_cutoff / nyquist
        b, a = butter(self.lowpass_order, normal_cutoff, btype='low', analog=False)

        # Apply the filter
        filtered_audio = lfilter(b, a, audio)

        return filtered_audio.astype(np.float32)
    
    def highpass_filter(self, audio):
        """Apply a highpass filter to the audio"""
        # Nyquist frequency is the highest possible frequency that can be accurately captured and reproduced at a given sample rate.
        nyquist = 0.5 * self.samplerate
        normal_cutoff = self.highpass_cutoff / nyquist
        # Clamp to valid range (0, 1)
        normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999)
        b, a = butter(self.highpass_order, normal_cutoff, btype='high', analog=False)

        # Apply the filter
        filtered_audio = lfilter(b, a, audio)

        return filtered_audio.astype(np.float32)