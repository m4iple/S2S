import sounddevice as sd
import time
import torch
import torchaudio
from silero_vad import load_silero_vad

class S2S:
    def __init__(self):
        self.debug_printend = False
        self.stream = None
        self.input = sd.default.device[0] #71
        self.output = sd.default.device[1] #51
        self.samplerate = sd.default.samplerate if sd.default.samplerate is not None else 44100
        self.blocksize = 1024
        self.dtype = sd.default.dtype
        self.latency = ['low', 'low']
        self.channels = sd.default.channels
        if self.channels is None or self.channels == [None, None]:
            self.channels = [1, 1]
        else:
            self.channels = [c if c is not None else 1 for c in self.channels]

        self.vad_model = load_silero_vad()
        self.vad_samplerate = 16000
        self.vad_speech_prob = 0.5
        self.vad_buffer_lenght = 512
        self.vad_audio_buffer = torch.tensor([])

        self.speech_audio_buffer = torch.tensor([])


    def get_audio_devices(self):
        return sd.query_devices()

    def get_current_input(self):
        return self.input

    def get_current_output(self):
        return self.output

    def get_current_samplerate(self):
        return self.samplerate

    def get_current_blocksize(self):
        return self.blocksize

    def get_current_dtype(self):
        return self.dtype[0]

    def get_current_latency(self):
        return self.latency[0]

    def get_current_channels(self):
        return self.channels[0]

    def set_input(self, input):
        self.input = input

    def set_output(self, output):
        self.output = output

    def set_samplerate(self, samplerate):
        self.samplerate = samplerate

    def set_blocksize(self, blocksize):
        self.blocksize = blocksize

    def set_dtype(self, dtype):
        self.dtype = [dtype, dtype]

    def set_latency(self, latency):
        if isinstance(latency, int):
            latency_map = {0: 'low', 1: 'high'}
            latency = latency_map.get(latency, 'low')
        self.latency = [latency, latency]

    def set_channels(self, channels):
        self.channels = [channels, channels]

    def callback(self, indata, outdata, frames, time, status):
        self.print_stream_device_timing()
        processed = self.process_audio(indata)
        outdata[:] = processed 

    def start_stream(self):
        if self.stream is None:
            #wasapi_exclusive = sd.WasapiSettings(exclusive=True) # not avaiable for my device (for now)
            try:
                self.stream = sd.Stream(
                    device=(self.input, self.output),
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    dtype=self.dtype,
                    latency=self.latency,
                    channels=self.channels,
                    callback=self.callback,
                )
                self.stream.start()
            except Exception as e:
                print(f'Stream exception {e}')

    def end_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
    
    def process_audio(self, indata):
        debug_speech_detect_start = time.time()
        # idea do later to capure dual chanel audio of original speech to later autotune tts to it?
        audio_tensor = torch.from_numpy(indata[:, 0]).to(torch.float32)
        resampled_audio = self.resample_audio(audio_tensor, self.samplerate, self.vad_samplerate)
        self.vad_audio_buffer = torch.cat([self.vad_audio_buffer, resampled_audio])

        while self.vad_audio_buffer.shape[0] > self.vad_buffer_lenght:
            chunk = self.vad_audio_buffer[:self.vad_buffer_lenght]
            self.vad_audio_buffer = self.vad_audio_buffer[self.vad_buffer_lenght:]

            speech_prob = self.vad_model(chunk, self.vad_samplerate).item()
            if speech_prob > self.vad_speech_prob:
                self.speech_audio_buffer = torch.cat([self.speech_audio_buffer, chunk])
                # speech dtect takes about 1.30 - 2.30 ms
                debug_speech_detect_end = time.time()
                self.debug_time_print("Speech detect:", debug_speech_detect_start, debug_speech_detect_end)
                print(f"Speech detected! (Confidence: {speech_prob:.2f})")

        return indata

    def resample_audio(self, audio_tensor, original_rate, target_rate):
        if original_rate == target_rate:
            return audio_tensor

        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
        return resampler(audio_tensor)

    def print_stream_device_timing(self):
        if self.debug_printend == False:
            self.debug_printend = True
            input_lat = self.stream.latency[0]
            output_lat = self.stream.latency[1]
        
            print("---")
            print(f"Reported Input Latency: {input_lat * 1000:.2f} ms")
            print(f"Reported Output Latency: {output_lat * 1000:.2f} ms")
            print(f"Total Reported Latency: {(input_lat + output_lat) * 1000:.2f} ms")
            print("---")
    
    def debug_time_print(self, text, debug_time_start, debug_time_end):
        elapsed_ms = (debug_time_end - debug_time_start) * 1000
        print(f"{text} {elapsed_ms:.2f} ms")