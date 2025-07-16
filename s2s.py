import sounddevice as sd
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

class S2S:
    def __init__(self):
        self.stream = None
        self.input = sd.default.device[0] #71
        self.output = sd.default.device[1] #51
        self.samplerate = sd.default.samplerate if sd.default.samplerate is not None else 44100
        self.blocksize = sd.default.blocksize
        self.dtype = sd.default.dtype
        self.latency = ['low', 'low']
        self.channels = sd.default.channels
        if self.channels is None or self.channels == [None, None]:
            self.channels = [1, 1]
        else:
            self.channels = [c if c is not None else 1 for c in self.channels]

        self.avd_model = load_silero_vad()

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
        self.print_timing()
        # create an audio queu and then trimm it with silero_vad
        outdata[:] = indata 

    def start_stream(self):
        if self.stream is None:
            wasapi_exclusive = sd.WasapiSettings(exclusive=True)
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

    def print_timing(self):
        input_lat = self.stream.latency[0]
        output_lat = self.stream.latency[1]

        print(f"Reported Input Latency: {input_lat * 1000:.2f} ms")
        print(f"Reported Output Latency: {output_lat * 1000:.2f} ms")
        print(f"Total Reported Latency: {(input_lat + output_lat) * 1000:.2f} ms")