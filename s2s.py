import sounddevice as sd

class S2S:
    def __init__(self):
        self.stream = None
        self.input = sd.default.device[0]
        self.output = sd.default.device[1]
        self.samplerate = sd.default.samplerate
        self.blocksize = sd.default.blocksize
        self.dtype = sd.default.dtype
        self.latency = [0.05, 0.05]
        self.channels = sd.default.channels

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
        self.blocksize = [blocksize, blocksize]

    def set_dtype(self, dtype):
        self.dtype = [dtype, dtype]

    def set_latency(self, latency):
        self.latency = [latency, latency]
        print(self.latency)

    def set_channels(self, channels):
        self.channels = [channels, channels]

    def callback(self, indata, outdata, frames, time, status):
        outdata[:] = indata 

    def start_stream(self):
        if self.stream is None:
            self.stream = sd.Stream(
                device=(self.input, self.output),
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                dtype=self.dtype,
                latency=self.latency,
                channels=self.channels,
                callback=self.callback
            )
            self.stream.start()

    def end_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None