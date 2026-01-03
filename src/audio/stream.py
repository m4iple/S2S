import sounddevice as sd
import threading

class AudioStream:
    """Handles the creation / stopping of the audio stream"""

    def __init__(self, config, callback):
        self.cfg_a = config["audio"]
        self.cfg_m = config["monitoring"]
        self.callback = callback
        self.stream = None
        self.monitor_stream = None
        self.is_running = threading.Event()

    def start(self):
        """Start the audio stream"""

        if self.stream:
            return

        self.is_running.set()
        
        try:
            
            self.stream = sd.Stream(
                device=(self.cfg_a["input"], self.cfg_a["output"]),
                samplerate=self.cfg_a["samplerate"],
                blocksize=self.cfg_a["blocksize"],
                dtype=self.cfg_a["dtype"],
                channels=self.cfg_a["channels"],
                callback=self._callback_wrapper,
            )

            self.stream.start()
        except Exception as e:
            print(f"[ERROR] Stream not started: {e}")
            self.stream = None

        if self.cfg_m["enabled"]:
            try:
                self.monitor_stream = sd.OutputStream(
                    device=self.cfg_m["device"],
                    samplerate=self.cfg_m["samplerate"],
                    blocksize=self.cfg_m["blocksize"],
                    dtype=self.cfg_m["dtype"],
                    channels=self.cfg_m["channels"],
                )
                
                self.monitor_stream.start()
            except Exception as e:
                print(f"[ERROR] monitor stream not started: {e}")
                self.monitor_stream = None
    
    def stop(self):
        """Stopps the audio stream"""

        self.is_running.clear()
        
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            finally:
                self.stream = None
        
        if self.monitor_stream:
            try:
                self.monitor_stream.stop()
                self.monitor_stream.close()
            finally:
                self.monitor_stream = None

    def _callback_wrapper(self, indata, outdata, frames, time, status):
        """Wrapper for the callback function"""

        self.callback(indata, outdata, frames, time, status)
        
        if self.monitor_stream is not None:
            try:
                monitor_channels = self.cfg_m["channels"]
                self.monitor_stream.write(outdata[:, :monitor_channels].copy())
            except Exception as e:
                print(f"[ERROR] writing to monitor: {e}")