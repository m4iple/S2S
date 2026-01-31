from .stt_provider import STTProvider
import sherpa_onnx
import numpy as np

class Stt(STTProvider):
    def __init__(self, config):
        self.cfg = config["stt"]
        self.recognizer = None
        self.stream = None
        self.last_result = ""
        
        self.load_model()

    def load_model(self):
        """Initialize the Streaming Recognizer"""
        # Load paths from config (ensure these exist in your .models folder)
        tokens = self.cfg.get("tokens", ".models/stt/sherpa/tokens.txt")
        encoder = self.cfg.get("encoder", ".models/stt/sherpa/encoder-epoch-99-avg-1.onnx")
        decoder = self.cfg.get("decoder", ".models/stt/sherpa/decoder-epoch-99-avg-1.onnx")
        joiner = self.cfg.get("joiner", ".models/stt/sherpa/joiner-epoch-99-avg-1.onnx")

        print(f"[INFO] Loading Sherpa-ONNX Streaming Model...")
        
        # FIX: Use .from_transducer() instead of direct constructor
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=1,
                sample_rate=16000,
                feature_dim=80,
                provider="cuda" if self.cfg.get("device") == "cuda" else "cpu"
            )
        except AttributeError:
            # Fallback for older versions or different bindings
            print("[WARN] 'from_transducer' not found, trying Config approach...")
            self._load_with_config(tokens, encoder, decoder, joiner)
            
        # Create the first stream
        self.stream = self.recognizer.create_stream()

    def _load_with_config(self, tokens, encoder, decoder, joiner):
        """Alternative loading method using Config objects"""
        transducer_config = sherpa_onnx.OnlineTransducerModelConfig(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            num_threads=1,
        )
        
        # Configure features (fbank)
        feature_config = sherpa_onnx.FeatureConfig(
            sample_rate=16000,
            feature_dim=80
        )
        
        recognizer_config = sherpa_onnx.OnlineRecognizerConfig(
            model_config=transducer_config,
            feature_config=feature_config,
            provider="cuda" if self.cfg.get("device") == "cuda" else "cpu",
        )
        
        self.recognizer = sherpa_onnx.OnlineRecognizer(recognizer_config)

    def transcribe(self, audio_chunk):
        """
        Feed new audio chunk and return ONLY the newly decoded text.
        'audio_chunk' should be a 1D numpy array (float32).
        """
        if self.stream is None:
            self.stream = self.recognizer.create_stream()

        # 1. Feed audio to the internal buffer
        self.stream.accept_waveform(16000, audio_chunk)

        # 2. Decode whatever is ready
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        # 3. Get the full result so far
        full_result = self.recognizer.get_result(self.stream)
        
        # 4. Diffing Logic: Return only what is NEW
        new_text = ""
        # Handle simple string result (strip leading spaces if needed)
        full_result = full_result.strip()
        
        if len(full_result) > len(self.last_result):
            # Only return the suffix
            new_text = full_result[len(self.last_result):].strip()
            self.last_result = full_result

        return new_text, 0  # 0 is dummy timestamp
    
    def reset(self):
        """Call this when you detect long silence to clear context"""
        if self.stream:
            self.stream = self.recognizer.create_stream()
            self.last_result = ""