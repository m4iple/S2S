import faster_whisper

class Stt:
    def __init__(self, config):
        self.cfg = config["stt"]
        self.model = None

        self.load_model()

    def load_model(self):
        """Loads the stt model"""
        self.model = faster_whisper.WhisperModel(
            self.cfg["model_path"],
            device=self.cfg["device"],
            compute_type=self.cfg["compute_type"]
        )

    def transcribe(self, audio):
        """Transcribes the audio"""
        segments, _ = self.model.transcribe(
            audio,
            language=self.cfg["language"],
            beam_size=self.cfg["beam_size"],
            word_timestamps=self.cfg["word_timestamps"],
            vad_filter=self.cfg["vad_filter"],
            initial_prompt=self.cfg["initial_prompt"]
        )

        full_text = []
        last_word_end_time = 0
        for segment in segments:
            full_text.append(segment.text)
            if segment.words:
                last_word_end_time = segment.words[-1].end

        text = "".join(full_text).strip()

        return text, last_word_end_time