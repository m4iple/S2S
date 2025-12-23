import faster_whisper

class stt:
    def __init__(self, config):
        self.cgf = config["stt"]
        self.model = None

        self.load_model()

    def load_model(self):
        self.model = faster_whisper.WhisperModel(
            self.cgf["model_path"],
            device=self.cgf["device"],
            compute_type=self.cgf["compute_type"]
        )

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio,
            language=self.cgf["language"],
            beam_size=self.cgf["beam_size"],
            word_timestamps=self.cgf["word_timestamps"],
            vad_filter=self.cgf["vad_filter"],
            initial_prompt=self.cgf["initial_prompt"]
        )

        full_text = []
        last_word_end_time = 0
        for segment in segments:
            full_text.append(segment.text)
            if segment.words:
                last_word_end_time = segment.words[-1].end

        text = "".join(full_text).strip()

        return text, last_word_end_time