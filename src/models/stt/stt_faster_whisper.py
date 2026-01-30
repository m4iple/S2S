from .stt_provider import STTProvider
import faster_whisper
import os
from pathlib import Path

class Stt(STTProvider):
    def __init__(self, config):
        self.cfg = config["stt"]
        self.model = None

        self.load_model()

    def load_model(self):
        """Loads the newest numbered stt model"""
        base_path = Path(self.cfg["model_path"])
        
        numbered_dirs = []
        if base_path.exists():
            for item in base_path.iterdir():
                if item.is_dir() and item.name.isdigit():
                    numbered_dirs.append(int(item.name))
        
        if numbered_dirs:
            newest_version = max(numbered_dirs)
            model_path = base_path / str(newest_version)
        else:
            model_path = base_path
        
        print(f"[DEBUG] {str(model_path)}")

        self.model = faster_whisper.WhisperModel(
            str(model_path),
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