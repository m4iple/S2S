from abc import ABC, abstractmethod

class STTProvider():
    
    @abstractmethod
    def load_model(self):
        """Loads the newest numbered stt model"""
        pass

    @abstractmethod
    def transcribe(self, audio):
        """Transcribes the audio"""
        pass
