from abc import ABC, abstractmethod

class VADProvider():
    
    @abstractmethod
    def process_chunk(self, audio_chunk):
        """Processes an audio chunck with an VAD and returns only the spoken audio"""
        pass

    @abstractmethod
    def reset(self):
        """Resets all the buffers and stats"""
        pass
