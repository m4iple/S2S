from abc import ABC, abstractmethod

class TTSProvider():
    
    @abstractmethod
    def load_model(self, model):
        """Loads tts model"""
        pass

    @abstractmethod
    def synthesize(self, text):
        """Synthesizes given text"""
        pass

    @abstractmethod
    def get_model_path(self, model):
        """Gets the model path by autodetecting .onnx files in the tts models folder"""
        pass

    @abstractmethod
    def get_all_models(self):
        """Detect all .onnx models in the tts models folder and return metadata list"""
        pass
