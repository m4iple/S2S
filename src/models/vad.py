import torch
from silero_vad import load_silero_vad


class Vad:
    def __init__(self, config):
        self.cfg = config["vad"]
        self.model = load_silero_vad()

        self.vad_buffer = torch.empty(0)
        self.speech_buffer = torch.empty(0)
        self.pre_speech_buffer = torch.empty(0)

        self.is_speaking = False
        self.silence_counter = 0
        self.process_now = False

    def process_chunk(self, audio_chunk):
        """Processes an audio chunck with an VAD and returns only the spoken audio"""
        self.vad_buffer = torch.cat([self.vad_buffer, audio_chunk])

        speech_audio = None
        should_process = False
        
        while self.vad_buffer.shape[0] >= self.cfg["buffer_chunk_size"]:
            chunk = self.vad_buffer[:self.cfg["buffer_chunk_size"]]
            self.vad_buffer = self.vad_buffer[self.cfg["buffer_chunk_size"]:]

            speech_prob = self.model(chunk, self.cfg["samplerate"]).item()

            self.pre_speech_buffer = torch.cat([self.pre_speech_buffer, chunk])
            max_pre_frames = self.cfg["pre_speech_frames"] * self.cfg["buffer_chunk_size"]
            if self.pre_speech_buffer.shape[0] > max_pre_frames:
                frames_to_remove = self.pre_speech_buffer.shape[0] - max_pre_frames
                self.pre_speech_buffer = self.pre_speech_buffer[frames_to_remove:]

            if speech_prob > self.cfg["speech_prob_threshold"]:
                self.silence_counter = 0
                
                if not self.is_speaking:
                    self.is_speaking = True
                    self.speech_buffer = torch.cat([self.speech_buffer, self.pre_speech_buffer])

                self.speech_buffer = torch.cat([self.speech_buffer, chunk])

                if self.speech_buffer.shape[0] > self.cfg["max_buffer_frames"]:
                    self.process_now = True
            else:
                if self.is_speaking:
                    if self.silence_counter < self.cfg["post_speech_silence_frames"]:
                        self.speech_buffer = torch.cat([self.speech_buffer, chunk])
                    
                    self.silence_counter += 1

                    threshold = self.cfg["silence_threshold_frames"] + self.cfg["post_speech_silence_frames"]

                    if self.silence_counter > threshold or self.process_now:
                        speech_audio = self.speech_buffer.clone()
                        self._reset_on_speech()
                        should_process = True

        return speech_audio, should_process
    
    def reset(self):
        """Resets all the buffers and stats"""
        self.vad_buffer = torch.empty(0)
        self.speech_buffer = torch.empty(0)
        self.pre_speech_buffer = torch.empty(0)
        self.is_speaking = False
        self.silence_counter = 0
        self.process_now = False

    def _reset_on_speech(self):
        """Resets the speech variables"""
        self.is_speaking = False
        self.speech_buffer = torch.empty(0)
        self.silence_counter = 0
        self.process_now = False