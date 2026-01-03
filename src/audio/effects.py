import pyrubberband as rb
from scipy.signal import butter, lfilter
import numpy as np

def pitch(audio, samplerate, pitch):
    """Pitch shifts audio using rubberband"""
    return rb.pitch_shift(audio, samplerate, pitch)

def speed(audio, samplerate, speed):
    """Changes speed of audio using rubberband"""
    return rb.time_stretch(audio, samplerate, speed)

def lowpass_filter(audio, samplerate, cfg):
    """Applies a lowpass filter"""
    nyquist = 0.5 * samplerate
    normal_cutoff = cfg["cutoff"] / nyquist
    
    b, a = butter(cfg["order"], normal_cutoff, btype=cfg["btype"], analog=cfg["analog"])

    filtered = lfilter(b, a, audio)

    return filtered.astype(np.float32)

def highpass_filter(audio, samplerate, cfg):
    """Applies a highpass filter"""
    nyquist = 0.5 * samplerate
    normal_cutoff = cfg["cutoff"] / nyquist

    normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999)

    b, a = butter(cfg["order"], normal_cutoff, btype=cfg["btype"], analog=cfg["analog"])

    filtered = lfilter(b, a, audio)

    return filtered.astype(np.float32)