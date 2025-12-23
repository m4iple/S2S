import torchaudio

def resample_audio(audio_tensor, original_rate, target_rate):
    """Resamples the audio using torchaudio. On error returns original audio"""
    if original_rate == target_rate:
        return audio_tensor

    try:
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
    except Exception as e:
        print(f"Error audo resample failed: {e}")
        return audio_tensor

    return resampler(audio_tensor)