import torch
import torchaudio
import zlib
import numpy as np

def resample_audio(audio_tensor, original_rate, target_rate):
    """Resamples the audio using torchaudio. On error returns original audio"""
    if original_rate == target_rate:
        return audio_tensor

    try:
        resampler = torchaudio.transforms.Resample(orig_freq=original_rate, new_freq=target_rate)
    except Exception as e:
        print(f"[ERROR] audo resample failed: {e}")
        return audio_tensor

    return resampler(audio_tensor)

def format_for_database(audio):
    """Prepare audio for the database"""
    try:
        audio_blob = None

        if audio is not None:
            if isinstance(audio, torch.Tensor):
                if audio.dtype in (torch.float32, torch.float64):
                    audio_int16 = torch.clamp(audio, -1.0, 1.0).mul_(32767).to(torch.int16).cpu().numpy()
                else:
                    audio_int16 = audio.cpu().numpy().astype(np.int16)
            elif isinstance(audio, np.ndarray):
                if audio.dtype in (np.float32, np.float64):
                    audio_int16 = np.clip(audio, -1.0, 1.0, out=None)
                    audio_int16 = (audio_int16 * 32767).astype(np.int16)
                elif audio.dtype == np.int16:
                    audio_int16 = audio
                else:
                    audio_int16 = audio.astype(np.int16)
            else:
                return audio_blob

            audio_blob = zlib.compress(audio_int16.tobytes(), level=1)
        
        return audio_blob

    except Exception as e:
        print(f"[ERROR] formating audio as a blob: {e}")
        return None

def format_from_database(audio_blob):
    """Decompress and convert audio blob back to tensor"""
    try:
        if audio_blob is None:
            return None
        
        audio_bytes = zlib.decompress(audio_blob)
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_tensor = torch.from_numpy(audio_array).float() / 32767.0
        return audio_tensor
    except Exception as e:
        print(f"[ERROR] Failed to format audio from database: {e}")
        return None