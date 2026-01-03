"""Download the proper PyTorch Whisper model for training"""
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os

# Model to download (distil-whisper small.en)
model_name = "distil-whisper/distil-small.en"
save_path = ".models/stt/distil-small.en-pytorch"

print(f"[INFO] Downloading {model_name}...")
print(f"[INFO] Saving to {save_path}")

# Create directory
os.makedirs(save_path, exist_ok=True)

# Download processor and model
print("[INFO] Downloading processor...")
processor = WhisperProcessor.from_pretrained(model_name)
processor.save_pretrained(save_path)

print("[INFO] Downloading model...")
model = WhisperForConditionalGeneration.from_pretrained(model_name)
model.save_pretrained(save_path)

print(f"[INFO] Model successfully downloaded to {save_path}")
print("[INFO] Update your training.toml to use this path:")
print(f'stt_model_path = ".\\\\.models\\\\stt\\\\distil-small.en-pytorch"')
