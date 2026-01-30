import sys
from utils.config import load_config

# toml config
backend_name = load_config("configs/models.toml")["stt"]["type"]


if backend_name == "faster_whisper":
    from . import stt_faster_whisper as selected_module
else:
    raise ValueError(f"Unknown TTS type: {backend_name}")

sys.modules[__name__] = selected_module