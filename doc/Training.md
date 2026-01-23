# Training Server

## Config (file: `configs/training.toml`)
Key settings and defaults:

- `training.flask_host` — HTTP host (default: `127.0.0.1`)
- `training.flask_port` — HTTP port (default: `8277`)
- `training.flask_debug` — Flask debug mode (default: `false`)
- `training.database_path` — SQLite DB path used by the training server
- `training.stt_model_path` — Path to base Whisper model to fine-tune
- `training.stt_faster_model_path` — Path to store Faster Whisper converted models
- Training hyperparams: `batch_size`, `gradient_accumulation_steps`, `learning_rate`, `warmup_steps`, `num_train_epochs`, `save_steps`, `eval_steps`, `logging_steps`

Note: `stt_model_path` must contain a saved Whisper model (processor + model). The server loads models with `local_files_only=True`.

---

## Web UI
- Dashboard
  - View stats: total / reviewed / trained / pending training
  - Trigger training (calls `/api/train`)
  - Reset DB (destructive) — calls `/api/database/reset`
  - Browse training items table

- Editor
  - Play item audio
  - Edit transcripts
  - Mark items as reviewed
  - Delete items