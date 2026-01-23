# Installation

AI GENERATED 

---

## Requirements

- **Operating system:** Windows 10 or later
- **Python:** 3.10 or 3.11 (use a Conda environment for easier dependency management)
- **Conda:** Anaconda or Miniconda (recommended for scripts provided)
- **Hardware:** CPU is supported; GPU recommended for faster STT/TT S and training (install CUDA-compatible PyTorch/ONNX for GPU acceleration)

Files with Python dependencies:

- `requirements.txt` (runtime / GUI / inference)
- `requirements_training.txt` (training server and fine-tuning)

---

## Quick start (desktop application)

1. Clone the repository and change into the project directory:

```powershell
git clone <repo-url> S2S
cd S2S
```

2. Create and activate the recommended Conda environment (`s2s_env`):

```powershell
conda create -n s2s_env python=3.11 -y
conda activate s2s_env
```

3. Install runtime dependencies:

```powershell
pip install -r requirements.txt
```

4. (Optional) If you plan to use GPU acceleration, install a CUDA-compatible PyTorch and ONNX runtime. Example (adjust CUDA version as needed):

```powershell
# Example for CUDA 11.8 (adjust per your CUDA toolkit)
pip install "torch" torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

5. Start the GUI app:

```powershell
# option A: use the included helper script
.\s2s.ps1

# option B: run directly
python main.py
```

---

## Training server (optional)

The training server is a small Flask app used to review and fine-tune models. It is expected to run in a separate Conda environment (`s2s_training_env`).

1. Create and activate the training environment:

```powershell
conda create -n s2s_training_env python=3.11 -y
conda activate s2s_training_env
```

2. Install training dependencies:

```powershell
pip install -r requirements_training.txt
```

3. Configure model paths and DB path in `configs/training.toml` as needed. The defaults point to local `.models` and a `database` folder; ensure the `database` folder exists.

4. Run the training server:

```powershell
.\training_server.ps1
# or
python training/src/app.py
```

5. Use the web UI under `training/www` (`index.html` / `editor.html`) to review and start training.

---

## Downloading a Whisper model for training

To download a PyTorch Whisper model suitable for fine-tuning, run:

```powershell
python download_model.py
```

This saves a Distil-Whisper model into `.\models\stt\distil-small.en-pytorch` and prints the `stt_model_path` setting to use in `configs/training.toml`.

---

## Configuration

- Global settings are in `configs/s2s.toml` (audio, VAD, STT, TTS, UI preferences).
- Training-specific settings are in `configs/training.toml` (Flask host/port, DB path, model paths, training hyperparameters).

Adjust `stt_model_path` or `stt_faster_model_path` if you place models in custom locations.

If the default database path in `configs/training.toml` does not exist, create the folder and file or point it to a location you prefer.

---

## Troubleshooting

- If `conda` is not found when running `s2s.ps1` or `training_server.ps1`, ensure Anaconda/Miniconda is installed and available in your shell, or update the script to point to the correct conda installation.

- ONNX/ CUDA warnings: if `onnxruntime` reports missing `CUDAExecutionProvider`, either install `onnxruntime-gpu` that matches your CUDA version or fall back to CPU (`onnxruntime`).

- If audio devices are misconfigured, adjust the device IDs and channels in `configs/s2s.toml` under `[audio]` and `[monitoring]`.

- For model-specific installation issues (e.g., `ctranslate2` GPU builds), consult the upstream project documentation for installation instructions that match your GPU and CUDA version.

---

## Notes & tips

- Use Conda environments named in the scripts (`s2s_env`, `s2s_training_env`) or change the script `envName` variables to match your preferred environment names.

- Keep large models in the `.models` folder (ignored by default by many tooling) and set the paths in the TOML configs accordingly.

- Before starting training, ensure you have reviewed and, if necessary, converted models to the correct format expected by the training pipeline (`transformers` or faster-whisper / `ctranslate2`).

---

If you need more targeted help (GPU installs, specific errors, or configuring particular audio hardware), add details about your environment and the exact error messages and I can expand this guide with commands and fixes.
