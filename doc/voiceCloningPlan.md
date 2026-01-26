# Project Plan: Local Piper TTS Model Creator App

## Overview
A local web application to train and export custom voice models for Piper TTS. The system will process audio, fine-tune a VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech) model, and export the final checkpoint to ONNX format.

---

## Phase 1: Environment & Dependency Setup
Because training requires GPU acceleration, the foundational setup is critical.

* **System Requirements:** Linux (Ubuntu 22.04 recommended) or Windows via WSL2.
* **Hardware:** An NVIDIA GPU with at least 8GB of VRAM (RTX 3060 or higher is ideal).
* **Core Dependencies:**
    * Python 3.10+
    * `espeak-ng` (used by Piper for phonemization).
    * `ffmpeg` or `sox` (for audio standardization).
    * CUDA toolkit and cuDNN (to interface with the GPU).
* **Python Libraries:** `FastAPI` (backend), `Celery` or `RQ` (for background task queuing), `Streamlit` or `React` (frontend), and the `piper-train` repository.

## Phase 2: Data Preprocessing Pipeline
Before Piper can train, the app needs to convert raw user recordings into a strict dataset format.

1. **Audio Ingestion:** Accept user audio (WAV/MP3) and transcripts.
2. **Standardization:** Use FFmpeg to convert all audio to:
    * **Format:** 16-bit WAV
    * **Channels:** Mono (Single Channel)
    * **Sample Rate:** 22050 Hz (Standard for Piper "medium" quality)
3. **LJSpeech Formatting:** Generate a `metadata.csv` file mapping the audio paths to the transcripts, formatted as: `audio_filename|Transcript text`.
4. **Phonemization:** Run the `piper_train.preprocess` script to convert the text/audio into a dataset folder with a `config.json` and spectrograms.

## Phase 3: The Training Pipeline
The app will perform **fine-tuning** on an existing base model rather than training from scratch.

* **The Base Checkpoint:** Download a pre-trained `.ckpt` model (e.g., the English "Lessac" voice) to act as the base.
* **The Training Script:** Execute the `piper_train` module via a background worker. 
    * *Key Hyperparameters:* Batch Size (lower to save VRAM), Max Epochs (1000-2000 for fine-tuning), and Learning Rate.
* **Checkpointing:** Output a PyTorch Checkpoint (`.ckpt`) periodically.

## Phase 4: ONNX Export
Once the PyTorch model reaches the target epochs, convert it so Piper can use it.

* **The Export Script:** Run the built-in `piper_train.export_onnx` utility.
* **Metadata Integration:** Combine the exported `.onnx` file with the training `config.json` to create the final `.onnx.json` config file required by the Piper runtime.

## Phase 5: Web Application Architecture

| Component | Technology | Responsibility |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit or React | A dashboard to upload audio/text, set training hyperparameters, view real-time loss graphs, and download final ONNX files. |
| **API Backend** | FastAPI (Python) | Handles file uploads, validates data, and sends the training job to the queue. |
| **Task Queue** | Redis + Celery | Executes the long-running Python training scripts on the GPU without freezing the web UI. |