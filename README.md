# S2S

# Installation
See [INSTALLATION.md](./doc/INSTALLATION.md) for setup instructions.

# TODO
- Model Training for now only Whisper Model, -- UI
  - load custom model instead of the web one
  - make an ui for easier editing of captured training data
   - text input (auto filled with the db data, and saves the changed text)
   - play back of saved audio blob, for review
   - ability to mark as "revewed", they shoud not show up on default (toggle in ui?)
   - button that initializes the training

- update INSTALLATION.md
- update requirements.txt

# Bugs
- Subtitles getting stuck
  - firgure out why
- onix bugs
  ```
  C:\Users\Aspen\miniconda3\envs\s2s_env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:121: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn( 
  ```

# Rewrite in future
- Subtitles
- Debug


# AI disclosure:
- UI - i hate to make UI's
- Used as faster google