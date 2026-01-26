# Issues

- Subtitles randomly getting stuck 
    - hard to reproduce...

- onix bugs
  ```
  C:\Users\Aspen\miniconda3\envs\s2s_env\lib\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:121: UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names.Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
  warnings.warn( 
  ```


- training: move configs int config file
- training: verify it still saves models in to the correct folder


If you hear popping/clicking:
    Change latency=0.05 to latency=0.08 (80ms).
    Change blocksize=256 to 512.
    This will add ~30ms of delay but make the audio "bulletproof."

- s2s: add audio loop to config