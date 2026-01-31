# Issues

- Subtitles randomly getting stuck 
    - hard to reproduce...


- training: move configs int config file
- training: verify it still saves models in to the correct folder


If you hear popping/clicking:
    Change latency=0.05 to latency=0.08 (80ms).
    Change blocksize=256 to 512.
    This will add ~30ms of delay but make the audio "bulletproof."

- s2s: add audio loop to config


```
[INFO] Loading Sherpa-ONNX Streaming Model...
Traceback (most recent call last):
  File "C:\Users\Aspen\Dev\S2S\main.py", line 27, in <module>
    main()
  File "C:\Users\Aspen\Dev\S2S\main.py", line 17, in main
    s2s_instance = S2s(subtitle_window=subtitle_controller)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aspen\Dev\S2S\src\s2s.py", line 44, in __init__
    self.stt = Stt(self.cfg)
               ^^^^^^^^^^^^^
  File "C:\Users\Aspen\Dev\S2S\src\models\stt\stt_sherpa.py", line 12, in __init__
    self.load_model()
  File "C:\Users\Aspen\Dev\S2S\src\models\stt\stt_sherpa.py", line 26, in load_model
    self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\Aspen\miniconda3\envs\s2s_env\Lib\site-packages\sherpa_onnx\online_recognizer.py", line 297, in from_transducer
    self.recognizer = _Recognizer(recognizer_config)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: D:\a\_work\1\s\onnxruntime\core\session\provider_bridge_ort.cc:1209 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "C:\Users\Aspen\miniconda3\envs\s2s_env\Lib\site-packages\onnxruntime_providers_cuda.dll"
```