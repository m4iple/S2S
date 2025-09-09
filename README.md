# S2S



# Installation

- check if you have the [Terminal App](https://github.com/microsoft/terminal) installed to make your life easier

- Download conda with Python 3.1
[Powershell using wget](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-powershell)

``` ps
wget "https://repo.anaconda.com/miniconda/Miniconda3-py311_25.5.1-1-Windows-x86_64.exe" -outfile ".\miniconda.exe"
Start-Process -FilePath ".\miniconda.exe" -ArgumentList "/S" -Wait
del .\miniconda.exe
```

- Restart the Terminal app

- Switch to the Conda Powershell Prompt

- [create a conda env](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
```
conda create --name s2s_env
```

- cd to the Project

- Install the Requirements (i hope i have added all of them)

```
pip install -r requirements.txt
```

- Download [Voice models](https://huggingface.co/rhasspy/piper-voices/tree/main) make shure the voices.json is in the root of the .models/tts folder

- Download Fonts and put the .ttne into the .fonts folder

# Runing the script

in the root of the Project

if the env is not active then activate it
```
conda activate s2s_env
```

if you have visual [studio code](https://code.visualstudio.com) installed and want to edit text

```
code .
```

Starting the pyton (starting the firs time will take a while because it needs to download the VAD and Whisper models)

```
python ./main.py
```


# TODO
- Subtitles getting stuck
  - firgure out why

# Rewrite in future
- Subtitles
- Debug


# AI disclosure:
- UI - i hate to make UI's
- Used as faster google