# Refactor

## TEMP IDEAS - just wite things down
- create a general utils folder
- split up debug - move out the training data saving
- crate a www folder for the web stuff
- the training related code int the trainig folder
- move the default optons into an ini or conf?
- create a UI folder for all the py ui stuff
- s2s some where different?

## New folder structure
S2S
|- .fonts   > fonts for the subtitles
|- .models  > local models
|  |- stt   > speech to text models
|  |- tts   > text to speech
|- configs  > config files (default values)
|- doc      > docs
|- src      > core application logic
|- training > training scripts
|  |- src   > core training
|  |- www   > web files
|- ui       > PyQt6 ui scripts
|- utils    > misc utils


### Spliting s2s.py / debug.py

- src/s2s.py          
 - parses the default config from the config file
 - handles changes to the values
 - inits the stt model (with the values -- values need to be changable)
 - inits the tts model (with the values -- values need to be changable)
 - inits the stream (and gives a callback)
 - has callback function that handles the stt - tts function
- src/audio/stream.py
 - handles the basic audio stream setup
- src/audio/effects.py
 - has the audio effect functions
- src/models/stt.py
 - STT model wrapper
- src/models/tts.py
 - TTS model wrapper
- src/models/vad.py
 - VAD model wrapper
- utils/database.py
 - handles connection to the database and queries
- utils/timing.py
 - handles the debug timing printing
- utils/config.py
 - reads the default config file
- utils/audio.py
 - audio utils like resamling
