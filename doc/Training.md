# Training System Requirements

## Overview

The training system for S2S provides a comprehensive solution for collecting, reviewing, and training custom Whisper models using real-world speech data captured during live sessions. This system enables users to improve transcription accuracy by fine-tuning the model with their specific voice characteristics and vocabulary.

## Database Schema

### Training Data Storage
The training data is stored in `C:/Users/Aspen/Dev/database/database.db` in the `s2s_training_data` table:

```sql
CREATE TABLE s2s_training_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript TEXT,
    audio_blob BLOB, -- Compressed audio data (zlib compressed int16)
    is_reviewed INTEGER DEFAULT 0, -- 0: not reviewed, 1: reviewed
    is_trained INTEGER DEFAULT 0,  -- 0: not used in training, 1: used in training
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Data Collection
- Audio data is automatically captured during live speech-to-speech sessions when:
  - `CAPTURE_TRAINING_DATA` flag is enabled in debug.py
  - `DATABASE` flag is enabled in debug.py
- Audio is stored as compressed int16 PCM data at 16kHz sample rate
- Transcripts are stored as plain text from the Whisper model output
- Each entry is timestamped for tracking purposes

## Training Web Interface (`training.py`)

### Core Requirements

#### 1. Web Server Setup
- **Framework**: Flask or FastAPI for lightweight web serving
- **Port**: Configurable (default: 8080)
- **Host**: Local development (127.0.0.1) with option for network access
- **Static Files**: Serve HTML, CSS, JS for the review interface

#### 2. Data Review Interface
- **Data Loading**: Display unreviewed training data (`is_reviewed = 0`)
- **Audio Playback**: 
  - Decompress and serve audio blobs as playable audio files
  - Support for standard web audio formats (WAV, MP3)
  - Playback controls (play, pause, stop, seek)
  - Visual waveform display (optional but recommended)
- **Text Editing**:
  - Editable text field pre-populated with original transcript
  - Real-time character/word count
  - Undo/redo functionality
  - Text validation (non-empty, reasonable length)

#### 3. Review Management
- **Navigation**: 
  - Previous/Next buttons for data browsing
  - Jump to specific entry by ID
  - Filter options (by date, reviewed status, etc.)
- **Review Actions**:
  - Save transcript changes
  - Mark as reviewed (`is_reviewed = 1`)
  - Delete entry (for bad quality data)
  - Skip to next unreviewed entry
- **Progress Tracking**:
  - Display total entries vs reviewed entries
  - Progress bar or percentage indicator
  - Statistics (average review time, entries per session)

#### 4. Quality Control Features
- **Audio Quality Indicators**:
  - Audio duration display
  - Volume level indicators
  - Signal-to-noise ratio estimation (if feasible)
- **Transcript Quality Checks**:
  - Length validation (too short/long warnings)
  - Common transcription error detection
  - Confidence scoring integration (from original Whisper output)

### User Interface Requirements

#### Layout Design
```
┌─ Header ────────────────────────────────────────────┐
│ S2S Training Data Review                            │
│ Progress: [██████░░░░] 60% (120/200 reviewed)      │
└─────────────────────────────────────────────────────┘

┌─ Audio Section ─────────────────────────────────────┐
│ Entry ID: 042 | Timestamp: 2025-10-04 14:23:15     │
│ ┌─ Audio Player ─────────────────────────────────┐  │
│ │ [▶] ──██████████████████──── 00:03 / 00:05   │  │
│ │ Volume: [████████░░] Download | Waveform     │  │
│ └───────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘

┌─ Transcript Section ────────────────────────────────┐
│ Original: "Hello how are you doing today?"          │
│ ┌─ Edit Box ─────────────────────────────────────┐  │
│ │ Hello, how are you doing today?               │  │
│ │                                               │  │
│ └───────────────────────────────────────────────┘  │
│ Characters: 32 | Words: 6                          │
└─────────────────────────────────────────────────────┘

┌─ Actions ───────────────────────────────────────────┐
│ [← Previous] [Save & Next →] [Mark Reviewed] [Delete] │
│ [Jump to ID: ___] [Filter ▼] [Export Data]         │
└─────────────────────────────────────────────────────┘
```

#### Keyboard Shortcuts
- `Space`: Play/Pause audio
- `Ctrl+S`: Save current transcript
- `Ctrl+Enter`: Save and move to next
- `Ctrl+D`: Delete current entry
- `Ctrl+R`: Mark as reviewed
- `←/→`: Navigate between entries

### Backend API Endpoints

#### Data Retrieval
- `GET /api/entries` - Get paginated list of training entries
- `GET /api/entries/{id}` - Get specific entry details
- `GET /api/entries/{id}/audio` - Stream audio data for playback
- `GET /api/stats` - Get review progress statistics

#### Data Modification
- `PUT /api/entries/{id}` - Update transcript and review status
- `DELETE /api/entries/{id}` - Delete training entry
- `POST /api/entries/{id}/review` - Mark entry as reviewed

#### Training Operations
- `POST /api/train` - Initiate model training process
- `GET /api/train/status` - Check training progress
- `GET /api/train/logs` - Get training logs

## Model Training System

### Training Pipeline Requirements

#### 1. Data Preparation
- **Data Validation**:
  - Verify audio quality (duration, sample rate, noise levels)
  - Validate transcript completeness and formatting
  - Remove entries with mismatched audio/text lengths
- **Data Formatting**:
  - Convert audio to required format for Whisper training
  - Prepare transcript files in proper format
  - Create train/validation splits (80/20 or 90/10)
- **Data Augmentation** (Optional):
  - Speed perturbation (0.9x, 1.1x)
  - Noise addition for robustness
  - Volume normalization

#### 2. Training Configuration
- **Model Selection**:
  - Base model choice (small, medium, large)
  - Fine-tuning vs full training decision
  - Language-specific model selection
- **Training Parameters**:
  - Learning rate: 1e-5 (typical for fine-tuning)
  - Batch size: 16-32 (depending on GPU memory)
  - Epochs: 10-50 (with early stopping)
  - Gradient accumulation steps: 2-4
- **Hardware Requirements**:
  - GPU memory requirements (minimum 8GB for medium model)
  - CPU fallback option (slower but functional)
  - Disk space requirements for model checkpoints

#### 3. Training Process
- **Initialization**:
  - Load base Whisper model
  - Set up optimizer and learning rate scheduler
  - Configure logging and checkpointing
- **Training Loop**:
  - Batch processing with proper error handling
  - Progress tracking and ETA estimation
  - Memory management and cleanup
  - Periodic validation and metric calculation
- **Monitoring**:
  - Loss tracking (training and validation)
  - WER (Word Error Rate) monitoring
  - Learning rate scheduling
  - Early stopping based on validation metrics

#### 4. Model Evaluation and Deployment
- **Testing**:
  - Evaluate on held-out test set
  - Compare with base model performance
  - Generate detailed metrics report
- **Model Saving**:
  - Save trained model in proper format
  - Create model metadata and configuration files
  - Backup original model before replacement
- **Integration**:
  - Hot-swap capability for live model updates
  - Fallback mechanism if new model performs poorly
  - Configuration updates for new model path

### Training Interface

#### Training Dashboard
```
┌─ Training Control Panel ────────────────────────────┐
│ Data Status: 156 reviewed entries ready for training│
│ Last Training: Never                                │
│ Current Model: distil-small.en (base)              │
│                                                     │
│ Training Configuration:                             │
│ ├─ Base Model: [distil-small.en ▼]                │
│ ├─ Learning Rate: [1e-5____]                      │
│ ├─ Batch Size: [16____]                           │
│ ├─ Max Epochs: [20____]                           │
│ └─ Validation Split: [10%___]                     │
│                                                     │
│ [🚀 Start Training] [📊 View Logs] [⚙️ Advanced]    │
└─────────────────────────────────────────────────────┘

┌─ Training Progress ─────────────────────────────────┐
│ Status: Training in progress...                     │
│ Progress: [████████░░] 80% (Epoch 8/10)           │
│ ETA: 15 minutes remaining                          │
│                                                     │
│ Current Metrics:                                    │
│ ├─ Training Loss: 0.245                           │
│ ├─ Validation Loss: 0.298                         │
│ ├─ Word Error Rate: 12.3%                         │
│ └─ Learning Rate: 8.7e-6                          │
│                                                     │
│ [⏸️ Pause] [⏹️ Stop] [📈 Live Metrics]              │
└─────────────────────────────────────────────────────┘
```

## Technical Implementation Details

### Required Dependencies
```python
# Core training framework
torch
transformers
datasets
accelerate

# Audio processing
librosa
soundfile
torchaudio

# Web interface
flask
flask-cors
flask-socketio # For real-time updates

# Database
sqlite3  # Built-in
sqlalchemy  # For ORM if preferred

# Utilities
tqdm
numpy
matplotlib  # For training plots
```

### File Structure
```
training/
├── training.py              # Main training server
├── static/
│   ├── css/
│   │   └── style.css       # UI styling
│   ├── js/
│   │   ├── app.js          # Main application logic
│   │   ├── audio-player.js # Audio playback handling
│   │   └── training.js     # Training interface logic
│   └── index.html          # Main review interface
├── templates/
│   ├── review.html         # Data review page
│   ├── training.html       # Training dashboard
│   └── base.html          # Base template
├── utils/
│   ├── __init__.py
│   ├── database.py         # Database operations
│   ├── audio_utils.py      # Audio processing utilities
│   ├── training_utils.py   # Training pipeline utilities
│   └── model_utils.py      # Model management utilities
└── models/
    ├── checkpoints/        # Training checkpoints
    ├── trained/           # Final trained models
    └── configs/           # Training configurations
```

### Security Considerations
- **Authentication**: Optional user authentication for multi-user setups
- **File Access**: Restrict file access to designated directories
- **Input Validation**: Sanitize all user inputs for XSS prevention
- **Rate Limiting**: Prevent excessive API calls
- **CORS Configuration**: Proper CORS setup for web interface

### Performance Optimization
- **Lazy Loading**: Load audio data only when needed
- **Caching**: Cache frequently accessed data and model outputs
- **Compression**: Use efficient audio compression for storage
- **Batch Processing**: Process multiple entries efficiently
- **Memory Management**: Proper cleanup of large tensors and audio data

### Error Handling and Logging
- **Comprehensive Logging**: Log all training operations and errors
- **Graceful Degradation**: Handle partial failures gracefully
- **User Feedback**: Clear error messages and recovery suggestions
- **Backup Systems**: Automatic backups before training operations
- **Recovery Mechanisms**: Ability to resume interrupted training

## Future Enhancements

### Advanced Features
- **Multi-speaker Support**: Handle different speakers in training data
- **Active Learning**: Prioritize difficult examples for review
- **Quality Scoring**: Automatic quality assessment of training pairs
- **Batch Operations**: Bulk edit and review capabilities
- **Export/Import**: Data portability between systems
- **Integration APIs**: Connect with external annotation tools
- **Real-time Training**: Continuous learning from new data
- **Model Versioning**: Track and manage multiple model versions

### Scalability Considerations
- **Distributed Training**: Support for multi-GPU training
- **Cloud Integration**: AWS/GCP training pipeline support
- **Database Scaling**: Migration to PostgreSQL for larger datasets
- **Microservices**: Split into separate services for better scalability
- **Container Support**: Docker containerization for easy deployment