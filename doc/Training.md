# Training

Flask web application for reviewing and training speech-to-text (STT) models. The system manages transcripts from the database, allows users to review and edit them, and trains the local model with validated data.


## Structure

```
S2S/
├── requirements_training.txt   # Python dependencies
├── training_server.ps1         # PowerShell startup script
├── main.py                     # Main entry point
├── training/
│   ├── src/
│   │   └── app.py             # Flask application & API routes
│   └── www/
│       ├── templates/
│       │   ├── index.html     # Dashboard home page
│       │   └── editor.html    # Data editor page
│       └── static/
│           ├── css/
│           │   └── style.css  # Styling
│           └── js/
│               ├── dashboard.js # Dashboard functionality
│               └── editor.js    # Editor functionality
├── utils/
│   ├── audio.py               # Audio utilities
│   ├── config.py              # Configuration loader
│   ├── database.py            # Database operations
└── configs/
    └── training.toml              # Training configuration
```


## Database Schema

Table: `s2s_training_data`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PRIMARY KEY | Auto-incrementing record ID |
| transcript | TEXT | Speech transcript text |
| audio_blob | BLOB | Audio data in binary format |
| is_reviewed | INTEGER | 0/1 flag, marks item as manually reviewed |
| is_trained | INTEGER | 0/1 flag, marks item used for model training |
| timestamp | DATETIME | Record creation timestamp |

Location: `C:\Users\Aspen\Dev\database\database.db`

STT Model: `.models/stt/distil-small.en/`


## Flask API Endpoints

### Health & Stats

- **GET** `/api/health` - Health check
- **GET** `/api/data/stats` - Get training statistics
  - Response: `{ total, reviewed, trained, pending_training }`

### Training Data

- **GET** `/api/data` - List all training items
  - Query params: `reviewed`, `trained` (boolean filters)
  - Response: Array of items with `id`, `transcript`, `is_reviewed`, `is_trained`, `timestamp`

- **GET** `/api/data/<id>` - Get specific item with audio
  - Response: Item data + base64-encoded audio

- **PUT** `/api/data/<id>` - Update item
  - Body: `{ transcript, is_reviewed }`
  - Updates transcript and review status

### Training

- **POST** `/api/train` - Start model training
  - Trains with all reviewed but untrained items
  - Sets `is_trained=1` for used items
  - Response: `{ success, trained_count, message }`


## Web Interface

### Dashboard (`/`)
- Statistics cards showing total, reviewed, trained, and pending items
- Filterable data table of all training items
- Train button to start model training with reviewed data

### Editor (`/editor`)
- Load unreviewed items sequentially
- Audio player for playback
- Text editor for transcript review and editing
- Mark items as reviewed before saving
- Navigation between items
- Item details panel

### Workflow

1. **Collect Data**: Audio and initial transcripts stored via the main S2S system
2. **Review**: Open Editor, listen to audio, correct transcripts
3. **Mark Reviewed**: Check "Mark as Reviewed" before saving
4. **Train**: Return to Dashboard, click "Train Model" when ready
5. **Monitor**: Dashboard shows statistics of completed items

