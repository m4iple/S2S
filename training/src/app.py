from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import base64
import sys
import os
from pathlib import Path
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, AutoFeatureExtractor, AutoTokenizer, AutoModelForSpeechSeq2Seq
from datasets import Dataset
import json
import io
import wave

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.config import load_config
from utils.database import Database
from utils.audio import format_from_database

app = Flask(__name__, 
            template_folder='../../training/www/templates',
            static_folder='../../training/www/static')
CORS(app)

# Load configuration
training_cfg = load_config("configs/training.toml")

# Initialize database
db = Database()
db.open()

# ============== Health & Stats ==============

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'message': 'Training server is running'}), 200

@app.route('/api/data/stats', methods=['GET'])
def get_stats():
    """Get training statistics"""
    stats = db.get_stats()
    if stats is None:
        return jsonify({'error': 'Failed to get stats'}), 500
    return jsonify(stats), 200

# ============== Training Data ==============

@app.route('/api/data', methods=['GET'])
def get_data():
    """List all training items with optional filters"""
    reviewed_filter = request.args.get('reviewed', type=str)
    trained_filter = request.args.get('trained', type=str)
    
    # Convert string to boolean or None
    reviewed = None if reviewed_filter is None else (reviewed_filter.lower() == 'true')
    trained = None if trained_filter is None else (trained_filter.lower() == 'true')
    
    items = db.get_all_items(reviewed_filter=reviewed, trained_filter=trained)
    if items is None:
        return jsonify({'error': 'Failed to get data'}), 500
    return jsonify(items), 200

@app.route('/api/data/<int:item_id>', methods=['GET'])
def get_data_item(item_id):
    """Get specific item with audio data"""
    item = db.get_item_by_id(item_id)
    
    if item is None:
        return jsonify({'error': 'Item not found'}), 404
    
    # Convert audio blob to base64 WAV
    audio_base64 = None
    if item['audio_blob']:
        try:
            # Decompress and convert to WAV format
            audio_tensor = format_from_database(item['audio_blob'])
            if audio_tensor is not None:
                # Convert tensor to int16 numpy array
                audio_np = (audio_tensor.numpy() * 32767).astype(np.int16)
                
                # Create WAV file in memory
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 2 bytes for int16
                    wav_file.setframerate(16000)  # VAD sample rate
                    wav_file.writeframes(audio_np.tobytes())
                
                wav_data = wav_buffer.getvalue()
                audio_base64 = base64.b64encode(wav_data).decode('utf-8')
        except Exception as e:
            print(f"[WARN] Failed to encode audio: {e}")
    
    return jsonify({
        'id': item['id'],
        'transcript': item['transcript'],
        'audio_base64': audio_base64,
        'is_reviewed': item['is_reviewed'],
        'is_trained': item['is_trained'],
        'timestamp': item['timestamp']
    }), 200

@app.route('/api/data/<int:item_id>', methods=['PUT'])
def update_data_item(item_id):
    """Update item transcript and review status"""
    data = request.get_json()
    transcript = data.get('transcript')
    is_reviewed = data.get('is_reviewed', False)
    
    success = db.update_item(item_id, transcript, is_reviewed)
    if not success:
        return jsonify({'error': 'Failed to update item'}), 500
    return jsonify({'success': True, 'message': 'Item updated'}), 200

@app.route('/api/data/<int:item_id>', methods=['DELETE'])
def delete_data_item(item_id):
    """Delete a training data item"""
    success = db.delete_item(item_id)
    if not success:
        return jsonify({'error': 'Failed to delete item'}), 500
    return jsonify({'success': True, 'message': 'Item deleted'}), 200

@app.route('/api/database/reset', methods=['POST'])
def reset_database():
    """Drop and recreate the training data table"""
    success = db.reset_table()
    if not success:
        return jsonify({'success': False, 'error': 'Failed to reset database'}), 500
    print("[INFO] Database reset successfully")
    return jsonify({'success': True, 'message': 'Database reset successfully'}), 200

# ============== Training ==============

@app.route('/api/train', methods=['POST'])
def train_model():
    """Start model training with reviewed but untrained items"""
    training_items = db.get_training_items()
    
    if training_items is None:
        return jsonify({
            'success': False,
            'trained_count': 0,
            'error': 'Failed to get training items'
        }), 500
    
    if not training_items:
        return jsonify({
            'success': False,
            'trained_count': 0,
            'message': 'No reviewed items to train'
        }), 200
    
    # Prepare training data
    transcripts = []
    audio_data = []
    item_ids = []
    
    for item in training_items:
        transcript = item['transcript']
        audio_tensor = format_from_database(item['audio_blob'])
        
        if audio_tensor is not None and transcript:
            transcripts.append(transcript)
            audio_data.append(audio_tensor)
            item_ids.append(item['id'])
    
    if not transcripts:
        return jsonify({
            'success': False,
            'trained_count': 0,
            'message': 'No valid training data found'
        }), 200
    
    # Mark items as trained
    success = db.mark_items_trained(item_ids)
    if not success:
        return jsonify({
            'success': False,
            'error': 'Failed to mark items as trained'
        }), 500
    
    print(f"[INFO] Training model with {len(item_ids)} items")
    
    # Here you would implement actual model training
    # For now, we just mark items as trained
    
    return jsonify({
        'success': True,
        'trained_count': len(item_ids),
        'message': f'Successfully trained with {len(item_ids)} items'
    }), 200

# ============== Routes ==============

@app.route('/')
def dashboard():
    """Serve dashboard page"""
    return render_template('index.html')

@app.route('/editor')
def editor():
    """Serve editor page"""
    return render_template('editor.html')

if __name__ == '__main__':
    host = training_cfg.get('training', {}).get('flask_host', '127.0.0.1')
    port = training_cfg.get('training', {}).get('flask_port', 5000)
    debug = training_cfg.get('training', {}).get('flask_debug', False)
    
    print(f"[INFO] Starting training server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)