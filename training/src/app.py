from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import base64
import sys
import os
from pathlib import Path
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor, WhisperForConditionalGeneration
import json
import io
import wave
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import re
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training.src.helper import Helper

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

# ============== Data Collator ==============

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech-to-text training"""
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths and need different padding methods
        # First, pad the audio features to the longest sequence in the batch
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # Pad the labels to the longest sequence in the batch
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # If bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

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
    try:
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
            
        num_items = len(item_ids)
        print(f"[INFO] Training model with {num_items} items")
        
        # ---------------------------------------------------------
        # DYNAMIC EPOCH SCALING FOR WHISPER SMALL + RTX 5090
        # ---------------------------------------------------------
        train_cfg = training_cfg.get('training', {})
        
        # Base parameters optimized for 5090
        batch_size = train_cfg.get('batch_size', 16) # Increased default
        gradient_accumulation = train_cfg.get('gradient_accumulation_steps', 1)
        learning_rate = train_cfg.get('learning_rate', 2e-5) # Slightly higher for adaptation
        
        # Dynamic Epoch Logic
        if num_items < 50:
            num_epochs = 20
            warmup_steps = 10  # Quick warmup for short runs
        elif num_items < 200:
            num_epochs = 10
            warmup_steps = 50
        elif num_items < 1000:
            num_epochs = 5
            warmup_steps = 100
        else:
            num_epochs = train_cfg.get('num_train_epochs', 3)
            warmup_steps = train_cfg.get('warmup_steps', 200)

        # Dynamic Save/Eval steps based on total steps
        # total_steps = (num_items / batch_size) * num_epochs
        # We want to log/eval about 5-10 times per run, not every 100 steps
        logging_steps = 5 
        eval_steps = 25 
        save_steps = 25

        print(f"[INFO] Dynamic Config -> Epochs: {num_epochs}, Batch: {batch_size}, Warmup: {warmup_steps}")
        # ---------------------------------------------------------

        # Get model path and load model/processor
        model_path = training_cfg.get('training', {}).get('stt_model_path', '.\\.models\\stt\\whisper')
        if not os.path.isabs(model_path):
            project_root = Path(__file__).parent.parent.parent
            model_path = os.path.abspath(os.path.join(project_root, model_path))
            
        print(f"[INFO] Loading model from {model_path}")
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        
        print("[INFO] Preparing dataset...")
        train_dataset = Helper.prepare_dataset(transcripts, audio_data, processor)
        
        # Set up training arguments - OPTIMIZED FOR 5090
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(model_path, "checkpoints"),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            report_to=["none"],
            load_best_model_at_end=False,
            push_to_hub=False,
            remove_unused_columns=False,
            label_names=["labels"],
            bf16=True,                  # 5090 natively supports Brain Float 16
            dataloader_num_workers=0,   # Feed the GPU faster
            generation_max_length=128,  # Optimized for short sentence generation
            dataloader_pin_memory=False, # Drastically speeds up Windows data loading
            torch_compile=False,            # Uses Intel optimizations to compile the graph
        )
        
        # Initialize data collator and trainer
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=processor.feature_extractor,
        )
        
        print("[INFO] Starting training...")
        trainer.train()
        
        Helper.saveIntoTemp(model, processor, trainer, model_path)
        
        # Convert to Faster Whisper format
        print("[INFO] Converting fine-tuned model to Faster Whisper format...")
        faster_whisper_path = training_cfg.get('training', {}).get('stt_faster_model_path', '.\\.models\\stt\\faster')
        if not os.path.isabs(faster_whisper_path):
            project_root = Path(__file__).parent.parent.parent
            faster_whisper_path = os.path.abspath(os.path.join(project_root, faster_whisper_path))
        
        conversion_success, version_path, conversion_error = Helper.convert_whisper_to_faster_whisper(model_path, faster_whisper_path)
        
        if not conversion_success:
            return jsonify({
                'success': False,
                'trained_count': len(item_ids),
                'error': f'Training completed but Faster Whisper conversion failed: {conversion_error}',
                'warning': 'Model was fine-tuned but not converted to Faster Whisper format'
            }), 500
        
        # Mark items as trained
        success = db.mark_items_trained(item_ids)
        if not success:
            return jsonify({
                'success': False,
                'error': 'Training and conversion completed but failed to mark items as trained'
            }), 500
        
        print(f"[INFO] Training completed successfully with {len(item_ids)} items")
        print(f"[INFO] Faster Whisper model saved to {version_path}")
        
        return jsonify({
            'success': True,
            'trained_count': num_items,
            'epochs_run': num_epochs, # Added this to API response for your tracking
            'message': f'Successfully trained with {num_items} items',
            'faster_whisper_version': version_path,
            'conversion_success': True
        }), 200
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'trained_count': 0,
            'error': f'Training failed: {str(e)}'
        }), 500

@app.route('/api/models/faster-whisper/versions', methods=['GET'])
def get_faster_whisper_versions():
    """Get list of available Faster Whisper model versions"""
    try:
        faster_whisper_path = training_cfg.get('training', {}).get('stt_faster_model_path', '.\\.models\\stt\\faster')
        if not os.path.isabs(faster_whisper_path):
            project_root = Path(__file__).parent.parent.parent
            faster_whisper_path = os.path.abspath(os.path.join(project_root, faster_whisper_path))
        
        if not os.path.exists(faster_whisper_path):
            return jsonify({'versions': []}), 200
        
        versions = []
        for item in sorted(os.listdir(faster_whisper_path)):
            match = re.match(r'(\d+)', item)
            if match:
                version_num = int(match.group(1))
                version_dir = os.path.join(faster_whisper_path, item)
                versions.append({
                    'version': version_num,
                    'path': version_dir,
                    'created': datetime.fromtimestamp(os.path.getctime(version_dir)).isoformat()
                })
        
        # Sort by version number descending (newest first)
        versions.sort(key=lambda x: x['version'], reverse=True)
        return jsonify({'versions': versions}), 200
    except Exception as e:
        print(f"[ERROR] Failed to get Faster Whisper versions: {str(e)}")
        return jsonify({'error': str(e)}), 500

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