from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import base64
import sys
import os
from pathlib import Path
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperProcessor, WhisperForConditionalGeneration
from datasets import Dataset, Audio
import json
import io
import wave
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import shutil
import re
from datetime import datetime

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


def prepare_dataset(transcripts: List[str], audio_data: List[torch.Tensor], processor: WhisperProcessor):
    """Prepare dataset for training"""
    
    # Convert audio tensors to numpy arrays
    audio_arrays = []
    for audio_tensor in audio_data:
        # Audio should be in 16kHz format
        audio_np = audio_tensor.numpy()
        audio_arrays.append(audio_np)
    
    # Create dataset dictionary
    dataset_dict = {
        "audio": audio_arrays,
        "transcription": transcripts
    }
    
    # Create dataset
    dataset = Dataset.from_dict(dataset_dict)
    
    def prepare_dataset_features(batch):
        # Load and process audio
        audio = batch["audio"]
        
        # Compute log-Mel input features from input audio array
        batch["input_features"] = processor.feature_extractor(
            audio, sampling_rate=16000
        ).input_features[0]
        
        # Encode target text to label ids
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        
        return batch
    
    # Process the dataset
    dataset = dataset.map(prepare_dataset_features, remove_columns=dataset.column_names)
    
    return dataset


def convert_whisper_to_faster_whisper(model_path: str, faster_whisper_base_path: str) -> tuple[bool, str, str]:
    """
    Convert fine-tuned Whisper model to Faster Whisper format with versioning.
    """
    try:
        print("[INFO] Converting Whisper model to Faster Whisper format...")
        
        # Import faster-whisper tools
        try:
            from faster_whisper import WhisperModel
            from ctranslate2.converters import TransformersConverter
        except ImportError:
            return False, "", "faster-whisper or ctranslate2 not installed. Install with: pip install faster-whisper ctranslate2"
        
        # Create base directory if it doesn't exist
        faster_whisper_base_path = os.path.abspath(faster_whisper_base_path)
        os.makedirs(faster_whisper_base_path, exist_ok=True)
        
        # Get next version number
        version_num = get_next_version(faster_whisper_base_path)
        version_path = os.path.join(faster_whisper_base_path, f"{version_num}")
        
        print(f"[INFO] Creating version: {version_path}")
        
        # Don't create the directory - let the converter create it or use force=True if it exists
        if os.path.exists(version_path):
            print(f"[INFO] Version directory exists, removing old files...")
            import shutil
            shutil.rmtree(version_path)
        
        # Convert using CTranslate2
        print(f"[INFO] Converting model from {model_path} to CTranslate2 format...")
        converter = TransformersConverter(model_path)
        converter.convert(vmap=None, quantization=None, output_dir=version_path, force=True)
        
        # Verify conversion
        if os.path.exists(os.path.join(version_path, "model.bin")):
            print(f"[INFO] Conversion successful! Model saved to {version_path}")
            return True, version_path, ""
        else:
            return False, "", "Conversion completed but model files not found"
            
    except Exception as e:
        error_msg = f"Conversion failed: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return False, "", error_msg


def get_next_version(base_path: str) -> int:
    """
    Get the next version number by checking existing version folders.
    """
    if not os.path.exists(base_path):
        return 1
    
    version_folders = []
    for item in os.listdir(base_path):
        match = re.match(r'v(\d+)', item)
        if match:
            version_folders.append(int(match.group(1)))
    
    return max(version_folders) + 1 if version_folders else 1


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
        
        print(f"[INFO] Training model with {len(item_ids)} items")
        
        # Get model path from config and convert to absolute path
        model_path = training_cfg.get('training', {}).get('stt_model_path', '.\\.models\\stt\\whisper')
        # Convert to absolute path
        if not os.path.isabs(model_path):
            # Resolve relative to the project root (3 levels up from this file)
            project_root = Path(__file__).parent.parent.parent
            model_path = os.path.abspath(os.path.join(project_root, model_path))
        
        # Load processor and model
        print(f"[INFO] Loading model from {model_path}")
        processor = WhisperProcessor.from_pretrained(model_path, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(model_path, local_files_only=True)
        
        # Prepare dataset
        print("[INFO] Preparing dataset...")
        train_dataset = prepare_dataset(transcripts, audio_data, processor)
        
        # Get training parameters from config
        train_cfg = training_cfg.get('training', {})
        batch_size = train_cfg.get('batch_size', 4)
        gradient_accumulation = train_cfg.get('gradient_accumulation_steps', 2)
        learning_rate = train_cfg.get('learning_rate', 1e-5)
        warmup_steps = train_cfg.get('warmup_steps', 50)
        num_epochs = train_cfg.get('num_train_epochs', 3)
        save_steps = train_cfg.get('save_steps', 100)
        eval_steps = train_cfg.get('eval_steps', 100)
        logging_steps = train_cfg.get('logging_steps', 25)
        
        # Set up training arguments
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
            greater_is_better=False,
            push_to_hub=False,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        
        # Initialize data collator
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
        )
        
        # Train the model
        print("[INFO] Starting training...")
        trainer.train()
        
        # Save the fine-tuned model to a temporary directory first (to avoid Windows file locking issues)
        import tempfile
        import shutil
        
        temp_save_dir = tempfile.mkdtemp(prefix="whisper_save_")
        try:
            print(f"[INFO] Saving fine-tuned model to temporary directory: {temp_save_dir}")
            model.save_pretrained(temp_save_dir)
            print(f"[INFO] Model saved. Files in temp directory: {os.listdir(temp_save_dir)}")
            
            processor.save_pretrained(temp_save_dir)
            print(f"[INFO] Processor saved. Files in temp directory: {os.listdir(temp_save_dir)}")
            
            # Clear model references to release memory locks
            del model
            del processor
            del trainer
            import gc
            gc.collect()
            print("[INFO] Memory released and garbage collected")
            
            # Make sure target directory exists
            os.makedirs(model_path, exist_ok=True)
            
            # Move saved files from temp directory to final location
            print(f"[INFO] Moving model from {temp_save_dir} to {model_path}")
            
            # Clear the target directory if it exists (except for specific subdirs we want to keep)
            if os.path.exists(model_path):
                existing_items = os.listdir(model_path)
                print(f"[INFO] Current items in {model_path}: {existing_items}")
                for item in existing_items:
                    item_path = os.path.join(model_path, item)
                    if item != "checkpoints":  # Keep checkpoints directory
                        if os.path.isfile(item_path):
                            try:
                                os.remove(item_path)
                                print(f"[INFO] Removed file: {item}")
                            except Exception as e:
                                print(f"[WARNING] Could not remove {item_path}: {e}")
                        elif os.path.isdir(item_path):
                            try:
                                shutil.rmtree(item_path)
                                print(f"[INFO] Removed directory: {item}")
                            except Exception as e:
                                print(f"[WARNING] Could not remove directory {item_path}: {e}")
            
            # Copy files from temp to final location
            temp_items = os.listdir(temp_save_dir)
            print(f"[INFO] Files to copy from temp dir: {temp_items}")
            for item in temp_items:
                src = os.path.join(temp_save_dir, item)
                dst = os.path.join(model_path, item)
                try:
                    if os.path.isfile(src):
                        print(f"[INFO] Copying file: {item}")
                        shutil.copy2(src, dst)
                        print(f"[INFO] Successfully copied: {item}")
                    elif os.path.isdir(src):
                        print(f"[INFO] Copying directory: {item}")
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                        print(f"[INFO] Successfully copied directory: {item}")
                except Exception as e:
                    print(f"[ERROR] Failed to copy {item}: {e}")
                    raise
            
            print(f"[INFO] Final check - files in {model_path}: {os.listdir(model_path)}")
        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_save_dir)
                print(f"[INFO] Cleaned up temporary directory: {temp_save_dir}")
            except Exception as e:
                print(f"[WARNING] Could not remove temporary directory {temp_save_dir}: {e}")
        
        # Convert to Faster Whisper format
        print("[INFO] Converting fine-tuned model to Faster Whisper format...")
        faster_whisper_path = training_cfg.get('training', {}).get('stt_faster_model_path', '.\\.models\\stt\\faster')
        if not os.path.isabs(faster_whisper_path):
            project_root = Path(__file__).parent.parent.parent
            faster_whisper_path = os.path.abspath(os.path.join(project_root, faster_whisper_path))
        
        conversion_success, version_path, conversion_error = convert_whisper_to_faster_whisper(model_path, faster_whisper_path)
        
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
            'trained_count': len(item_ids),
            'message': f'Successfully trained with {len(item_ids)} items',
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