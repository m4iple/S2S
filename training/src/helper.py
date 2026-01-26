import tempfile
import shutil
import os
from datasets import Dataset
from typing import List
import torch
import re
from transformers import WhisperProcessor

class Helper:
        
    @staticmethod
    def saveIntoTemp(model, processor, trainer, model_path):
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

    @staticmethod
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
            version_num = Helper.get_next_version(faster_whisper_base_path)
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
        
    @staticmethod
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
    
    @staticmethod
    def prepare_dataset(transcripts: List[str], audio_data: List[torch.Tensor], processor: WhisperProcessor):
        """Prepare dataset for training using Batched Processing"""
        
        dataset_dict = {
            "audio": [t.numpy() for t in audio_data],
            "transcription": transcripts
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        def prepare_dataset_features(batch):
            audio = batch["audio"]
            
            batch["input_features"] = processor.feature_extractor(
                audio, sampling_rate=16000
            ).input_features

            batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
            
            return batch
        
        dataset = dataset.map(
            prepare_dataset_features, 
            remove_columns=dataset.column_names,
            batched=True,        # <-- CRUCIAL FOR i9 SPEED
            batch_size=8         # Adjust based on RAM (8 is safe)
        )
        
        return dataset
