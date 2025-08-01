import os
import json

tts_model_path = '.models/tts'

def get_model_path(model):
    voices_json_path = tts_model_path + '/voices.json'
    if os.path.exists(voices_json_path):
        try:
            with open(voices_json_path, 'r', encoding='utf-8') as f:
                voices_data = json.load(f)
            
            if model in voices_data:
                voice_info = voices_data[model]
                for file_path in voice_info.get('files', {}).keys():
                    if file_path.endswith('.onnx'):
                        test_path = os.path.join(tts_model_path, file_path)
                        if os.path.exists(test_path):
                            voice_path = test_path
                            break
        except json.JSONDecodeError as e:
            print(f"Error reading voices.json: {e}")
    
    if not voice_path:
        raise FileNotFoundError(f"No TTS model found. Please ensure you have models in the .models directory.")
    
    return voice_path

def get_all_models():
    voices_json_path = tts_model_path + '/voices.json'
    if not os.path.exists(voices_json_path):
        models_dir = tts_model_path
        if not os.path.exists(models_dir):
            return []
    
    try:
        with open(voices_json_path, 'r', encoding='utf-8') as f:
            voices_data = json.load(f)
        
        available_models = []
        for voice_key, voice_info in voices_data.items():
            onnx_file_path = None
            for file_path in voice_info.get('files', {}).keys():
                if file_path.endswith('.onnx'):
                    full_path = os.path.join(tts_model_path, file_path)
                    if os.path.exists(full_path):
                        onnx_file_path = file_path
                        break
            
            if onnx_file_path:
                language = voice_info.get('language', {})
                name = voice_info.get('name', voice_key)
                quality = voice_info.get('quality', '')
                
                display_name = f"{language.get('name_english', language.get('code', ''))} {language.get('region', '')} - {name}"
                if quality:
                    display_name += f" ({quality})"
                
                available_models.append({
                    'key': voice_key,
                    'display_name': display_name,
                    'file_path': onnx_file_path,
                    'language': language,
                    'name': name,
                    'quality': quality
                })

        available_models.sort(key=lambda x: (x['language'].get('name_english', ''), x['language'].get('region', ''), x['name'], x['quality']))
        return available_models
        
    except Exception as e:
        print(f"Error reading voices.json: {e}")
        return []