from flask import Flask, request, jsonify
import threading
from src.models.tts import Tts
from src.audio import effects
import numpy as np

class ApiServer:
    def __init__(self, s2s_instance):
        self.s2s = s2s_instance
        self.app = Flask(__name__)
        self.server_thread = None
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.route('/synthesize', methods=['POST'])
        def synthesize():
            try:
                data = request.get_json()
                
                if not data or 'text' not in data:
                    return jsonify({'error': 'Missing text parameter'}), 400
                
                text = data['text']

                default_api_model = self.s2s.cfg.get("api", {}).get("default_model", None)
                model = data.get('model', default_api_model)

                default_font = self.s2s.cfg.get("api", {}).get("default_subtitle_font", "03-HurmitNerdFontMono-Regular")
                default_color = self.s2s.cfg.get("api", {}).get("default_subtitle_color", "#FFFFFF")
                font = data.get('font', default_font)
                color = data.get('color', default_color)
                
                if not text.strip():
                    return jsonify({'error': 'Text cannot be empty'}), 400
                
                if not model:
                    return jsonify({'error': 'No model specified and no default_model configured'}), 400

                self.s2s.synthesize_via_api(text, model, font, color)
                
                return jsonify({
                    'success': True,
                    'text': text,
                    'model': model,
                    'font': font,
                    'color': color
                }), 200
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/models', methods=['GET'])
        def get_models():
            try:
                models = self.s2s.get_all_tts_models()
                return jsonify({'models': models}), 200
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def status():
            return jsonify({
                'status': 'running',
                'stream_active': self.s2s.stream.is_running.is_set() if self.s2s.stream else False
            }), 200
    
    def start(self, host='127.0.0.1', port=5050):
        """Start the API server in a separate thread"""
        if self.server_thread and self.server_thread.is_alive():
            print("[API] Server is already running")
            return
        
        def run_server():
            self.app.run(host=host, port=port, debug=False, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        print(f"[API] Server started on http://{host}:{port}")
    
    def stop(self):
        """Stop the API server"""
        if self.server_thread:
            print("[API] Server stopped")
