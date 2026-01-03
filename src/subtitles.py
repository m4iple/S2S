import threading
import queue
from typing import Optional


class SubtitleController:
    """
    Controller that manages subtitle updates in a separate thread.
    This prevents subtitle operations from interfering with audio/TTS performance.
    """
    
    def __init__(self, subtitle_window=None):
        """
        Initialize the subtitle controller.
        """
        self.subtitle_window = subtitle_window
        self.subtitle_queue = queue.Queue()
        self.running = False
        self.thread = None
        
        if self.subtitle_window:
            self.start()
    
    def start(self):
        """Start the subtitle update thread."""
        if not self.running and self.subtitle_window:
            self.running = True
            self.thread = threading.Thread(target=self._subtitle_update_worker, daemon=True)
            self.thread.start()
    
    def stop(self):
        """Stop the subtitle update thread."""
        self.running = False
        if self.thread:
            self.subtitle_queue.put(None)
            self.thread.join(timeout=1.0)
            self.thread = None
    
    def _subtitle_update_worker(self):
        """Worker thread that processes subtitle updates."""
        while self.running:
            try:
                item = self.subtitle_queue.get(timeout=0.5)
                
                if item is None:
                    break
                
                action, text, font, color = item
                
                if action == 'set' and self.subtitle_window:
                    self.subtitle_window.set_subtitle(text, font, color)
                elif action == 'clear' and self.subtitle_window:
                    self.subtitle_window.subtitle_clear_signal.emit()
                
                self.subtitle_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Error in subtitle update worker: {e}")
    
    def set_subtitle(self, text: str, font: str = None, color: str = None):
        """
        Queue a subtitle update (thread-safe).
        """
        if self.subtitle_window and self.running:
            self.subtitle_queue.put(('set', text, font, color))
    
    def clear_subtitle(self):
        """Clear all subtitles (thread-safe)."""
        if self.subtitle_window and self.running:
            self.subtitle_queue.put(('clear', '', None, None))
    
    def set(self, text: str):
        """
        Alias for set_subtitle for backward compatibility.
        """
        self.set_subtitle(text)
    
    def clear(self):
        """Alias for clear_subtitle for backward compatibility."""
        self.clear_subtitle()
    
    def __del__(self):
        """Cleanup when the controller is destroyed."""
        self.stop()
