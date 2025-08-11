"""
Debug module for S2S - Performance timing and debugging utilities
"""
import time
from collections import defaultdict
from contextlib import contextmanager
import os
import torch
import numpy as np
import wave
import sqlite3

# Global debug flag
DEBUG = True

class DebugTimer:
    """Performance timing collector with statistics"""
    
    def __init__(self):
        self.current_session = {}  # Current timing session
        self.history = defaultdict(list)  # Historical timing data
        self.active_timers = {}  # Currently running timers
        self.temp_path = "./.temp"
        self.db_connection = sqlite3.connect("database.db")
        cur = self.db_connection.cursor()
        cur.execute("CREATE TABLE IF NOT EXIST s2s_transcribt (id VARCHAR(50) PRIMARY KEY, transcribt TEXT, audio_lenght INT, timings TEXT, timestamp DATETIME)")


    def save_database_data(self, data):
        cur = self.db_connection.cursor()
        cur.executemany("INSERT INTO s2s_transcribt VALUES (?, ?, ?, ?, ?)")
        self.db_connection.commit()

    def start_timer(self, name):
        """Start a named timer"""
        if not DEBUG:
            return
        self.active_timers[name] = time.time()
    
    def end_timer(self, name):
        """End a named timer and store the duration"""
        if not DEBUG:
            return
        
        if name not in self.active_timers:
            return
        
        duration = (time.time() - self.active_timers[name]) * 1000  # Convert to ms
        self.current_session[name] = duration
        self.history[name].append(duration)
        del self.active_timers[name]
    
    def get_average(self, name):
        """Get average time for a named timer"""
        if not self.history[name]:
            return 0
        return sum(self.history[name]) / len(self.history[name])
    
    def print_session_summary(self, transcribed_text=""):
        """Print timing summary for the current session"""
        if not DEBUG or not self.current_session:
            return
        
        print("\n" + "="*60)
        print(f"DEBUG TIMING SUMMARY")
        if transcribed_text:
            print(f"Text: '{transcribed_text}'")
        print("-" * 60)
        
        # Define the order we want to display timings
        timing_order = [
            'complete', 'buffer_prep', 'transcription_total', 'stt', "stt_text", 'tts', 'resample', 'audio_mod', 'buffer_ops'
        ]
        
        total_measured = 0
        for timer_name in timing_order:
            if timer_name in self.current_session:
                current = self.current_session[timer_name]
                average = self.get_average(timer_name)
                print(f"{timer_name.upper():>12}: {current:6.2f} ms - average {average:6.2f} ms")
                if timer_name != 'complete':  # Don't include complete in the sum
                    total_measured += current
        
        # Show the sum of individual components vs complete time
        if 'complete' in self.current_session and total_measured > 0:
            complete_time = self.current_session['complete']
            unaccounted = complete_time - total_measured
            print("-" * 60)
            print(f"{'MEASURED SUM':>12}: {total_measured:6.2f} ms")
            print(f"{'UNACCOUNTED':>12}: {unaccounted:6.2f} ms ({unaccounted/complete_time*100:.1f}%)")
        
        print("="*60)
        
        # Clear current session for next cycle
        self.current_session.clear()
    
    def clear_history(self):
        """Clear all timing history"""
        self.history.clear()
        self.current_session.clear()
        self.active_timers.clear()
    
    @contextmanager
    def timer(self, name):
        """Context manager for timing code blocks"""
        if not DEBUG:
            yield
            return
        
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name)

    def save_audio(self, audio_data):
        if not DEBUG:
            return

        # Ensure temp path exists
        os.makedirs(self.temp_path, exist_ok=True)

        # Generate unique filename
        temp_file = os.path.join(self.temp_path, f"audio_{int(time.time() * 1000)}.wav")
        try:
            # Handle torch tensor input
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.cpu().numpy()
            # Handle numpy array input
            elif isinstance(audio_data, np.ndarray):
                audio_np = audio_data
            else:
                print(f"Unsupported audio data type: {type(audio_data)}")
                return

            # Ensure audio is 1D (mono)
            if audio_np.ndim > 1:
                # Take first channel if multi-channel
                audio_np = audio_np[:, 0] if audio_np.shape[1] < audio_np.shape[0] else audio_np[0, :]
            
            # Ensure audio is in the range [-1, 1] for float32 data
            if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
                # Clip to prevent overflow
                audio_np = np.clip(audio_np, -1.0, 1.0)
                # Convert to int16
                audio_int16 = np.int16(audio_np * 32767)
            elif audio_np.dtype == np.int16:
                audio_int16 = audio_np
            else:
                # For other types, assume they're in range [-1, 1] and convert
                audio_np = audio_np.astype(np.float32)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = np.int16(audio_np * 32767)

            # Save as WAV file
            with wave.open(temp_file, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)  # 16kHz sample rate (VAD sample rate)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"Audio saved to {temp_file}")
        except Exception as e:
            print(f"Failed to save audio: {e}")

# Global debug timer instance
debug_timer = DebugTimer()

# Convenience functions for easy use
def start_timer(name):
    """Start a named timer"""
    debug_timer.start_timer(name)

def end_timer(name):
    """End a named timer"""
    debug_timer.end_timer(name)

def print_timing_summary(transcribed_text=""):
    """Print timing summary for current session"""
    debug_timer.print_session_summary(transcribed_text)

def clear_timing_history():
    """Clear all timing history"""
    debug_timer.clear_history()

def set_debug(enabled):
    """Enable or disable debug mode"""
    global DEBUG
    DEBUG = enabled

def is_debug_enabled():
    """Check if debug mode is enabled"""
    return DEBUG

# Context manager for timing
def timer(name):
    """Context manager for timing code blocks
    
    Usage:
        with timer('operation_name'):
            # code to time
            pass
    """
    return debug_timer.timer(name)

def debug_save_audio(audio_data):
    """Save audio data to a temporary file"""
    debug_timer.save_audio(audio_data)
