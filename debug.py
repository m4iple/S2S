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
DATABASE = True
CAPTURE_TRAINING_DATA = False

class DebugTimer:
    """Performance timing collector with statistics"""
    
    def __init__(self):
        self.current_session = {}  # Current timing session
        self.history = defaultdict(list)  # Historical timing data
        self.active_timers = {}  # Currently running timers
        self.temp_path = "./.temp"
        self.db_connection = None
        self.db_path = "C:/Users/Aspen/Dev/database/database.db"
        self._init_database()
        

    def _init_database(self):
        """Initialize database connection and create table if it doesn't exist"""
        # Respect the global DATABASE toggle
        if not DATABASE:
            # Ensure no open connection when DB is disabled
            self.db_connection = None
            return

        try:
            self.db_connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cur = self.db_connection.cursor()
            # Make shure the table exists
            cur.execute("""CREATE TABLE IF NOT EXISTS s2s_transcript (
                id VARCHAR(50) PRIMARY KEY, 
                transcript TEXT, 
                audio_length INTEGER, 
                timings TEXT, 
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            cur.execute("""CREATE TABLE IF NOT EXISTS s2s_training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript TEXT,
                audio_blob BLOB,
                is_reviewed INTEGER,
                is_trained INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )""")
            self.db_connection.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            self.db_connection = None

    def save_database_data(self, record_id, transcript, audio_length, timings, timestamp=None):
        """Save transcription data to database
        
        Args:
            record_id: Unique identifier for the record
            transcript: The transcribed text
            audio_length: Length of audio in milliseconds
            timings: JSON string of timing data
            timestamp: Optional timestamp (defaults to current time)
        """
        # Respect global toggle first
        if not DATABASE:
            print("Database disabled (DATABASE=False), not saving data")
            return False

        if not self.db_connection:
            print("Database not initialized, cannot save data")
            return False
            
        try:
            cur = self.db_connection.cursor()
            if timestamp:
                cur.execute("INSERT INTO s2s_transcript VALUES (?, ?, ?, ?, ?)", 
                           (record_id, transcript, audio_length, timings, timestamp))
            else:
                cur.execute("INSERT INTO s2s_transcript (id, transcript, audio_length, timings) VALUES (?, ?, ?, ?)", 
                           (record_id, transcript, audio_length, timings))
            self.db_connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"Database save error: {e}")
            return False

    def close_database(self):
        """Close database connection"""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None

    def get_transcription_history(self, limit=10):
        """Get recent transcription history from database
        
        Args:
            limit: Maximum number of records to retrieve
            
        Returns:
            List of tuples containing (id, transcript, audio_length, timings, timestamp)
        """
        if not DATABASE:
            print("Database disabled (DATABASE=False), no history available")
            return []

        if not self.db_connection:
            print("Database not initialized")
            return []
            
        try:
            cur = self.db_connection.cursor()
            cur.execute("SELECT * FROM s2s_transcript ORDER BY timestamp DESC LIMIT ?", (limit,))
            return cur.fetchall()
        except sqlite3.Error as e:
            print(f"Database query error: {e}")
            return []

    def save_session_to_database(self, transcript="", audio_length=0):
        """Save current timing session to database with auto-generated ID"""
        # Respect global toggle
        if not DATABASE:
            print("Database disabled (DATABASE=False), not saving session")
            return False

        if not self.current_session:
            return False
            
        import json
        import uuid
        
        record_id = str(uuid.uuid4())
        timings_json = json.dumps(self.current_session)
        
        return self.save_database_data(record_id, transcript, audio_length, timings_json)

    def save_training_data(self, transcript, audio_data):
        """Save training data (audio and transcript) to the database."""
        if not CAPTURE_TRAINING_DATA:
            return False

        if not self.db_connection:
            print("Database not initialized, cannot save training data")
            return False

        try:
            # Convert audio data to bytes
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.cpu().numpy()
            elif isinstance(audio_data, np.ndarray):
                audio_np = audio_data
            else:
                print(f"Unsupported audio data type for training data: {type(audio_data)}")
                return False
            
            # Ensure audio is in int16 format before saving as blob
            if audio_np.dtype == np.float32 or audio_np.dtype == np.float64:
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = np.int16(audio_np * 32767)
            elif audio_np.dtype == np.int16:
                audio_int16 = audio_np
            else:
                audio_np = audio_np.astype(np.float32)
                audio_np = np.clip(audio_np, -1.0, 1.0)
                audio_int16 = np.int16(audio_np * 32767)

            audio_blob = audio_int16.tobytes()

            cur = self.db_connection.cursor()
            cur.execute("INSERT INTO s2s_training_data (transcript, audio_blob) VALUES (?, ?)",
                       (transcript, audio_blob))
            self.db_connection.commit()
            print("Training data saved to database.")
            return True
        except sqlite3.Error as e:
            print(f"Database save error for training data: {e}")
            return False

    def __del__(self):
        """Cleanup database connection when object is destroyed"""
        self.close_database()

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
    
    def print_session_summary(self, transcribed_text="", save_to_db=False, audio_length=0):
        """Print timing summary for the current session
        
        Args:
            transcribed_text: The transcribed text to display and optionally save
            save_to_db: Whether to save this session to the database
            audio_length: Length of the audio in milliseconds (for database)
        """
        if not DEBUG or not self.current_session:
            return
        
        print("\n" + "="*60)
        print(f"DEBUG TIMING SUMMARY")
        if transcribed_text:
            print(f"Text: '{transcribed_text}'")
        print("-" * 60)
        
        # Define the order we want to display timings
        timing_order = [
            'complete', 'buffer_prep', 'transcription_total', 'stt', "stt_text", 'tts', 'resample', 'audio_mod', 'buffer_ops', 'training'
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
        
        # Save to database if requested
        if save_to_db:
            if self.save_session_to_database(transcribed_text, audio_length):
                print("Item saved to database")
            else:
                print("Failed to save session to database")
        
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

def print_timing_summary(transcribed_text="", save_to_db=False, audio=None):
    """Print timing summary for current session
    
    Args:
        transcribed_text: The transcribed text to display
        save_to_db: Whether to save this session to the database
        audio_length: Length of the audio in milliseconds
    """
    audio_length = calculate_audio_length(audio) if audio is not None else 0

    debug_timer.print_session_summary(transcribed_text, save_to_db, audio_length)

def clear_timing_history():
    """Clear all timing history"""
    debug_timer.clear_history()

def save_session_to_database(transcript="", audio_length=0):
    """Save current timing session to database"""
    return debug_timer.save_session_to_database(transcript, audio_length)

def close_debug_database():
    """Close the debug database connection"""
    debug_timer.close_database()

def set_debug(enabled):
    """Enable or disable debug mode"""
    global DEBUG
    DEBUG = enabled

def is_debug_enabled():
    """Check if debug mode is enabled"""
    return DEBUG


def set_database(enabled: bool):
    """Enable or disable database usage globally.

    When enabling, attempt to initialize the DB connection. When disabling,
    any open connection will be closed.
    """
    global DATABASE
    DATABASE = bool(enabled)

    # Apply change to the running debug_timer instance
    try:
        if DATABASE:
            # Initialize DB if not already connected
            if debug_timer.db_connection is None:
                debug_timer._init_database()
        else:
            # Close any open connection when disabling
            debug_timer.close_database()
    except Exception:
        # Keep toggle best-effort and avoid throwing from this helper
        pass


def is_database_enabled():
    """Return whether the global database toggle is enabled."""
    return DATABASE


def set_capture_training_data(enabled: bool):
    """Enable or disable capturing of training data."""
    global CAPTURE_TRAINING_DATA
    CAPTURE_TRAINING_DATA = bool(enabled)


def is_capture_training_data_enabled():
    """Return whether capturing training data is enabled."""
    return CAPTURE_TRAINING_DATA


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

def save_training_data(transcript, audio_data):
    """Save training data to the database."""
    return debug_timer.save_training_data(transcript, audio_data)

def calculate_audio_length(audio):
    """Calculates the length of the given audio in milliseconds (assumes 16kHz sample rate)"""
    if audio is None:
        return 0
    # Handle torch tensor or numpy array
    if isinstance(audio, torch.Tensor):
        num_samples = audio.numel()
    elif isinstance(audio, np.ndarray):
        num_samples = audio.size
    else:
        return 0
    
    duration_ms = (num_samples / 48000) * 1000 # use stream sample rate
    return int(duration_ms)
