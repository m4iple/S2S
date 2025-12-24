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
import zlib

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
            
            # Optimize database for faster inserts
            cur = self.db_connection.cursor()
            cur.execute("PRAGMA synchronous = NORMAL")  # Faster than FULL, safer than OFF
            cur.execute("PRAGMA cache_size = 10000")    # Increase cache size
            cur.execute("PRAGMA temp_store = MEMORY")   # Use memory for temp operations
            cur.execute("PRAGMA journal_mode = WAL")    # Write-Ahead Logging for better concurrency
            
            # Check if table exists and what columns it has
            cur.execute("PRAGMA table_info(s2s_training_data)")
            columns = [row[1] for row in cur.fetchall()]
            
            if not columns:  # Table doesn't exist
                # Create new table with all columns
                cur.execute("""CREATE TABLE s2s_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript TEXT,
                    audio_blob BLOB,
                    is_reviewed INTEGER,
                    is_trained INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )""")
            else:
                # Table exists, no missing columns to add
                pass
            
            self.db_connection.commit()
        except sqlite3.Error as e:
            print(f"Database initialization error: {e}")
            self.db_connection = None

    def close_database(self):
        """Close database connection"""
        if self.db_connection:
            # Flush any pending commits before closing
            if hasattr(self, '_pending_inserts') and self._pending_inserts > 0:
                try:
                    self.db_connection.commit()
                    self._pending_inserts = 0
                except sqlite3.Error as e:
                    print(f"Error committing pending inserts: {e}")
            
            self.db_connection.close()
            self.db_connection = None
            
            # Clean up cursor reference
            if hasattr(self, '_cursor'):
                delattr(self, '_cursor')

    def save_training_data(self, transcript, audio_data):
        """Save training data (audio and transcript) to the database."""
        # Respect global DATABASE toggle first
        if not DATABASE:
            return False

        # Block ALL database saves when CAPTURE_TRAINING_DATA is False
        if not CAPTURE_TRAINING_DATA:
            return False

        if not self.db_connection:
            return False

        try:
            audio_blob = None
            
            # Convert audio data to bytes if provided (optimized pipeline)
            if audio_data is not None:
                # Optimized conversion pipeline - minimize copies and type checks
                if isinstance(audio_data, torch.Tensor):
                    # Direct conversion to int16 if possible
                    if audio_data.dtype == torch.float32 or audio_data.dtype == torch.float64:
                        # Clip and convert in one operation
                        audio_int16 = torch.clamp(audio_data, -1.0, 1.0).mul_(32767).to(torch.int16).cpu().numpy()
                    else:
                        audio_int16 = audio_data.cpu().numpy().astype(np.int16)
                elif isinstance(audio_data, np.ndarray):
                    if audio_data.dtype in (np.float32, np.float64):
                        # Use in-place operations where possible
                        audio_int16 = np.clip(audio_data, -1.0, 1.0, out=None)
                        audio_int16 = (audio_int16 * 32767).astype(np.int16)
                    elif audio_data.dtype == np.int16:
                        audio_int16 = audio_data  # No copy needed
                    else:
                        audio_int16 = audio_data.astype(np.int16)
                else:
                    return False

                audio_blob = zlib.compress(audio_int16.tobytes(), level=1)  # Fast compression

            # Use existing cursor and prepared statement if available
            if not hasattr(self, '_cursor'):
                self._cursor = self.db_connection.cursor()
                # Prepare the statement once
                self._prepared_stmt = "INSERT INTO s2s_training_data (transcript, audio_blob) VALUES (?, ?)"
                self._pending_inserts = 0
            
            self._cursor.execute(self._prepared_stmt, (transcript, audio_blob))
            self._pending_inserts += 1
            
            # For real-time usage, commit immediately but reuse cursor
            self.db_connection.commit()
            self._pending_inserts = 0
                
            return True
        except sqlite3.Error as e:
            print(f"Database save error: {e}")
            return False

    def flush_pending_commits(self):
        """Manually flush any pending database commits"""
        if not self.db_connection:
            return
        
        if hasattr(self, '_pending_inserts') and self._pending_inserts > 0:
            try:
                self.db_connection.commit()
                self._pending_inserts = 0
            except sqlite3.Error as e:
                print(f"Error flushing pending commits: {e}")

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
    
    def print_session_summary(self, transcribed_text="", save_to_db=False):
        """Print timing summary for the current session
        
        Args:
            transcribed_text: The transcribed text to display and optionally save
            save_to_db: Whether to save this session to the database
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
        
        # Note: save_to_db functionality removed - no longer saving timing data
        
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
        save_to_db: Whether to save this session to the database (deprecated - no longer functional)
        audio: Audio data (parameter kept for compatibility but not used)
    """
    debug_timer.print_session_summary(transcribed_text, save_to_db)

def clear_timing_history():
    """Clear all timing history"""
    debug_timer.clear_history()

def save_session_to_database(transcript="", audio_length=0):
    """Save current timing session to database (deprecated - no longer functional)"""
    print("save_session_to_database is deprecated - timing data is no longer saved to database")
    return False

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
