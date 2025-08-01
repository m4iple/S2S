"""
Debug module for S2S - Performance timing and debugging utilities
"""
import time
from collections import defaultdict
from contextlib import contextmanager

# Global debug flag
DEBUG = True

class DebugTimer:
    """Performance timing collector with statistics"""
    
    def __init__(self):
        self.current_session = {}  # Current timing session
        self.history = defaultdict(list)  # Historical timing data
        self.active_timers = {}  # Currently running timers
    
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
            'complete', 'buffer_prep', 'transcription_total', 'whisper', 'tts', 'resample', 'audio_mod', 'buffer_ops'
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
