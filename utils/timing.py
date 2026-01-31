import time
import os
from collections import defaultdict


class Timing:
    def __init__(self):
        self.current_session = {}
        self.history = defaultdict(list)
        self.active_timers = {}
        self.latency = ()
        self.timing_order = {
            'complete', 'tensor_ops', 'resample', 'vad', 'cpu_ops', 'buffer_prep', 'transcription_total', 'stt', 'stt_text', 'tts', 'audio_mod', 'buffer_ops', 'training'
        }
        self.log_file = os.path.join(os.path.dirname(__file__), '..', '.temp', 'timing.log')
        log_dir = os.path.dirname(self.log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(self.log_file, 'a') as f:
            f.write(f"\n=== New Session: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    def start(self, name):
        """Starts an timer"""
        self.active_timers[name] = time.time()

    def end(self, name):
        """Ends running timer"""
        if name not in self.active_timers:
            return
        
        duration = (time.time() - self.active_timers[name]) * 1000
        self.current_session[name] = duration
        self.history[name].append(duration)
        
        del self.active_timers[name]

    def average(self, name):
        """Get average time for a named timer"""
        if not self.history[name]:
            return 0
        return sum(self.history[name]) / len(self.history[name])
    
    def print_summary(self, transcribed_text=""):
        """Prints timing summary to the std and logs to file"""
        lines = []
        lines.append("\n" + "="*60)
        lines.append(f"TIMING SUMMARY")

        if transcribed_text:
            lines.append(f"Text: '{transcribed_text}'")
        lines.append("-" * 60)

        total_measured = 0
        for timer_name in self.timing_order:
            if timer_name in self.current_session:
                current = self.current_session[timer_name]
                average = self.average(timer_name)

                lines.append(f"{timer_name.upper():>12}: {current:6.2f} ms - average {average:6.2f} ms")

                if timer_name != 'complete':
                    total_measured += current
        
        if 'complete' in self.current_session and total_measured > 0:
            complete_time = self.current_session['complete']
            unaccounted = complete_time - total_measured

            lines.append("-" * 60)
            lines.append(f"{'MEASURED SUM':>12}: {total_measured:6.2f} ms")
            lines.append(f"{'UNACCOUNTED':>12}: {unaccounted:6.2f} ms ({unaccounted/complete_time*100:.1f}%)")

        if self.latency:
            lines.append("-" * 60)
            lines.append(f"LATENCY: {self.latency}")

        lines.append("="*60)

        output = "\n".join(lines)
        print(output)

        with open(self.log_file, 'a') as f:
            f.write(output + "\n")

        self.current_session.clear()

    def clear_history(self):
        """Clear all timing history"""
        self.history.clear()
        self.current_session.clear()
        self.active_timers.clear()

    def set_stream_latency(self, latency):
        self.latency = latency