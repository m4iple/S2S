import sqlite3
from utils.config import load_config
from utils.audio import format_for_database

class Database:

    def __init__(self):
        config =load_config("configs/database.toml")
        self.cfg = config["database"]
        self.connection = None
        pass

    def open(self):
        """Open database connection"""
        try:
            self.connection = sqlite3.connect(self.cfg["path"], check_same_thread=self.cfg["check_same_thread"])

            cur = self.connection.cursor()
            cur.execute("PRAGMA synchronous = " + self.cfg["synchronous"])
            cur.execute("PRAGMA cache_size = " + self.cfg["cache_size"])
            cur.execute("PRAGMA temp_store = " + self.cfg["temp_store"])
            cur.execute("PRAGMA journal_mode = " + self.cfg["journal_mode"])

            cur.execute("PRAGMA table_info(s2s_training_data)")
            columns = [row[1] for row in cur.fetchall()]

            if not columns:
                cur = self.create(cur)
            
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Database init error: {e}")
            self.connection = None

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            
            if hasattr(self, '_cursor'):
                delattr(self, '_cursor')

    def create(self, cur):
        """Create database"""
        cur.execute("""CREATE TABLE s2s_training_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript TEXT,
            audio_blob BLOB,
            is_reviewed INTEGER,
            is_trained INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        return cur
    
    def insert_training_data(self, transcript, audio):
        """Saves the training data"""
        try:
            cur = self.connection.cursor()

            blob = format_for_database(audio)

            stmt = "INSERT INTO s2s_training_data (transcript, audio_blob) VALUES (?, ?)"

            cur.execute(stmt, (transcript, blob))

            self.connection.commit()

            return True
        except sqlite3.Error as e:
            print(f"Database save error: {e}")
            return False
        
    def __del__(self):
        self.close()