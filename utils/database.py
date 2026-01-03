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
            cur.execute("PRAGMA synchronous = " + str(self.cfg["synchronous"]))
            cur.execute("PRAGMA cache_size = " + str(self.cfg["cache_size"]))
            cur.execute("PRAGMA temp_store = " + str(self.cfg["temp_store"]))
            cur.execute("PRAGMA journal_mode = " + str(self.cfg["journal_mode"]))

            cur.execute("PRAGMA table_info(s2s_training_data)")
            columns = [row[1] for row in cur.fetchall()]

            if not columns:
                cur = self.create(cur)
            
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"[ERROR] Database init error: {e}")
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

            stmt = "INSERT INTO s2s_training_data (transcript, audio_blob, is_reviewed, is_trained) VALUES (?, ?, 0, 0)"

            cur.execute(stmt, (transcript, blob))

            self.connection.commit()

            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Database save error: {e}")
            return False
    
    def get_stats(self):
        """Get training statistics"""
        try:
            cur = self.connection.cursor()
            
            cur.execute("SELECT COUNT(*) as total FROM s2s_training_data")
            total = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) as reviewed FROM s2s_training_data WHERE is_reviewed = 1")
            reviewed = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) as trained FROM s2s_training_data WHERE is_trained = 1")
            trained = cur.fetchone()[0]
            
            return {
                'total': total,
                'reviewed': reviewed,
                'trained': trained,
                'pending_training': reviewed - trained
            }
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to get stats: {e}")
            return None
    
    def get_all_items(self, reviewed_filter=None, trained_filter=None):
        """Get all training items with optional filters"""
        try:
            cur = self.connection.cursor()
            
            query = "SELECT id, transcript, is_reviewed, is_trained, timestamp FROM s2s_training_data"
            conditions = []
            
            if reviewed_filter is not None:
                if reviewed_filter:
                    conditions.append("is_reviewed = 1")
                else:
                    conditions.append("(is_reviewed = 0 OR is_reviewed IS NULL)")
            
            if trained_filter is not None:
                if trained_filter:
                    conditions.append("is_trained = 1")
                else:
                    conditions.append("(is_trained = 0 OR is_trained IS NULL)")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY id DESC"
            
            cur.execute(query)
            rows = cur.fetchall()
            
            items = []
            for row in rows:
                items.append({
                    'id': row[0],
                    'transcript': row[1],
                    'is_reviewed': bool(row[2]),
                    'is_trained': bool(row[3]),
                    'timestamp': row[4]
                })
            
            return items
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to get all items: {e}")
            return None
    
    def get_item_by_id(self, item_id):
        """Get specific training item by ID"""
        try:
            cur = self.connection.cursor()
            
            cur.execute("""
                SELECT id, transcript, audio_blob, is_reviewed, is_trained, timestamp 
                FROM s2s_training_data WHERE id = ?
            """, (item_id,))
            
            row = cur.fetchone()
            
            if not row:
                return None
            
            return {
                'id': row[0],
                'transcript': row[1],
                'audio_blob': row[2],
                'is_reviewed': bool(row[3]),
                'is_trained': bool(row[4]),
                'timestamp': row[5]
            }
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to get item: {e}")
            return None
    
    def update_item(self, item_id, transcript, is_reviewed):
        """Update training item transcript and review status"""
        try:
            cur = self.connection.cursor()
            
            cur.execute("""
                UPDATE s2s_training_data 
                SET transcript = ?, is_reviewed = ? 
                WHERE id = ?
            """, (transcript, 1 if is_reviewed else 0, item_id))
            
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to update item: {e}")
            return False
    
    def delete_item(self, item_id):
        """Delete a training item"""
        try:
            cur = self.connection.cursor()
            cur.execute("DELETE FROM s2s_training_data WHERE id = ?", (item_id,))
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to delete item: {e}")
            return False
    
    def reset_table(self):
        """Drop and recreate the training data table"""
        try:
            cur = self.connection.cursor()
            
            # Drop the table
            cur.execute("DROP TABLE IF EXISTS s2s_training_data")
            
            # Recreate the table
            cur = self.create(cur)
            
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to reset table: {e}")
            return False
    
    def get_training_items(self):
        """Get all reviewed but untrained items for training"""
        try:
            cur = self.connection.cursor()
            
            cur.execute("""
                SELECT id, transcript, audio_blob 
                FROM s2s_training_data 
                WHERE is_reviewed = 1 AND is_trained = 0
            """)
            
            rows = cur.fetchall()
            
            items = []
            for row in rows:
                items.append({
                    'id': row[0],
                    'transcript': row[1],
                    'audio_blob': row[2]
                })
            
            return items
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to get training items: {e}")
            return None
    
    def mark_items_trained(self, item_ids):
        """Mark items as trained"""
        try:
            cur = self.connection.cursor()
            
            for item_id in item_ids:
                cur.execute("""
                    UPDATE s2s_training_data 
                    SET is_trained = 1 
                    WHERE id = ?
                """, (item_id,))
            
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Failed to mark items trained: {e}")
            return False
        
    def __del__(self):
        self.close()