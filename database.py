import sqlite3
import datetime

DB_NAME = "database/attendance.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create Users Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            roll_number TEXT UNIQUE NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Migrate existing users table if 'email' missing
    try:
        cursor.execute("SELECT email FROM users LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE users ADD COLUMN email TEXT")

    # Create Attendance Table (Supports Migration)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            date TEXT,
            type TEXT DEFAULT 'IN',
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    
    # Migrate existing table if 'type' missing
    try:
        cursor.execute("SELECT type FROM attendance LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE attendance ADD COLUMN type TEXT DEFAULT 'IN'")
    
    conn.commit()
    conn.close()

def add_user(name, roll_number, email):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, roll_number, email) VALUES (?, ?, ?)", (name, roll_number, email))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        return None

def mark_attendance(user_id, force_status=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d")
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Get last record for today
    cursor.execute("SELECT type, timestamp FROM attendance WHERE user_id = ? AND date = ? ORDER BY id DESC LIMIT 1", (user_id, today))
    last_record = cursor.fetchone()
    
    new_type = 'IN'
    
    # Logic:
    # 1. If force_status is set (IN/OUT), try to use it.
    # 2. But still respect debounce (don't allow spamming).
    
    if last_record:
        last_type, last_time_str = last_record
        try:
            last_time = datetime.datetime.strptime(last_time_str, "%Y-%m-%d %H:%M:%S")
            if (now - last_time).total_seconds() < 10:
                conn.close()
                return False, f"Wait 10s ({last_type})"
        except:
            pass 

        if force_status:
            new_type = force_status
        else:
            # Auto Toggle
            if last_type == 'IN': new_type = 'OUT'
            else: new_type = 'IN'
    else:
        # No record today
        if force_status: new_type = force_status
        else: new_type = 'IN'
        
    cursor.execute("INSERT INTO attendance (user_id, date, timestamp, type) VALUES (?, ?, ?, ?)", (user_id, today, timestamp_str, new_type))
    conn.commit()
    conn.close()
    return True, new_type

def get_logs():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.id, u.name, u.roll_number, a.timestamp, a.type
        FROM attendance a 
        JOIN users u ON a.user_id = u.id 
        ORDER BY a.timestamp DESC
    ''')
    data = cursor.fetchall()
    conn.close()
    
    formatted_data = []
    for row in data:
        log_id, name, roll, time_str, log_type = row
        try:
            dt = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
            nice_time = dt.strftime("%B %d, %Y - %I:%M %p") 
        except (ValueError, TypeError):
             nice_time = str(time_str)
        
        if not log_type: log_type = "IN" # Default for old logs
        
        formatted_data.append((log_id, name, roll, nice_time, log_type))
        
    return formatted_data

def get_attendance_report(date_filter=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    query = '''
        SELECT u.name, u.roll_number, a.date, min(a.timestamp) as first_in, max(a.timestamp) as last_out, count(a.id) as punches
        FROM attendance a
        JOIN users u ON a.user_id = u.id
    '''
    
    params = []
    if date_filter:
        query += " WHERE a.date = ?"
        params.append(date_filter)
        
    query += " GROUP BY u.id, a.date ORDER BY a.date DESC, u.name ASC"
    
    cursor.execute(query, params)
    data = cursor.fetchall()
    conn.close()
    
    # Process duration
    processed_data = []
    for row in data:
        name, roll, date_val, first_in, last_out, punches = row
        
        duration_str = "-"
        if punches > 1 and first_in and last_out:
            try:
                t1 = datetime.datetime.strptime(first_in, "%Y-%m-%d %H:%M:%S")
                t2 = datetime.datetime.strptime(last_out, "%Y-%m-%d %H:%M:%S")
                delta = t2 - t1
                total_seconds = delta.total_seconds()
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                duration_str = f"{hours}h {minutes}m"
            except:
                pass
                
        processed_data.append((name, roll, date_val, first_in, last_out, punches, duration_str))
        
    return processed_data

def delete_log(log_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id = ?", (log_id,))
    conn.commit()
    conn.close()

def delete_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, roll_number, created_at, email FROM users ORDER BY id DESC")
    data = cursor.fetchall()
    conn.close()
    return data

def get_user_email(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users WHERE id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None
