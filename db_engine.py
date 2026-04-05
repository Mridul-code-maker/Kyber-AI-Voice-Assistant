import sqlite3
import datetime

DB_NAME = "assistant_brain.db"

def init_db():
    "Initializes the database and creates tables with RLS contexts."
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Drop existing tables to establish the new RLS schema
    cursor.execute('DROP TABLE IF EXISTS conversations')
    cursor.execute('DROP TABLE IF EXISTS user_memory')
    cursor.execute('DROP VIEW IF EXISTS admin_memory_view')
    
    # Table 1: Conversations Log with Row-Level Context (user_id)
    cursor.execute('''
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp DATETIME,
            user_input TEXT,
            ai_response TEXT
        )
    ''')
    
    # Table 2: User Memory with Row-Level Context (user_id)
    cursor.execute('''
        CREATE TABLE user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            fact_key TEXT,
            fact_value TEXT,
            UNIQUE(user_id, fact_key)
        )
    ''')
    
    # Create a View (DBMS feature) showing how we can restrict data at the DB level
    cursor.execute('''
        CREATE VIEW admin_memory_view AS 
        SELECT * FROM user_memory WHERE user_id = 'admin'
    ''')
    
    conn.commit()
    conn.close()
    print("DBMS Initialized with Row-Level Context (user_id) enabled.")

def log_conversation(user_id, user_input, ai_response):
    "Logs a conversation turn, explicitly tied to a user."
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO conversations (user_id, timestamp, user_input, ai_response) VALUES (?, ?, ?, ?)",
        (user_id, datetime.datetime.now(), user_input, ai_response)
    )
    conn.commit()
    conn.close()

def save_memory(user_id, key, value):
    "Saves or updates a fact tied to a specific user (Application-enforced RLS)."
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR REPLACE INTO user_memory (user_id, fact_key, fact_value) VALUES (?, ?, ?)",
        (user_id, key, value)
    )
    conn.commit()
    conn.close()

def get_memory(user_id, key):
    "Retrieves a fact ONLY if the current user has access to that row (Row-Level Security)."
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT fact_value FROM user_memory WHERE user_id = ? AND fact_key = ?", (user_id, key))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None
