import collections
import datetime
import logging
import sqlite3
import threading
import time

from config import DB_RETRY_INTERVAL_SECONDS, MAX_RETRY_QUEUE_SIZE

DB_NAME = "assistant_brain.db"
logger = logging.getLogger(__name__)

# ── Retry Queue ──────────────────────────────────────────────
# Thread-safe deque with max size; stores (sql, params) tuples.
_retry_queue = collections.deque(maxlen=MAX_RETRY_QUEUE_SIZE)
_retry_lock = threading.Lock()


def _enqueue_retry(sql, params):
    """Add a failed write to the retry queue.  Drops oldest if full."""
    with _retry_lock:
        if len(_retry_queue) >= MAX_RETRY_QUEUE_SIZE:
            dropped = _retry_queue.popleft()
            logger.critical(
                "Retry queue full (%d). Dropped oldest entry: %s",
                MAX_RETRY_QUEUE_SIZE, dropped[0][:80],
            )
        _retry_queue.append((sql, params))


def _flush_retry_queue():
    """Attempt to replay all queued writes."""
    with _retry_lock:
        pending = list(_retry_queue)
        _retry_queue.clear()

    if not pending:
        return

    logger.info("Flushing %d queued DB writes.", len(pending))
    for sql, params in pending:
        try:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute(sql, params)
            conn.commit()
            conn.close()
        except Exception:
            logger.error("Retry flush failed for: %s", sql[:80], exc_info=True)
            _enqueue_retry(sql, params)


def _retry_worker():
    """Background thread that retries queued DB writes every N seconds."""
    while True:
        try:
            time.sleep(DB_RETRY_INTERVAL_SECONDS)
            _flush_retry_queue()
        except Exception:
            logger.error("DB retry worker error.", exc_info=True)


def start_retry_daemon():
    """Start the background DB retry thread."""
    t = threading.Thread(target=_retry_worker, daemon=True, name="db-retry")
    t.start()


def _safe_write(sql, params):
    """Execute a write, queuing for retry on failure. Never raises."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        conn.close()
    except Exception:
        logger.error("DB write failed; queuing for retry: %s", sql[:80], exc_info=True)
        _enqueue_retry(sql, params)


# ── Init ─────────────────────────────────────────────────────

def init_db():
    "Initializes the database and creates tables if they do not exist."
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL CHECK(role IN ('admin', 'guest')),
            created_at DATETIME NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_admin
        ON users(role) WHERE role = 'admin'
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp DATETIME,
            user_input TEXT,
            ai_response TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS user_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            fact_key TEXT,
            fact_value TEXT,
            UNIQUE(user_id, fact_key)
        )
        """
    )

    cursor.execute(
        """
        CREATE VIEW IF NOT EXISTS admin_memory_view AS
        SELECT * FROM user_memory WHERE user_id = 'admin'
        """
    )

    conn.commit()
    conn.close()
    logger.info("Database initialized (if-not-exists schema).")

    # One-time startup repair: ensure the 'admin' user always has the 'admin' role
    conn2 = sqlite3.connect(DB_NAME)
    cur2 = conn2.cursor()
    cur2.execute("UPDATE users SET role = 'admin' WHERE name = 'admin' AND role != 'admin'")
    if cur2.rowcount > 0:
        logger.warning("Startup repair: corrected 'admin' user role from guest to admin.")
    conn2.commit()
    conn2.close()

    # Start the retry daemon after DB init
    start_retry_daemon()


# ── Conversation Logging (safe write) ────────────────────────

def log_conversation(user_id, user_input, ai_response):
    "Logs a conversation turn, explicitly tied to a user."
    _safe_write(
        "INSERT INTO conversations (user_id, timestamp, user_input, ai_response) VALUES (?, ?, ?, ?)",
        (user_id, datetime.datetime.now(), user_input, ai_response),
    )


def clear_conversation_history(user_id):
    "Deletes all logged conversations for a specific user."
    _safe_write("DELETE FROM conversations WHERE user_id = ?", (user_id,))


# ── Memory (safe write) ─────────────────────────────────────

def save_memory(user_id, key, value):
    "Saves or updates a fact tied to a specific user."
    _safe_write(
        "INSERT OR REPLACE INTO user_memory (user_id, fact_key, fact_value) VALUES (?, ?, ?)",
        (user_id, key, value),
    )


def get_memory(user_id, key):
    "Retrieves a user-scoped memory by key."
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT fact_value FROM user_memory WHERE user_id = ? AND fact_key = ?",
            (user_id, key),
        )
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    except Exception:
        logger.error("DB read failed (get_memory).", exc_info=True)
        return None


def search_memory(user_id, query):
    """Search for a memory whose key is contained in (or contains) the query phrase.

    Returns a (fact_key, fact_value) tuple for the best match, or None.
    Preference order:
        1. Exact key match.
        2. Key that is a substring of the query (longest key wins).
        3. Query that is a substring of a key (shortest key wins).
    """
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()

        # Exact match first
        cursor.execute(
            "SELECT fact_key, fact_value FROM user_memory WHERE user_id = ? AND fact_key = ?",
            (user_id, query),
        )
        row = cursor.fetchone()
        if row:
            conn.close()
            return row

        # Fetch all keys for this user and rank matches in Python
        cursor.execute(
            "SELECT fact_key, fact_value FROM user_memory WHERE user_id = ?",
            (user_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        # Keys that appear inside the query  (e.g. key="favorite color", query="what is my favorite color")
        contained = [(k, v) for k, v in rows if k in query]
        if contained:
            # Longest key = most specific match
            return max(contained, key=lambda pair: len(pair[0]))

        # Query appears inside a key  (e.g. query="color", key="favorite color")
        contains = [(k, v) for k, v in rows if query in k]
        if contains:
            # Shortest key = most specific match
            return min(contains, key=lambda pair: len(pair[0]))

        return None
    except Exception:
        logger.error("DB read failed (search_memory).", exc_info=True)
        return None


def delete_memory(user_id, key):
    "Deletes a single memory key for a user."
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM user_memory WHERE user_id = ? AND fact_key = ?",
            (user_id, key),
        )
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted > 0
    except Exception:
        logger.error("DB write failed (delete_memory).", exc_info=True)
        return False


# ── User Management ──────────────────────────────────────────

def ensure_user(name, role=None):
    "Ensures user exists. First user becomes admin, others guest unless explicit valid role."
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE name = ?", (name,))
        existing = cursor.fetchone()
        if existing:
            if role in ("admin", "guest") and existing[0] != role:
                cursor.execute("UPDATE users SET role = ? WHERE name = ?", (role, name))
                conn.commit()
                logger.warning("Corrected role for '%s': was '%s', now '%s'", name, existing[0], role)
            conn.close()
            return role if role in ("admin", "guest") else existing[0]

        cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        has_admin = cursor.fetchone()[0] > 0
        assigned_role = role if role in ("admin", "guest") else ("guest" if has_admin else "admin")

        cursor.execute(
            "INSERT INTO users (name, role, created_at) VALUES (?, ?, ?)",
            (name, assigned_role, datetime.datetime.now()),
        )
        conn.commit()
        conn.close()
        return assigned_role
    except Exception:
        logger.error("DB write failed (ensure_user).", exc_info=True)
        return "guest"


def get_user_role(name):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT role FROM users WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        logger.error("DB read failed (get_user_role).", exc_info=True)
        return None


def has_admin_user():
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE role = 'admin' LIMIT 1")
        row = cursor.fetchone()
        conn.close()
        return row is not None
    except Exception:
        logger.error("DB read failed (has_admin_user).", exc_info=True)
        return False


# ── Conversation History ─────────────────────────────────────

def list_recent_conversations(user_id, limit=10):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_input, ai_response, timestamp
            FROM conversations
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception:
        logger.error("DB read failed (list_recent_conversations).", exc_info=True)
        return []


def list_context_messages(user_id, turns=6):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT user_input, ai_response
            FROM conversations
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, turns),
        )
        rows = cursor.fetchall()
        conn.close()
        return list(reversed(rows))
    except Exception:
        logger.error("DB read failed (list_context_messages).", exc_info=True)
        return []
