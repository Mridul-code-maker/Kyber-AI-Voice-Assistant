"""
app.py — Main application entry point for Kyber AI Assistant.
This file initializes the Flask server, WebSocket communications, and launches background assistant loops.
"""

import atexit
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO

from logger import setup_logging, get_logger

# ── Initialize logging FIRST ─────────────────────────────────
setup_logging()
logger = get_logger(__name__)

import speech_engine
from config import FLASK_HOST, FLASK_PORT, WAKE_WORD
from core.assistant_loop import run_assistant_loop
from db import db_engine
from ui.status_manager import status_manager
from gesture_controller import start_gesture_daemon, stop_gesture_daemon
from vision.hand_tracking import stop_virtual_mouse

app = Flask(__name__)
socketio = SocketIO(app, async_mode="threading")
overlay_process = None


# ── Flask Web Routes ──────────────────────────────────────────

@app.route("/")
def home():
    """Renders the main dashboard UI."""
    return render_template("index.html")


@app.route("/api/set_user", methods=["POST"])
def set_user():
    """Updates the current active user context."""
    data = request.json or {}
    user_id = (data.get("user_id") or "guest").strip().lower()
    status_manager.set_user(user_id)
    return jsonify({"success": True, "current_user": user_id})


@app.route("/api/status", methods=["GET"])
def get_status():
    """Provides a detailed JSON snapshot of the assistant's state and recent history."""
    snap = status_manager.snapshot()
    current_user_id = snap["user"]
    history = db_engine.list_recent_conversations(current_user_id, limit=10)

    return jsonify({
        "state": snap["state"],
        "user": snap["user"],
        "last_command": snap["last_command"],
        "last_response": snap["last_response"],
        "current_user_id": current_user_id,
        "current_brain": snap["current_brain"],
        "mic_unavailable": snap["mic_unavailable"],
        "history": [{"user": row[0], "ai": row[1], "time": row[2]} for row in history],
    })


@app.route("/status", methods=["GET"])
def status():
    """Legacy endpoint for simple status snapshots."""
    snap = status_manager.snapshot()
    return jsonify(snap)


# ── WebSocket Status Emitter ─────────────────────────────────

def _status_emitter():
    """
    Background thread that pushes state updates to the frontend via WebSocket.
    Ensures the dashboard stays synchronized with the assistant's current activity.
    """
    prev_snapshot = {}
    while True:
        try:
            time.sleep(0.5)
            snap = status_manager.snapshot()
            # Only emit if something changed to minimize network traffic
            if snap != prev_snapshot:
                prev_snapshot = snap.copy()
                socketio.emit("status_update", snap)
        except Exception:
            logger.error("WebSocket emitter error.", exc_info=True)


# ── Overlay Lifecycle Management ──────────────────────────────

def _launch_overlay():
    """Spawns the orb_overlay.py script as a separate process to maintain UI responsiveness."""
    global overlay_process
    overlay_script = Path(__file__).with_name("orb_overlay.py")
    if not overlay_script.exists():
        logger.warning("orb_overlay.py not found; skipping overlay startup.")
        return
    overlay_process = subprocess.Popen([sys.executable, str(overlay_script)])
    logger.info("Stitch overlay process launched with PID %s.", overlay_process.pid)


def _shutdown_overlay():
    """Attempts to gracefully terminate the overlay process, falling back to SIGKILL if necessary."""
    global overlay_process
    if overlay_process and overlay_process.poll() is None:
        try:
            logger.info("Terminating Stitch overlay process.")
            overlay_process.terminate()
            overlay_process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            logger.warning("Overlay did not exit cleanly; killing process.")
            overlay_process.kill()
            overlay_process.wait(timeout=3)
        except Exception:
            logger.error("Failed to terminate overlay process cleanly.", exc_info=True)
        finally:
            overlay_process = None


def _handle_shutdown_signal(signum, frame):
    """Signal handler for OS-level termination requests."""
    logger.info("Shutdown signal received: %s", signum)
    stop_virtual_mouse()
    stop_gesture_daemon()
    _shutdown_overlay()
    raise SystemExit(0)


# ── Application Entry Point ──────────────────────────────────

if __name__ == "__main__":
    logger.info("Voice assistant started. Wake word: %s. User: checking...", WAKE_WORD)
    
    # Initialize the database schema if it doesn't exist
    db_engine.init_db()

    try:
        # Initial mic calibration and start periodic re-calibration
        speech_engine.calibrate_microphone(duration=1.0)
        speech_engine.start_calibration_daemon(interval_seconds=60)
    except Exception:
        logger.error("Startup mic pre-warm failed.", exc_info=True)

    # Pre-warm the STT engine in the background to avoid latency on first command
    threading.Thread(target=speech_engine.warm_stt_backend, daemon=True, name="stt-warmup").start()

    # Launch core assistant and WebSocket status threads
    threading.Thread(target=run_assistant_loop, daemon=True).start()
    threading.Thread(target=_status_emitter, daemon=True, name="ws-emitter").start()

    # Gesture controller will be started via voice command to save resources.

    # Startup the visual overlay and register cleanup hooks
    _launch_overlay()
    atexit.register(stop_virtual_mouse)
    atexit.register(stop_gesture_daemon)
    atexit.register(_shutdown_overlay)
    
    # Register OS signals for graceful shutdown
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    
    try:
        # Run the Flask app with WebSocket support
        socketio.run(app, debug=False, port=FLASK_PORT, host=FLASK_HOST, allow_unsafe_werkzeug=True)
    finally:
        # Final cleanup safety net
        stop_virtual_mouse()
        stop_gesture_daemon()
        _shutdown_overlay()
