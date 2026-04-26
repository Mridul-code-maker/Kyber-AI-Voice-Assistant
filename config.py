"""
Configuration settings for Kira AI Assistant.
Centralizes all environment variables and constant parameters.
"""

import os
from dotenv import load_dotenv

# Load variables from .env file if it exists
load_dotenv()

# --- Assistant Core Configuration ---
WAKE_WORD = "kira"
CONVERSATION_CONTEXT_TURNS = 3

# --- Networking (Flask / Web UI) ---
FLASK_PORT = 5000
FLASK_HOST = "127.0.0.1"

# --- Speech Recognition and Synthesis ---
TTS_ENGINE = "pyttsx3"
STT_ENGINE = os.getenv("STT_ENGINE", "faster-whisper")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small.en")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda") # Default to GPU for speed
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "4"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.70"))  # Strict for security
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "99"))  # Offload all layers to GPU by default

# --- UI / Overlay ---
STATUS_POLL_INTERVAL = 0.4

# --- Voice Authentication & Biometrics ---
ADMIN_RECORD_SECONDS = 8
EMBEDDINGS_PATH = "embeddings.pkl"

# --- Vision and Hand Gesture Tracking ---
VIRTUAL_MOUSE_SMOOTHING = 0.15
PINCH_THRESHOLD = 0.04
VIRTUAL_MOUSE_CAMERA_INDEX = int(os.getenv("VIRTUAL_MOUSE_CAMERA_INDEX", "2"))  # 0=projector, 1=laptop webcam
VIRTUAL_MOUSE_FRAME_WIDTH = int(os.getenv("VIRTUAL_MOUSE_FRAME_WIDTH", "960"))
VIRTUAL_MOUSE_FRAME_HEIGHT = int(os.getenv("VIRTUAL_MOUSE_FRAME_HEIGHT", "540"))
VIRTUAL_MOUSE_ACTIVE_BORDER = float(os.getenv("VIRTUAL_MOUSE_ACTIVE_BORDER", "0.12"))
VIRTUAL_MOUSE_DRAG_HOLD_SECONDS = float(os.getenv("VIRTUAL_MOUSE_DRAG_HOLD_SECONDS", "0.35"))
VIRTUAL_MOUSE_SCROLL_STEP = int(os.getenv("VIRTUAL_MOUSE_SCROLL_STEP", "90"))

# --- System Paths and Logging ---
LOG_FILE = os.path.join("logs", "kira.log")

# --- Database & Persistence Retry Logic ---
MAX_RETRY_QUEUE_SIZE = 100
DB_RETRY_INTERVAL_SECONDS = 30

