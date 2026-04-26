"""
status_manager.py — Thread-safe state orchestration for the Kira assistant.
Provides a central repository for the assistant's current operational state.
"""

import threading
import logging

logger = logging.getLogger(__name__)


class AssistantState:
    """Enumeration of possible assistant operational states."""
    IDLE = "idle"
    LISTENING = "listening"
    WAITING = "waiting"
    THINKING = "thinking"
    SPEAKING = "speaking"


class StatusManager:
    """
    Thread-safe state manager for the Kira assistant.
    Coordinates state updates across multiple background threads (assistant loop, 
    gesture controller, and Flask/WebSocket server).
    
    All state transitions are protected by a reentrant-style lock to prevent race conditions.
    """

    def __init__(self):
        self._lock = threading.Lock()
        
        # Core State
        self._state = AssistantState.IDLE
        self._user = "guest"
        self._last_command = ""
        self._last_response = ""
        
        # System Health & Hardware State
        self._current_brain = "gemma2"
        self._gesture_active = True         # Controls whether the camera/gesture thread is processing
        self._mic_unavailable = False       # Flag for hardware failures
        self._last_gesture = ""             # The most recently detected hand sign
        self._speaking = False              # Boolean flag synchronized with the TTS engine
        self._stt_degraded = False          # Indicates fallback to lower-quality STT models
        self._virtual_mouse_active = False  # Track if hand-to-mouse tracking is running
        self._virtual_mouse_error = ""

    # ── State Setters ──────────────────────────────────────────

    def set_state(self, state):
        """Updates the high-level assistant state (e.g., IDLE -> LISTENING)."""
        with self._lock:
            old_state = self._state
            self._state = state
        if old_state != state:
            logger.info("State: %s -> %s", old_state, state)

    def set_user(self, user):
        """Updates the currently authenticated user identity."""
        with self._lock:
            self._user = user

    def set_last_command(self, command):
        """Stores the most recent transcribed text command."""
        with self._lock:
            self._last_command = command

    def set_last_response(self, response):
        """Stores the most recent AI or system response text."""
        with self._lock:
            self._last_response = response

    def set_current_brain(self, brain):
        """Sets the active AI model being used for generation."""
        with self._lock:
            self._current_brain = brain

    def set_gesture_active(self, active: bool):
        """Enables or disables gesture processing (e.g., disable during speech)."""
        with self._lock:
            self._gesture_active = active

    def set_mic_unavailable(self, unavailable: bool):
        """Sets the microphone health flag."""
        with self._lock:
            self._mic_unavailable = unavailable

    def set_last_gesture(self, gesture: str):
        """Records the latest detected hand gesture name."""
        with self._lock:
            self._last_gesture = gesture

    def set_speaking(self, speaking: bool):
        """Updates the TTS activity flag."""
        with self._lock:
            self._speaking = speaking

    def set_stt_degraded(self, degraded: bool):
        """Updates the speech-to-text quality flag."""
        with self._lock:
            self._stt_degraded = degraded

    def set_virtual_mouse_active(self, active: bool):
        """Tracks the activation status of the virtual mouse system."""
        with self._lock:
            self._virtual_mouse_active = active

    def set_virtual_mouse_error(self, error: str):
        """Stores diagnostic error messages from the vision system."""
        with self._lock:
            self._virtual_mouse_error = error

    # ── State Getters ──────────────────────────────────────────

    def is_gesture_active(self) -> bool:
        """Checks if gesture recognition is currently processing frames."""
        with self._lock:
            return self._gesture_active

    def is_mic_unavailable(self) -> bool:
        """Checks if the microphone is in an error state."""
        with self._lock:
            return self._mic_unavailable

    def is_speaking(self) -> bool:
        """Checks if the assistant is currently speaking via TTS."""
        with self._lock:
            return self._speaking

    def get_state(self) -> str:
        """Retrieves the current high-level state string."""
        with self._lock:
            return self._state

    def get_current_brain(self) -> str:
        """Retrieves the name of the currently active AI brain."""
        with self._lock:
            return self._current_brain

    def is_stt_degraded(self) -> bool:
        """Checks if speech-to-text is running in fallback mode."""
        with self._lock:
            return self._stt_degraded

    def is_virtual_mouse_active(self) -> bool:
        """Checks if the hand-tracking virtual mouse is enabled."""
        with self._lock:
            return self._virtual_mouse_active

    # ── Full Snapshot ──────────────────────────────────────────

    def snapshot(self):
        """
        Returns a complete, read-only dictionary of the current state.
        Safe to call from the Flask/WebSocket emitter threads.
        """
        with self._lock:
            return {
                "state": self._state,
                "user": self._user,
                "last_command": self._last_command,
                "last_response": self._last_response,
                "current_brain": self._current_brain,
                "gesture_active": self._gesture_active,
                "mic_unavailable": self._mic_unavailable,
                "last_gesture": self._last_gesture,
                "speaking": self._speaking,
                "stt_degraded": self._stt_degraded,
                "virtual_mouse_active": self._virtual_mouse_active,
                "virtual_mouse_error": self._virtual_mouse_error,
            }


# Singleton instance shared across all application modules
status_manager = StatusManager()
