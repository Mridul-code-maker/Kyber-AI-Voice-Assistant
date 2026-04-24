"""
assistant_loop.py — Core orchestrator for the Kyber AI Assistant.
Manages the lifecycle of listening, authenticating, thinking, and speaking.
"""

import os
import time
import logging

import speech_recognition as sr
import speech_engine
from auth.voice_auth import identify_speaker, register_speaker
from core.command_router import route_command, normalize_command
from config import WAKE_WORD
from db import db_engine
from ui.status_manager import AssistantState, status_manager

logger = logging.getLogger(__name__)


def _speak(text, block=True):
    """
    Utility to speak text and manage assistant state transitions.
    If block=True, it waits until speech is complete before returning.
    """
    status_manager.set_state(AssistantState.SPEAKING)
    status_manager.set_last_response(text)
    speech_engine.speak(text, block=block)

    # If blocking, speaking is guaranteed done — force-clear the flag
    if block:
        status_manager.set_speaking(False)
        status_manager.set_gesture_active(True)


has_greeted = False


def _authenticate_from_audio(audio_path):
    """
    Performs voice biometric authentication on the provided audio file.
    Returns (user_id, greeting_message).
    """
    global has_greeted
    if not audio_path or not os.path.exists(audio_path):
        return "guest", ""

    try:
        # Bootstrap admin only if BOTH the DB has no admin AND embeddings are missing.
        from auth.voice_auth import is_admin_registered
        if not db_engine.has_admin_user() and not is_admin_registered():
            logger.warning("Admin bootstrap triggered — registering first voice as admin.")
            try:
                register_speaker("admin", audio_path)
            except Exception:
                logger.error("Admin voice bootstrap failed; creating admin user anyway.", exc_info=True)
            db_engine.ensure_user("admin", role="admin")
            has_greeted = True
            return "admin", "Admin profile created. You are now the administrator."

        speaker = identify_speaker(audio_path)
        if speaker == "admin":
            db_engine.ensure_user("admin", role="admin")
            msg = "" if has_greeted else "Welcome back, admin."
            has_greeted = True
            return "admin", msg
        if speaker == "unknown":
            return "unknown", "Access denied. Voice not recognized."
        
        db_engine.ensure_user(speaker, role="guest")
        msg = "" if has_greeted else f"Welcome back, {speaker}."
        has_greeted = True
        return speaker, msg
    except Exception:
        logger.error("Voice authentication failed.", exc_info=True)
        return "unknown", "Access denied. Authentication error."
    finally:
        # Always clean up the temporary audio file used for authentication
        try:
            os.remove(audio_path)
        except OSError:
            logger.error("Failed to remove temporary auth audio.", exc_info=True)


def _wait_for_speech_done(timeout=30):
    """
    Wait for the TTS engine to finish its current queue, with a safety timeout.
    Prevents the assistant from listening to its own voice.
    """
    deadline = time.time() + timeout
    while status_manager.is_speaking():
        if time.time() > deadline:
            logger.warning("TTS wait timed out after %ds — force-clearing speaking flag.", timeout)
            status_manager.set_speaking(False)
            status_manager.set_gesture_active(True)
            break
        time.sleep(0.1)


def run_assistant_loop():
    """
    Main background loop that listens for the wake word and routes commands.
    This function never returns under normal conditions.
    """
    logger.info("Assistant loop starting.")
    status_manager.set_state(AssistantState.IDLE)
    status_manager.set_user(None)  # No user until voice is authenticated
    
    try:
        # Pre-calibrate to environmental noise floor
        speech_engine.calibrate_microphone(duration=2.5)
        speech_engine.start_calibration_daemon(interval_seconds=60)
    except Exception:
        logger.error("Mic pre-warm failed at loop startup.", exc_info=True)

    while True:
        try:
            # Skip loop if microphone hardware is missing or blocked
            if status_manager.is_mic_unavailable():
                status_manager.set_state(AssistantState.IDLE)
                time.sleep(2)
                continue

            try:
                # Capture a chunk of audio from the microphone
                audio = speech_engine.capture_audio(timeout=3, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                status_manager.set_state(AssistantState.IDLE)
                continue
            except OSError:
                logger.error("Microphone unavailable.", exc_info=True)
                status_manager.set_mic_unavailable(True)
                status_manager.set_state(AssistantState.IDLE)
                time.sleep(2)
                continue

            # Write audio to disk ONCE -- reused for both STT and voice auth.
            audio_path = speech_engine.save_audio_to_temp_file(audio)
            heard = speech_engine.audio_to_text_from_file(audio_path)

            if not heard:
                # No speech detected -- clean up temp file and go idle
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                status_manager.set_state(AssistantState.IDLE)
                continue

            # -- Wake word gate (BEFORE authentication) -----------
            if normalize_command(WAKE_WORD) not in normalize_command(heard):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
                status_manager.set_state(AssistantState.IDLE)
                continue

            # -- Voice authentication (only after wake word confirmed) --
            # _authenticate_from_audio owns the file and will delete it.
            current_user, auth_message = _authenticate_from_audio(audio_path)

            status_manager.set_user(current_user)
            status_manager.set_state(AssistantState.LISTENING)
            
            # Reject unknown voices to maintain security
            if current_user == "unknown":
                _speak("Access denied. Voice not recognized.")
                status_manager.set_state(AssistantState.IDLE)
                continue

            if auth_message:
                _speak(auth_message)

            # Strip the wake word from the command text
            command = normalize_command(heard).replace(normalize_command(WAKE_WORD), "", 1).strip()
            status_manager.set_last_command(command)

            if not command:
                if not auth_message:
                    _speak("Yes? I am listening.", block=True)

                # Follow-up: Give the user time to speak their command without repeating the wake word
                status_manager.set_state(AssistantState.WAITING)
                try:
                    follow_up_audio = speech_engine.capture_audio(timeout=7, phrase_time_limit=15)
                    command = speech_engine.audio_to_text(follow_up_audio)
                    if command:
                        logger.info("Follow-up command received: %s", command)
                        status_manager.set_last_command(command)
                except Exception:
                    command = ""

            if not command:
                status_manager.set_state(AssistantState.IDLE)
                continue

            # Process the command
            status_manager.set_state(AssistantState.THINKING)
            streamed_chunks = []

            def _stream_response_chunk(chunk):
                """Callback to handle streaming text chunks for real-time speech."""
                if not streamed_chunks:
                    status_manager.set_state(AssistantState.SPEAKING)
                streamed_chunks.append(chunk)
                status_manager.set_last_response(" ".join(streamed_chunks).strip())
                speech_engine.speak(chunk, block=False)

            # Route to either local tools or AI brain
            response, from_ai = route_command(command, current_user, on_response_chunk=_stream_response_chunk)

            # Only log AI-generated responses to the context database to avoid pollution
            if from_ai:
                db_engine.log_conversation(current_user, command, response)
            
            if streamed_chunks:
                status_manager.set_last_response(response)
            else:
                _speak(response, block=False)

            # Wait for TTS to finish before returning to idle/listening
            _wait_for_speech_done(timeout=30)

            status_manager.set_state(AssistantState.IDLE)
            time.sleep(0.2)
        except Exception:
            logger.error("Unhandled assistant loop error.", exc_info=True)
            # Force-clear any stuck speaking state for resilience
            status_manager.set_speaking(False)
            status_manager.set_gesture_active(True)
            status_manager.set_state(AssistantState.IDLE)
            continue

