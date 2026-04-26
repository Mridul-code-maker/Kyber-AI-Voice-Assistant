import logging
import os
import queue
import re
import tempfile
import threading
import time

import speech_recognition as sr

from config import (
    STT_ENGINE,
    TTS_ENGINE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CPU_THREADS,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL_SIZE,
    WAKE_WORD,
)

logger = logging.getLogger(__name__)

# ── TTS State ────────────────────────────────────────────────
# The pyttsx3 engine is created INSIDE the worker thread to avoid
# Windows COM apartment threading issues that cause runAndWait() to
# deadlock.  The module-level variables below are only accessed from
# the worker thread (except _speech_queue / _speech_stop_event which
# are thread-safe).

_speech_queue = queue.Queue()
_speech_stop_event = threading.Event()
_speech_worker_started = False
_speech_worker_lock = threading.Lock()
_tts_volume = 1.0
_tts_volume_lock = threading.Lock()

_whisper_model = None
_whisper_model_lock = threading.Lock()

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.5
recognizer.dynamic_energy_threshold = True
recognizer.dynamic_energy_adjustment_damping = 0.15  # Adapts to fan noise faster
_calibration_lock = threading.Lock()
_last_calibration = 0.0


# ── Volume Control ───────────────────────────────────────────

def set_volume(level: float):
    """Set TTS volume (0.0 to 1.0)."""
    global _tts_volume
    level = max(0.0, min(1.0, level))
    with _tts_volume_lock:
        _tts_volume = level
    logger.info("TTS volume set to %.2f", level)


def get_volume() -> float:
    """Get current TTS volume."""
    with _tts_volume_lock:
        return _tts_volume


# ── TTS ──────────────────────────────────────────────────────

def _split_sentences(text):
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]


def _clear_speech_queue():
    while True:
        try:
            queued = _speech_queue.get_nowait()
        except queue.Empty:
            return
        done_event = queued.get("done_event")
        if done_event:
            done_event.set()
        _speech_queue.task_done()


def _start_speech_worker():
    global _speech_worker_started
    with _speech_worker_lock:
        if _speech_worker_started:
            return
        thread = threading.Thread(target=_speech_worker, daemon=True, name="tts-worker")
        thread.start()
        _speech_worker_started = True


def _speech_worker():
    """Background TTS worker.

    The pyttsx3 engine is created HERE so that all COM calls happen on
    the same thread, which prevents the Windows SAPI deadlock.
    """
    import pyttsx3
    from ui.status_manager import status_manager

    engine = pyttsx3.init()
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)
    engine.setProperty("rate", 160)
    logger.info("TTS engine initialized on worker thread.")

    while True:
        job = _speech_queue.get()
        text = job["text"]
        done_event = job.get("done_event")

        if not text:
            if done_event:
                done_event.set()
            _speech_queue.task_done()
            continue

        status_manager.set_speaking(True)
        status_manager.set_gesture_active(False)

        # Apply current volume setting
        with _tts_volume_lock:
            engine.setProperty("volume", _tts_volume)

        try:
            chunks = _split_sentences(text) if len(text) > 80 else [text]
            for sentence in chunks:
                if _speech_stop_event.is_set():
                    break
                engine.say(sentence)
                engine.runAndWait()
        except Exception:
            logger.error("TTS worker failed while speaking.", exc_info=True)
        finally:
            try:
                engine.stop()
            except Exception:
                logger.debug("TTS engine stop failed during cleanup.", exc_info=True)

            if _speech_queue.empty():
                status_manager.set_speaking(False)
                status_manager.set_gesture_active(True)

            if done_event:
                done_event.set()
            _speech_queue.task_done()


def speak_async(text):
    """Queue text for speech and return immediately."""
    if TTS_ENGINE != "pyttsx3":
        logger.warning("Unsupported TTS engine %s; falling back to pyttsx3.", TTS_ENGINE)
    logger.info("Assistant: %s", text)

    from ui.status_manager import status_manager
    status_manager.set_speaking(True)
    status_manager.set_gesture_active(False)

    _start_speech_worker()
    _speech_stop_event.clear()
    _speech_queue.put({"text": text, "done_event": None})


def speak(text, block=True):
    """Speaks the given text out loud.

    By default this preserves the legacy blocking behavior. Callers that need
    interruption-friendly output can pass block=False or use speak_async().
    """
    if TTS_ENGINE != "pyttsx3":
        logger.warning("Unsupported TTS engine %s; falling back to pyttsx3.", TTS_ENGINE)
    logger.info("Assistant: %s", text)

    from ui.status_manager import status_manager
    status_manager.set_speaking(True)
    status_manager.set_gesture_active(False)

    _start_speech_worker()
    _speech_stop_event.clear()
    done_event = threading.Event() if block else None
    _speech_queue.put({"text": text, "done_event": done_event})
    if done_event:
        done_event.wait()


def stop_speaking():
    """Stop current and queued speech as soon as the TTS driver allows."""
    _speech_stop_event.set()
    _clear_speech_queue()
    from ui.status_manager import status_manager
    status_manager.set_speaking(False)
    status_manager.set_gesture_active(True)


# ── Microphone Calibration ───────────────────────────────────

def calibrate_microphone(duration=1.0):
    global _last_calibration
    try:
        with _calibration_lock:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=duration)
                _last_calibration = time.time()
                logger.info("Microphone calibrated. Energy threshold=%.2f", recognizer.energy_threshold)
        # Mic is working — clear the flag
        from ui.status_manager import status_manager
        status_manager.set_mic_unavailable(False)
    except OSError:
        logger.error("Microphone unavailable during calibration.", exc_info=True)
        from ui.status_manager import status_manager
        status_manager.set_mic_unavailable(True)


def start_calibration_daemon(interval_seconds=60):
    def _worker():
        while True:
            try:
                time.sleep(interval_seconds)
                calibrate_microphone(duration=0.5)
            except Exception:
                logger.error("Background mic calibration failed.", exc_info=True)

    thread = threading.Thread(target=_worker, daemon=True, name="mic-calibration")
    thread.start()


# ── Audio Capture ────────────────────────────────────────────

def capture_audio(timeout=3, phrase_time_limit=8):
    try:
        with sr.Microphone() as source:
            logger.debug("Listening for speech...")
            return recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except OSError:
        logger.error("Microphone unavailable during capture.", exc_info=True)
        from ui.status_manager import status_manager
        status_manager.set_mic_unavailable(True)
        raise


def _get_whisper_model():
    global _whisper_model
    with _whisper_model_lock:
        if _whisper_model is not None:
            return _whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error(
                "faster-whisper is not installed. Install dependencies to enable offline STT."
            )
            return None

        # Attempt to load on configured device (default CUDA), with fallback to CPU
        devices_to_try = [WHISPER_DEVICE]
        if WHISPER_DEVICE != "cpu":
            devices_to_try.append("cpu")

        for device in devices_to_try:
            try:
                compute_type = WHISPER_COMPUTE_TYPE
                # CUDA often requires specific compute types; float32/int8 are usually safe fallbacks
                if device == "cpu" and compute_type not in ["int8", "float32"]:
                    compute_type = "int8"
                
                logger.info(
                    "Loading faster-whisper model '%s' on %s (%s).",
                    WHISPER_MODEL_SIZE,
                    device,
                    compute_type,
                )
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=WHISPER_CPU_THREADS,
                )
                return _whisper_model
            except Exception as e:
                if device == "cpu":
                    logger.error("Failed to initialize faster-whisper even on CPU.", exc_info=True)
                    return None
                logger.warning(
                    "Failed to load faster-whisper on %s. Error: %s. Falling back to CPU...",
                    device, e
                )
        return None


def _transcribe_with_whisper(audio=None, file_path=None):
    """Transcribe audio using faster-whisper.

    Accepts either an ``audio`` object (speech_recognition AudioData) or a
    pre-existing ``file_path`` to a WAV file.  When *file_path* is provided
    the caller owns the file and this function will NOT delete it.
    """
    from ui.status_manager import status_manager

    model = _get_whisper_model()
    if model is None:
        # Surface degraded state so the rest of the app knows STT is down
        status_manager.set_stt_degraded(True)
        logger.warning("STT is degraded: Whisper model unavailable. Transcription skipped.")
        return ""

    # If a file was already written by the caller, reuse it; otherwise write one.
    owns_file = file_path is None
    if owns_file:
        if audio is None:
            logger.error("_transcribe_with_whisper called with no audio and no file_path.")
            return ""
        file_path = save_audio_to_temp_file(audio)

    try:
        segments, info = model.transcribe(
            file_path,
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=True,
            initial_prompt=f"Kira. The assistant's name is {WAKE_WORD}. Please transcribe it as {WAKE_WORD} even if it sounds like Ira, Era, Hera, or Keira."
        )
        # We must exhaust the generator inside the try block to catch runtime CUDA errors
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
        
        if text:
            logger.debug(
                "Recognized locally: %s (language=%s, prob=%.2f)",
                text,
                getattr(info, "language", WHISPER_LANGUAGE),
                getattr(info, "language_probability", 0.0),
            )
        else:
            logger.debug("Local Whisper produced no text.")

        # STT succeeded — clear any previous degraded flag
        status_manager.set_stt_degraded(False)
        return text.lower()
    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cudnn" in str(e).lower() or "cuda" in str(e).lower():
            logger.error("CUDA runtime error during transcription: %s. Forcing CPU fallback.", e)
            global _whisper_model
            with _whisper_model_lock:
                _whisper_model = None  # Clear cache to force reload on CPU
            # We don't recurse here to avoid infinite loops, but the next call will use CPU.
            # However, we can try one more time immediately.
            return _transcribe_with_whisper(audio, file_path)
        logger.error("Inference failed: %s", e, exc_info=True)
        return ""
    except Exception:
        logger.error("Unexpected speech-to-text error during inference.", exc_info=True)
        return ""
    finally:
        # Only clean up files we created ourselves
        if owns_file:
            try:
                os.remove(file_path)
            except OSError:
                logger.debug("Failed to remove temporary STT audio file.", exc_info=True)


def warm_stt_backend():
    """Load the offline STT backend ahead of first use."""
    if STT_ENGINE != "faster-whisper":
        logger.warning("Skipping STT warmup because STT_ENGINE=%s.", STT_ENGINE)
        return
    model = _get_whisper_model()
    if model is None:
        from ui.status_manager import status_manager
        status_manager.set_stt_degraded(True)
        logger.warning("STT warmup failed — assistant will operate in degraded STT mode.")


def audio_to_text(audio):
    """Transcribe an AudioData object to text."""
    try:
        if STT_ENGINE != "faster-whisper":
            logger.warning("Unsupported STT engine %s configured; expected faster-whisper.", STT_ENGINE)
            return ""

        return _transcribe_with_whisper(audio=audio)
    except Exception:
        logger.error("Unexpected speech-to-text error.", exc_info=True)
        return ""


def audio_to_text_from_file(file_path):
    """Transcribe a pre-existing WAV file to text.

    The caller retains ownership of the file — it will NOT be deleted.
    """
    try:
        if STT_ENGINE != "faster-whisper":
            logger.warning("Unsupported STT engine %s configured; expected faster-whisper.", STT_ENGINE)
            return ""

        return _transcribe_with_whisper(file_path=file_path)
    except Exception:
        logger.error("Unexpected speech-to-text error.", exc_info=True)
        return ""


def save_audio_to_temp_file(audio):
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_audio.write(audio.get_wav_data())
    temp_audio.close()
    return temp_audio.name


def listen(return_audio=False):
    "Compatibility helper for older callers."
    try:
        audio = capture_audio(timeout=3, phrase_time_limit=8)
        text = audio_to_text(audio)
        if return_audio:
            return text, save_audio_to_temp_file(audio)
        return text
    except sr.WaitTimeoutError:
        return ("", "") if return_audio else ""
    except OSError:
        logger.error("Microphone unavailable during listen.", exc_info=True)
        return ("", "") if return_audio else ""
    except Exception:
        logger.error("Unexpected listen() failure.", exc_info=True)
        return ("", "") if return_audio else ""

if __name__ == "__main__":
    speak("Hello, the speech engine is working.")
    result = listen()
    if result:
        speak(f"I heard you say: {result}")
