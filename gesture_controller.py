"""
gesture_controller.py — Background gesture recognition daemon for Kyber.

Runs as a daemon thread. Uses MediaPipe Hands to detect gestures:
  - Open palm   → Toggle listening
  - Index finger → Volume +10%
  - Peace sign  → Switch user
  - Fist        → Stop TTS

Gesture detection is disabled during speech (via status_manager.gesture_active).
"""

import logging
import threading
import time

from config import VIRTUAL_MOUSE_CAMERA_INDEX

logger = logging.getLogger(__name__)

# Global thread state
_gesture_thread = None
_stop_event = threading.Event()


def _count_extended_fingers(hand_landmarks):
    """
    Analyzes hand landmarks to count how many fingers are extended.
    Returns a dictionary with boolean status for each finger and a total count.
    """
    lm = hand_landmarks.landmark

    # Thumb: tip (4) vs IP (3) — x-axis for right hand
    thumb = lm[4].x < lm[3].x

    # Index: tip (8) vs PIP (6)
    index = lm[8].y < lm[6].y

    # Middle: tip (12) vs PIP (10)
    middle = lm[12].y < lm[10].y

    # Ring: tip (16) vs PIP (14)
    ring = lm[16].y < lm[14].y

    # Pinky: tip (20) vs PIP (18)
    pinky = lm[20].y < lm[18].y

    return {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
        "count": sum([thumb, index, middle, ring, pinky]),
    }


def _classify_gesture(fingers):
    """
    Maps finger counts and patterns to high-level semantic gestures.
    """
    c = fingers["count"]

    # Open palm — all 5 fingers extended
    if c == 5:
        return "open_palm"

    # Fist — 0 fingers extended
    if c == 0:
        return "fist"

    # Index finger — only index extended
    if c == 1 and fingers["index"]:
        return "index_finger"

    # Peace sign — index + middle extended
    if c == 2 and fingers["index"] and fingers["middle"]:
        return "peace_sign"

    return None


def _gesture_worker():
    """
    Main background loop that captures webcam frames and executes commands based on detected gestures.
    Includes logic for cooldowns and state-based pausing (e.g., during speech).
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        logger.error("opencv-python or mediapipe not installed; gesture control disabled.")
        return

    from ui.status_manager import status_manager
    import speech_engine

    mp_hands = mp.solutions.hands
    cap = None
    last_gesture = None
    gesture_cooldown = 0.0  # Prevent rapid-fire duplicate gestures

    try:
        cap = cv2.VideoCapture(VIRTUAL_MOUSE_CAMERA_INDEX)
        if not cap.isOpened():
            logger.warning("Webcam not available; gesture control disabled.")
            return

        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while not _stop_event.is_set():
                try:
                    # Don't conflict with virtual mouse if it's active
                    if status_manager.is_virtual_mouse_active():
                        time.sleep(0.1)
                        continue

                    # Skip if gestures are disabled (e.g., when the assistant is speaking)
                    if not status_manager.is_gesture_active():
                        time.sleep(0.1)
                        continue

                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.1)
                        continue

                    # Mirror the frame for more intuitive interaction
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = hands.process(rgb)

                    if not result.multi_hand_landmarks:
                        last_gesture = None
                        continue

                    hand = result.multi_hand_landmarks[0]
                    fingers = _count_extended_fingers(hand)
                    gesture = _classify_gesture(fingers)

                    now = time.time()
                    # Only act on the gesture if it changed or the cooldown passed
                    if gesture and gesture != last_gesture and now > gesture_cooldown:
                        gesture_cooldown = now + 2.0  # 2-second cooldown for action stability
                        last_gesture = gesture
                        status_manager.set_last_gesture(gesture)
                        logger.info("Gesture detected: %s", gesture)

                        if gesture == "open_palm":
                            # Toggle listening state to trigger wake word manually
                            snap = status_manager.snapshot()
                            if snap["state"] == "idle":
                                status_manager.set_state("listening")
                            else:
                                status_manager.set_state("idle")

                        elif gesture == "index_finger":
                            # Increment system volume
                            current = speech_engine.get_volume()
                            speech_engine.set_volume(min(1.0, current + 0.1))

                        elif gesture == "peace_sign":
                            # Downgrade user to guest (security measure: cannot upgrade to admin via gesture)
                            snap = status_manager.snapshot()
                            if snap["user"] == "admin":
                                status_manager.set_user("guest")
                                logger.info("Gesture switched user to: guest")
                            else:
                                logger.warning("Gesture ignored: cannot switch to admin without voice auth.")

                        elif gesture == "fist":
                            # Emergency stop for any active speech output
                            try:
                                speech_engine.stop_speaking()
                            except Exception:
                                logger.error("Failed to stop TTS via gesture.", exc_info=True)

                    elif gesture is None:
                        last_gesture = None

                except Exception:
                    logger.error("Gesture frame processing error.", exc_info=True)
                    continue

    except Exception:
        logger.error("Gesture controller crashed.", exc_info=True)
    finally:
        if cap is not None:
            cap.release()
        logger.info("Gesture controller stopped.")


def start_gesture_daemon():
    """
    Initializes and starts the gesture controller background thread.
    Idempotent: does nothing if already running.
    """
    global _gesture_thread
    if _gesture_thread and _gesture_thread.is_alive():
        logger.info("Gesture daemon already running.")
        return

    _stop_event.clear()
    _gesture_thread = threading.Thread(target=_gesture_worker, daemon=True, name="gesture-ctrl")
    _gesture_thread.start()
    logger.info("Gesture controller daemon started.")


def stop_gesture_daemon():
    """Signals the gesture controller thread to shut down gracefully."""
    _stop_event.set()
    logger.info("Gesture controller stop requested.")
