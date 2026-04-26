"""
hand_tracking.py — Virtual mouse control via webcam hand tracking for Kira.

Architecture
────────────
Uses a background daemon THREAD (not a subprocess) to capture webcam frames
and translate MediaPipe hand landmarks into real-time mouse actions.

Threading is used instead of multiprocessing because Windows' "spawn" mode
creates a fresh Python interpreter for child processes, which silently fails
to initialize OpenCV camera handles — a known issue on Windows 10/11 with
DirectShow and MSMF backends.

Gesture Vocabulary
──────────────────
  • Index finger only  → Move cursor (smoothed)
  • Index + thumb pinch → Left click (tap) / Drag (hold)
  • Middle + thumb pinch → Right click
  • Index + middle up   → Scroll mode (vertical)
  • Open palm (5 fingers) → Pause tracking (hands-free freeze)
  • Fist (0 fingers)    → Resume tracking

Camera Acquisition
──────────────────
  1. Stops the gesture daemon if it holds the camera.
  2. Tries the configured VIRTUAL_MOUSE_CAMERA_INDEX first.
  3. Falls back through indices [0, 1, 2] with DirectShow → default backend.
  4. Validates each camera with a real frame read (isOpened alone is unreliable).
  5. Retries up to 3 times with exponential backoff.
  6. Sends status updates (ready / error) through a thread-safe queue.
"""

import logging
import math
import queue
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

from config import (
    PINCH_THRESHOLD,
    VIRTUAL_MOUSE_ACTIVE_BORDER,
    VIRTUAL_MOUSE_CAMERA_INDEX,
    VIRTUAL_MOUSE_DRAG_HOLD_SECONDS,
    VIRTUAL_MOUSE_FRAME_HEIGHT,
    VIRTUAL_MOUSE_FRAME_WIDTH,
    VIRTUAL_MOUSE_SCROLL_STEP,
    VIRTUAL_MOUSE_SMOOTHING,
)

logger = logging.getLogger(__name__)

# Disable pyautogui safety features for uninterrupted cursor control
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ── Module-level thread state ────────────────────────────────
_thread: threading.Thread | None = None
_stop_event = threading.Event()
_status_queue: queue.Queue = queue.Queue()


# ══════════════════════════════════════════════════════════════
#  CAMERA UTILITIES
# ══════════════════════════════════════════════════════════════

def _acquire_camera(stop_event: threading.Event, max_retries: int = 3):
    """
    Acquires a verified, working camera handle.

    Returns a cv2.VideoCapture object that has been proven to deliver
    real frames, or None if all attempts fail.

    Strategy:
      - Try the configured index first, then fallback indices.
      - For each index, try DirectShow (most reliable on Windows), then default.
      - Verify each candidate with an actual frame read.
      - Verify the camera is not a "dummy" or "virtual" static image using variance and inter-frame noise.
      - Retry the full cycle with exponential backoff.
    """
    indices = [VIRTUAL_MOUSE_CAMERA_INDEX]
    for alt in [0, 1, 2]:
        if alt not in indices:
            indices.append(alt)

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("Default", None),
    ]

    for attempt in range(1, max_retries + 1):
        if stop_event.is_set():
            return None

        for idx in indices:
            for name, flag in backends:
                try:
                    cap = cv2.VideoCapture(idx, flag) if flag is not None else cv2.VideoCapture(idx)
                except Exception as e:
                    logger.debug("Camera(%d, %s) constructor failed: %s", idx, name, e)
                    continue

                if not cap.isOpened():
                    cap.release()
                    logger.debug("Camera(%d, %s) isOpened=False on attempt %d.", idx, name, attempt)
                    continue

                # isOpened() is unreliable on Windows — verify with a real frame
                # Warm up slightly to let the camera start streaming
                for _ in range(5):
                    cap.read()
                    time.sleep(0.02)
                
                ok1, frame1 = cap.read()
                time.sleep(0.05)
                ok2, frame2 = cap.read()
                
                if not ok1 or not ok2 or frame1 is None or frame2 is None:
                    logger.warning("Camera(%d, %s) opened but read() failed — phantom device (attempt %d).", idx, name, attempt)
                    cap.release()
                    continue

                # Check if it's a dummy virtual camera (e.g. solid black)
                variance = np.var(frame1)
                if variance < 1.0:
                    logger.warning("Camera(%d, %s) is a DUMMY/BLANK virtual camera (Variance: %.2f). Skipping.", idx, name, variance)
                    cap.release()
                    continue
                    
                # Check if it's a static image (e.g. OBS placeholder)
                diff = cv2.absdiff(frame1, frame2)
                noise_level = np.mean(diff)
                if noise_level == 0.0:
                    logger.warning("Camera(%d, %s) is a STATIC IMAGE virtual camera (Noise: 0.0). Skipping.", idx, name)
                    cap.release()
                    continue

                logger.info("Camera VERIFIED LIVE: index=%d, backend=%s, frame=%s, variance=%.2f, noise=%.4f", idx, name, frame1.shape, variance, noise_level)
                return cap

        # All indices/backends failed this round — backoff and retry
        delay = 1.0 * attempt
        logger.warning("Camera acquisition failed on attempt %d/%d. Retrying in %.1fs...", attempt, max_retries, delay)
        time.sleep(delay)

    logger.error("Camera acquisition exhausted all %d retries across indices %s.", max_retries, indices)
    return None


# ══════════════════════════════════════════════════════════════
#  HAND ANALYSIS UTILITIES
# ══════════════════════════════════════════════════════════════

def _get_finger_states(landmarks):
    """
    Determines which fingers are extended based on landmark positions.
    Returns a dict of booleans and a total count.
    """
    lm = landmarks
    thumb = lm[4].x < lm[3].x
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    return {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
        "count": sum([thumb, index, middle, ring, pinky]),
    }


def _normalize_axis(value: float, border: float) -> float:
    """
    Maps a raw 0–1 coordinate to 0–1 with a dead-zone border on each side.
    Prevents accidental edge-of-screen movements.
    """
    if value <= border:
        return 0.0
    if value >= 1.0 - border:
        return 1.0
    return (value - border) / (1.0 - 2.0 * border)


def _distance(a, b) -> float:
    """Euclidean distance between two landmarks in normalized space."""
    return math.hypot(a.x - b.x, a.y - b.y)


# ══════════════════════════════════════════════════════════════
#  MAIN TRACKING WORKER
# ══════════════════════════════════════════════════════════════

def _hand_tracking_worker():
    """
    Core tracking loop. Runs in a daemon thread.

    Lifecycle:
      1. Acquire + verify camera.
      2. Configure resolution and buffer.
      3. Warm up (discard initial unstable frames).
      4. Signal 'ready' to the caller.
      5. Enter frame loop: detect hand → interpret gesture → move/click/scroll.
      6. On stop or error, release camera and signal status.
    """
    cap = None
    drag_active = False

    try:
        # ── Step 1: Camera acquisition ──
        cap = _acquire_camera(_stop_event)
        if cap is None:
            _status_queue.put({
                "status": "error",
                "message": "Could not open any webcam. Tried multiple indices and backends. Check camera connections and ensure no other app is using the camera.",
            })
            return

        # ── Step 2: Configure camera ──
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIRTUAL_MOUSE_FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIRTUAL_MOUSE_FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

        # ── Step 3: Warm-up ──
        # Discard initial frames so the camera auto-exposes and the sensor stabilizes.
        for i in range(8):
            ok, _ = cap.read()
            if not ok:
                logger.debug("Warm-up frame %d failed.", i)
            time.sleep(0.05)

        # Verify the camera is still healthy after warm-up
        ok, test = cap.read()
        if not ok or test is None:
            _status_queue.put({
                "status": "error",
                "message": "Camera opened but stopped delivering frames during warm-up.",
            })
            cap.release()
            return

        # ── Step 4: Signal ready ──
        _status_queue.put({"status": "ready"})
        logger.info("Virtual mouse is active. Entering tracking loop.")

        # ── Step 5: Tracking state ──
        screen_w, screen_h = pyautogui.size()
        prev_x = screen_w / 2
        prev_y = screen_h / 2
        pinch_active = False
        right_click_latched = False
        drag_active = False
        pinch_started_at = 0.0
        scroll_anchor_y = None
        last_hand_seen = time.time()
        tracking_paused = False  # Open palm pauses, fist resumes
        consecutive_failures = 0

        mp_hands = mp.solutions.hands
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            model_complexity=1,
        ) as hands:
            while not _stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    consecutive_failures += 1
                    if consecutive_failures > 30:
                        logger.error("Camera delivered 30+ consecutive bad frames. Aborting.")
                        _status_queue.put({"status": "error", "message": "Camera disconnected during tracking."})
                        return
                    time.sleep(0.03)
                    continue
                consecutive_failures = 0

                # Mirror the frame so hand movements feel natural
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                # ── No hand detected ──
                if not result.multi_hand_landmarks:
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                    pinch_active = False
                    right_click_latched = False
                    scroll_anchor_y = None
                    # Throttle CPU when no hand is visible
                    if time.time() - last_hand_seen > 0.5:
                        time.sleep(0.01)
                    continue

                last_hand_seen = time.time()
                landmarks = result.multi_hand_landmarks[0].landmark
                fingers = _get_finger_states(landmarks)

                # ── Pause / Resume via open palm / fist ──
                if fingers["count"] == 5:
                    if not tracking_paused:
                        tracking_paused = True
                        logger.debug("Tracking paused (open palm).")
                        if drag_active:
                            pyautogui.mouseUp()
                            drag_active = False
                    time.sleep(0.05)
                    continue

                if fingers["count"] == 0:
                    if tracking_paused:
                        tracking_paused = False
                        logger.debug("Tracking resumed (fist).")
                    time.sleep(0.05)
                    continue

                if tracking_paused:
                    time.sleep(0.05)
                    continue

                # ── Key landmarks ──
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                thumb_tip = landmarks[4]

                # ── Cursor movement (smoothed) ──
                norm_x = _normalize_axis(index_tip.x, VIRTUAL_MOUSE_ACTIVE_BORDER)
                norm_y = _normalize_axis(index_tip.y, VIRTUAL_MOUSE_ACTIVE_BORDER)
                target_x = norm_x * screen_w
                target_y = norm_y * screen_h

                smooth_x = prev_x + (target_x - prev_x) * VIRTUAL_MOUSE_SMOOTHING
                smooth_y = prev_y + (target_y - prev_y) * VIRTUAL_MOUSE_SMOOTHING
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                prev_x, prev_y = smooth_x, smooth_y

                # ── Pinch distances ──
                pinch_dist = _distance(index_tip, thumb_tip)
                middle_pinch_dist = _distance(middle_tip, thumb_tip)
                now = time.time()

                # ── Scroll mode: index + middle fingers up ──
                scroll_mode = (
                    fingers["index"]
                    and fingers["middle"]
                    and not fingers["ring"]
                    and not fingers["pinky"]
                )
                if scroll_mode:
                    avg_y = (index_tip.y + middle_tip.y) / 2.0
                    if scroll_anchor_y is not None:
                        delta = scroll_anchor_y - avg_y
                        if abs(delta) > 0.015:
                            pyautogui.scroll(int(delta * VIRTUAL_MOUSE_SCROLL_STEP * 10))
                    scroll_anchor_y = avg_y
                else:
                    scroll_anchor_y = None

                # ── Right click: middle + thumb pinch ──
                if middle_pinch_dist < PINCH_THRESHOLD and not right_click_latched and not drag_active:
                    pyautogui.rightClick()
                    right_click_latched = True
                    continue
                if middle_pinch_dist >= PINCH_THRESHOLD:
                    right_click_latched = False

                # ── Left click / drag: index + thumb pinch ──
                if pinch_dist < PINCH_THRESHOLD:
                    if not pinch_active:
                        pinch_active = True
                        pinch_started_at = now
                    elif not drag_active and now - pinch_started_at >= VIRTUAL_MOUSE_DRAG_HOLD_SECONDS:
                        pyautogui.mouseDown()
                        drag_active = True
                else:
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                    elif pinch_active and now - pinch_started_at < VIRTUAL_MOUSE_DRAG_HOLD_SECONDS:
                        pyautogui.click()
                    pinch_active = False

    except Exception:
        logger.error("Virtual mouse tracking crashed.", exc_info=True)
        _status_queue.put({"status": "error", "message": "Virtual mouse crashed unexpectedly."})
    finally:
        # Always release the mouse button and camera on exit
        if drag_active:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        if cap is not None:
            cap.release()
            logger.info("Camera released by virtual mouse.")


# ══════════════════════════════════════════════════════════════
#  PUBLIC API — start / stop
# ══════════════════════════════════════════════════════════════

def start_virtual_mouse():
    """
    Starts the virtual mouse in a background daemon thread.

    Returns:
        (success: bool, message: str) — Result and a human-readable description.
    """
    global _thread

    from ui.status_manager import status_manager

    # ── Already running? ──
    if _thread is not None and _thread.is_alive():
        status_manager.set_virtual_mouse_active(True)
        status_manager.set_virtual_mouse_error("")
        logger.info("Virtual mouse already running.")
        return True, "Virtual mouse is already enabled."

    # ── Release the gesture daemon's camera hold ──
    try:
        from gesture_controller import stop_gesture_daemon, _gesture_thread
        if _gesture_thread is not None and _gesture_thread.is_alive():
            stop_gesture_daemon()
            time.sleep(1.0)  # Allow the OS to fully release the camera handle
            logger.info("Stopped gesture daemon to free camera for virtual mouse.")
    except Exception:
        logger.debug("Could not check/stop gesture daemon.", exc_info=True)

    # ── Prepare fresh state ──
    _stop_event.clear()
    # Drain stale messages from any previous session
    while not _status_queue.empty():
        try:
            _status_queue.get_nowait()
        except queue.Empty:
            break

    # ── Launch worker thread ──
    _thread = threading.Thread(
        target=_hand_tracking_worker,
        daemon=True,
        name="virtual-mouse",
    )
    _thread.start()
    logger.info("Virtual mouse thread started. Waiting for camera initialization...")

    # ── Wait for camera init (up to 15s to account for retries + warm-up) ──
    deadline = time.time() + 15.0
    while time.time() < deadline:
        # Thread died before signalling?
        if _thread is not None and not _thread.is_alive():
            break
        try:
            update = _status_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        if update.get("status") == "ready":
            status_manager.set_virtual_mouse_active(True)
            status_manager.set_virtual_mouse_error("")
            logger.info("Virtual mouse is fully operational.")
            return True, (
                "Virtual mouse enabled. Move your index finger to steer the cursor, "
                "pinch index and thumb to click, hold the pinch to drag, "
                "and raise two fingers to scroll. Open palm pauses, fist resumes."
            )

        if update.get("status") == "error":
            msg = update.get("message", "Virtual mouse failed to start.")
            logger.error("Virtual mouse initialization error: %s", msg)
            stop_virtual_mouse()
            status_manager.set_virtual_mouse_error(msg)
            return False, msg

    # ── Timeout — no ready signal received ──
    logger.error("Virtual mouse timed out waiting for camera (15s).")
    stop_virtual_mouse()
    status_manager.set_virtual_mouse_error("Camera initialization timed out.")
    return False, "Virtual mouse failed to start — camera initialization timed out after 15 seconds."


def stop_virtual_mouse():
    """
    Signals the virtual mouse thread to stop and waits for cleanup.
    Safe to call multiple times or when the mouse is not running.
    """
    global _thread

    from ui.status_manager import status_manager

    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
        if _thread.is_alive():
            logger.warning("Virtual mouse thread did not exit within 5 seconds.")
    _thread = None
    status_manager.set_virtual_mouse_active(False)
    status_manager.set_virtual_mouse_error("")
    logger.info("Virtual mouse stopped.")
