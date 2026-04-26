"""
hand_tracking.py — Virtual mouse control via hand tracking for Kira.

Uses a background THREAD (not a separate process) to capture webcam frames
and translate hand landmarks into mouse movements, clicks, drags, and scrolls.
Threading is used because Windows multiprocessing (spawn mode) causes silent
failures with OpenCV camera initialization in child processes.
"""

import math
import queue
import threading
import time
import logging

import cv2
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

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
mp_hands = mp.solutions.hands

# Global thread state
_thread = None
_stop_event = threading.Event()
_status_queue = queue.Queue()


def _count_extended_fingers(landmarks):
    thumb = landmarks[4].x < landmarks[3].x
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    return {
        "thumb": thumb,
        "index": index,
        "middle": middle,
        "ring": ring,
        "pinky": pinky,
    }


def _normalize_axis(value, border):
    if value <= border:
        return 0.0
    if value >= 1.0 - border:
        return 1.0
    return (value - border) / (1.0 - (2.0 * border))


def _hand_tracking_worker():
    """
    Main hand-tracking loop. Runs in a daemon thread.
    Opens the camera, verifies it delivers real frames, then processes
    hand landmarks to control the mouse cursor.
    """
    cap = None
    MAX_OPEN_RETRIES = 3

    indices_to_try = [VIRTUAL_MOUSE_CAMERA_INDEX]
    for alt in [0, 1, 2]:
        if alt not in indices_to_try:
            indices_to_try.append(alt)

    # --- Camera acquisition with retry and frame verification ---
    for retry in range(MAX_OPEN_RETRIES):
        if _stop_event.is_set():
            return

        for idx in indices_to_try:
            for backend_name, backend_flag in [("DirectShow", cv2.CAP_DSHOW), ("default", None)]:
                if backend_flag is not None:
                    attempt = cv2.VideoCapture(idx, backend_flag)
                else:
                    attempt = cv2.VideoCapture(idx)

                if not attempt.isOpened():
                    attempt.release()
                    logger.debug("Camera index %d (%s) failed isOpened on attempt %d.", idx, backend_name, retry + 1)
                    continue

                # Verify with a real frame read — isOpened() alone is unreliable on Windows
                test_ok, test_frame = attempt.read()
                if not test_ok or test_frame is None:
                    logger.warning(
                        "Camera index %d (%s) opened but test read failed on attempt %d.",
                        idx, backend_name, retry + 1,
                    )
                    attempt.release()
                    continue

                cap = attempt
                logger.info(
                    "Camera VERIFIED on index %d (%s), attempt %d. Frame: %s",
                    idx, backend_name, retry + 1, test_frame.shape,
                )
                break  # backend loop

            if cap is not None:
                break  # index loop

        if cap is not None:
            break  # retry loop

        wait_seconds = 1.0 * (retry + 1)
        logger.warning("All camera indices failed on attempt %d. Retrying in %.1fs...", retry + 1, wait_seconds)
        time.sleep(wait_seconds)

    if cap is None or not cap.isOpened():
        msg = (
            f"Could not open any webcam. Tried indices {indices_to_try} "
            f"x {MAX_OPEN_RETRIES} retries. Check camera connections."
        )
        logger.error(msg)
        _status_queue.put({"status": "error", "message": msg})
        return

    # --- Configure camera ---
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIRTUAL_MOUSE_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIRTUAL_MOUSE_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Warm-up: discard initial frames to let the camera auto-expose
    for _ in range(5):
        cap.read()

    screen_w, screen_h = pyautogui.size()
    prev_x = screen_w / 2
    prev_y = screen_h / 2
    pinch_active = False
    right_click_latched = False
    drag_active = False
    pinch_started_at = 0.0
    scroll_anchor_y = None
    last_seen = time.time()

    # Signal that the camera is ready
    _status_queue.put({"status": "ready"})
    logger.info("Virtual mouse camera ready — entering tracking loop.")

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.65,
            model_complexity=0,
        ) as hands:
            while not _stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if not result.multi_hand_landmarks:
                    if drag_active:
                        pyautogui.mouseUp()
                        drag_active = False
                    pinch_active = False
                    right_click_latched = False
                    scroll_anchor_y = None
                    if time.time() - last_seen > 0.5:
                        time.sleep(0.01)
                    continue

                last_seen = time.time()
                landmarks = result.multi_hand_landmarks[0].landmark
                fingers = _count_extended_fingers(landmarks)

                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                thumb_tip = landmarks[4]
                norm_x = _normalize_axis(index_tip.x, VIRTUAL_MOUSE_ACTIVE_BORDER)
                norm_y = _normalize_axis(index_tip.y, VIRTUAL_MOUSE_ACTIVE_BORDER)
                target_x = norm_x * screen_w
                target_y = norm_y * screen_h

                smooth_x = prev_x + (target_x - prev_x) * VIRTUAL_MOUSE_SMOOTHING
                smooth_y = prev_y + (target_y - prev_y) * VIRTUAL_MOUSE_SMOOTHING
                pyautogui.moveTo(int(smooth_x), int(smooth_y))
                prev_x, prev_y = smooth_x, smooth_y

                pinch_distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                middle_pinch_distance = math.hypot(middle_tip.x - thumb_tip.x, middle_tip.y - thumb_tip.y)
                now = time.time()

                scroll_mode = fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]
                if scroll_mode:
                    average_y = (index_tip.y + middle_tip.y) / 2.0
                    if scroll_anchor_y is not None:
                        delta = scroll_anchor_y - average_y
                        if abs(delta) > 0.015:
                            pyautogui.scroll(int(delta * VIRTUAL_MOUSE_SCROLL_STEP * 10))
                    scroll_anchor_y = average_y
                else:
                    scroll_anchor_y = None

                if middle_pinch_distance < PINCH_THRESHOLD and not right_click_latched and not drag_active:
                    pyautogui.rightClick()
                    right_click_latched = True
                    continue
                if middle_pinch_distance >= PINCH_THRESHOLD:
                    right_click_latched = False

                if pinch_distance < PINCH_THRESHOLD:
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
    finally:
        if drag_active:
            try:
                pyautogui.mouseUp()
            except Exception:
                pass
        cap.release()
        logger.info("Virtual mouse camera released.")


def start_virtual_mouse():
    """Starts the virtual mouse in a background daemon thread."""
    global _thread

    from ui.status_manager import status_manager

    if _thread and _thread.is_alive():
        status_manager.set_virtual_mouse_active(True)
        status_manager.set_virtual_mouse_error("")
        logger.info("Virtual mouse already running.")
        return True, "Virtual mouse is already enabled."

    # If the gesture controller is holding the camera, release it first
    try:
        from gesture_controller import stop_gesture_daemon, _gesture_thread
        if _gesture_thread and _gesture_thread.is_alive():
            stop_gesture_daemon()
            time.sleep(1.0)
            logger.info("Paused gesture daemon to free camera for virtual mouse.")
    except Exception:
        logger.debug("Could not check/stop gesture daemon.", exc_info=True)

    # Clear previous state
    _stop_event.clear()
    # Drain any stale messages from a previous run
    while not _status_queue.empty():
        try:
            _status_queue.get_nowait()
        except queue.Empty:
            break

    _thread = threading.Thread(target=_hand_tracking_worker, daemon=True, name="virtual-mouse")
    _thread.start()

    # Wait for the camera to initialize (up to 10s to account for retries)
    deadline = time.time() + 10.0
    while time.time() < deadline:
        if _thread is not None and not _thread.is_alive():
            break
        try:
            update = _status_queue.get(timeout=0.1)
            if update.get("status") == "ready":
                status_manager.set_virtual_mouse_active(True)
                status_manager.set_virtual_mouse_error("")
                logger.info("Virtual mouse started successfully.")
                return True, (
                    "Virtual mouse enabled. Move your index finger to steer, pinch index and thumb to click, "
                    "hold the pinch to drag, and use two fingers to scroll."
                )
            if update.get("status") == "error":
                stop_virtual_mouse()
                message = update.get("message", "Virtual mouse failed to start.")
                status_manager.set_virtual_mouse_error(message)
                return False, message
        except queue.Empty:
            continue

    # Timeout — no ready signal received
    if _thread is not None and _thread.is_alive():
        logger.warning("Virtual mouse thread alive but no ready signal after 10s — stopping.")
        stop_virtual_mouse()

    stop_virtual_mouse()
    message = "Virtual mouse failed to start — camera could not be initialized."
    status_manager.set_virtual_mouse_error(message)
    return False, message


def stop_virtual_mouse():
    """Signals the virtual mouse thread to stop and waits for cleanup."""
    global _thread

    from ui.status_manager import status_manager

    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5)
    _thread = None
    status_manager.set_virtual_mouse_active(False)
    status_manager.set_virtual_mouse_error("")
    logger.info("Virtual mouse stopped.")
