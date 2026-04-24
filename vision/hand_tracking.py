import math
import multiprocessing
import queue
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

_process = None
_stop_event = None
_status_queue = None


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


def _safe_queue_put(status_queue, payload):
    if status_queue is None:
        return
    try:
        status_queue.put_nowait(payload)
    except Exception:
        pass


def run_hand_tracking(stop_event: multiprocessing.Event, status_queue: multiprocessing.Queue):
    cap = cv2.VideoCapture(VIRTUAL_MOUSE_CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(VIRTUAL_MOUSE_CAMERA_INDEX)

    if not cap.isOpened():
        _safe_queue_put(status_queue, {"status": "error", "message": "Could not open the webcam for virtual mouse control."})
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIRTUAL_MOUSE_FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIRTUAL_MOUSE_FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    screen_w, screen_h = pyautogui.size()
    prev_x = screen_w / 2
    prev_y = screen_h / 2
    pinch_active = False
    right_click_latched = False
    drag_active = False
    pinch_started_at = 0.0
    scroll_anchor_y = None
    last_seen = time.time()

    _safe_queue_put(status_queue, {"status": "ready"})

    try:
        with mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.65,
            model_complexity=0,
        ) as hands:
            while not stop_event.is_set():
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
        logger.error("Virtual mouse frame handling failed.", exc_info=True)
        _safe_queue_put(status_queue, {"status": "error", "message": "Virtual mouse crashed while processing camera frames."})
    finally:
        if drag_active:
            try:
                pyautogui.mouseUp()
            except Exception:
                logger.debug("Failed to release mouse during virtual mouse shutdown.", exc_info=True)
        cap.release()


def _drain_status_queue():
    updates = []
    if _status_queue is None:
        return updates
    while True:
        try:
            updates.append(_status_queue.get_nowait())
        except queue.Empty:
            return updates


def start_virtual_mouse():
    global _process, _stop_event, _status_queue

    from ui.status_manager import status_manager

    if _process and _process.is_alive():
        status_manager.set_virtual_mouse_active(True)
        status_manager.set_virtual_mouse_error("")
        logger.info("Virtual mouse already running.")
        return True, "Virtual mouse is already enabled."

    _stop_event = multiprocessing.Event()
    _status_queue = multiprocessing.Queue()
    _process = multiprocessing.Process(
        target=run_hand_tracking,
        args=(_stop_event, _status_queue),
        daemon=True,
        name="virtual-mouse",
    )
    _process.start()

    deadline = time.time() + 3.0
    while time.time() < deadline:
        if _process is not None and not _process.is_alive():
            break
        for update in _drain_status_queue():
            if update.get("status") == "ready":
                status_manager.set_virtual_mouse_active(True)
                status_manager.set_virtual_mouse_error("")
                logger.info("Virtual mouse started.")
                return True, (
                    "Virtual mouse enabled. Move your index finger to steer, pinch index and thumb to click, "
                    "hold the pinch to drag, and use two fingers to scroll."
                )
            if update.get("status") == "error":
                stop_virtual_mouse()
                message = update.get("message", "Virtual mouse failed to start.")
                status_manager.set_virtual_mouse_error(message)
                return False, message
        time.sleep(0.05)

    if _process is not None and _process.is_alive():
        status_manager.set_virtual_mouse_active(True)
        status_manager.set_virtual_mouse_error("")
        logger.info("Virtual mouse started without ready signal.")
        return True, (
            "Virtual mouse enabled. Move your index finger to steer, pinch index and thumb to click, "
            "hold the pinch to drag, and use two fingers to scroll."
        )

    stop_virtual_mouse()
    message = "Virtual mouse failed to start."
    status_manager.set_virtual_mouse_error(message)
    return False, message


def stop_virtual_mouse():
    global _process, _stop_event, _status_queue

    from ui.status_manager import status_manager

    if _stop_event is not None:
        _stop_event.set()
    if _process is not None:
        _process.join(timeout=3)
        if _process.is_alive():
            _process.kill()
            _process.join(timeout=2)
    _process = None
    _stop_event = None
    _status_queue = None
    status_manager.set_virtual_mouse_active(False)
    status_manager.set_virtual_mouse_error("")
    logger.info("Virtual mouse stopped.")
