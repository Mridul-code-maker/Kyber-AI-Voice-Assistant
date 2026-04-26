"""
orb_overlay.py — Siri / Wispr Flow–style frosted-glass pill overlay for Kira.

Renders a compact orb that expands into a pill shape depending on the assistant state.
Drawn entirely with tkinter on a transparent canvas — no heavy OpenCV dependency for the UI.

Window Features:
  - Always-on-top and borderless.
  - Transparent background with glassmorphism effects.
  - Click-through enabled on Windows (win32) to avoid blocking user interaction.
  - Positioned at the bottom-center of the screen.
"""

import json
import logging
import math
import platform
import time
import urllib.request
import tkinter as tk

from config import FLASK_HOST, FLASK_PORT

# Configure logging for the overlay process
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STATUS_URL = f"http://{FLASK_HOST}:{FLASK_PORT}/status"

# ── UI Design Tokens & Metrics ──────────────────────────────
PILL_HEIGHT      = 52
MAX_CANVAS_W     = 280        # Sufficient width for expanded states
MAX_CANVAS_H     = 110        # Sufficient height for labels
ORB_RADIUS       = 22         # Idle state half-size
CORNER_RADIUS    = 26         # Half of PILL_HEIGHT for perfect rounding
BOTTOM_GAP       = 20         # Vertical offset from screen bottom

# Glassmorphism Color Palette
GLASS_FILL       = "#2B313C"
GLASS_INNER      = "#394150"
GLASS_BORDER     = "#7A869B"
DOT_COLOR_WHITE  = "#F5F7FA"
DOT_COLOR_BLUE   = "#7EC8E3"
DOT_COLOR_PURPLE = "#72A7FF"
LABEL_FONT       = ("Segoe UI Semibold", 9)
STATUS_FONT      = ("Segoe UI", 9)

# Mapping of Assistant State to Pill Width
STATE_WIDTHS = {
    "idle":      52,
    "listening": 180,
    "waiting":   170,
    "thinking":  210,   # Gemma 2 specific width
    "speaking":  200,
}
THINKING_WIDTHS = {"gemma2": 210}

# Mapping of State to Accent Glow Colors
STATE_ACCENT = {
    "idle":               "#F5F7FA",
    "listening":          "#73D2F6",
    "waiting":            "#FFD166",
    "thinking_gemma2":    "#72A7FF",
    "speaking":           "#F5F7FA",
}

# Animation & Polling Timing
POLL_MS          = 50          # 20 Frames per second
SPRING_DURATION  = 0.45        # Duration of the width expansion/contraction
BREATHE_PERIOD   = 2.0         # Seconds per idle "breathing" cycle
BRAIN_LABEL_DUR  = 3.0         # Seconds to show the AI model label
GESTURE_LABEL_DUR = 2.0        # Seconds to show gesture notifications


def _spring_ease(t: float) -> float:
    """
    Approximation of a spring-like cubic-bezier(0.34, 1.56, 0.64, 1) easing function.
    Provides a subtle overshoot effect when the pill expands.
    """
    if t >= 1.0:
        return 1.0
    s = t * t
    return s * (3.0 * (1.0 - t) * 1.56 + t)


class PillOverlay:
    """
    Handles the rendering and animation of the Kira visual orb/pill.
    Synchronizes with the main assistant state via periodic HTTP polling.
    """
    def __init__(self):
        # Internal State
        self.state        = "idle"
        self.brain        = "gemma2"
        self.speaking     = False
        self.mic_unavail  = False

        # Labels & Notifications
        self.brain_label       = ""
        self.brain_label_time  = 0.0
        self.gesture_label     = ""
        self.gesture_label_time = 0.0
        self.prev_response     = ""
        self.prev_gesture      = ""

        # Animation Interpolation state
        self.start_time        = time.time()
        self.current_width     = float(STATE_WIDTHS["idle"])
        self.target_width      = self.current_width
        self.transition_start  = 0.0
        self.transition_from   = self.current_width

        # Tkinter Window Initialization
        self.root = tk.Tk()
        self.root.title("Kira Overlay")
        self.root.overrideredirect(True)         # Remove window decorations (title bar, etc.)
        self.root.attributes("-topmost", True)   # Ensure it stays above other windows
        self.root.attributes("-alpha", 0.92)     # Global window transparency

        # Handle Platform-specific Transparency
        if platform.system() == "Windows":
            # Chromakey transparency for Windows
            self.root.attributes("-transparentcolor", "#010101")
            bg = "#010101"
        else:
            # Fallback for non-Windows platforms
            try:
                self.root.attributes("-transparentcolor", "#010101")
                bg = "#010101"
            except tk.TclError:
                bg = "#1A1A1A"

        self.canvas = tk.Canvas(
            self.root,
            width=MAX_CANVAS_W,
            height=MAX_CANVAS_H,
            bg=bg,
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack()

        # Center the window horizontally at the bottom
        self._position_window()

        # Enable click-through behavior on Windows after a short delay
        self.root.after(200, self._apply_click_through)

    def _position_window(self):
        """Calculates and sets the window geometry based on current screen resolution."""
        self.root.update_idletasks()
        scr_w = self.root.winfo_screenwidth()
        scr_h = self.root.winfo_screenheight()
        x = (scr_w - MAX_CANVAS_W) // 2
        y = scr_h - MAX_CANVAS_H - BOTTOM_GAP
        self.root.geometry(f"{MAX_CANVAS_W}x{MAX_CANVAS_H}+{x}+{y}")

    def _apply_click_through(self):
        """
        Uses Win32 API to make the window transparent to mouse clicks.
        This allows users to interact with windows behind the overlay.
        """
        if platform.system() != "Windows":
            return
        try:
            import ctypes
            hwnd = ctypes.windll.user32.FindWindowW(None, "Kira Overlay")
            if not hwnd:
                logger.warning("HWND not found for click-through.")
                return
            
            # Windows Extended Styles for layering and transparency
            GWL_EXSTYLE       = -20
            WS_EX_LAYERED     = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            WS_EX_TOPMOST     = 0x00000008
            WS_EX_TOOLWINDOW  = 0x00000080
            
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            style |= WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOPMOST | WS_EX_TOOLWINDOW
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style)
            
            # Set the transparency key to match the canvas background color
            LWA_COLORKEY = 0x1
            ctypes.windll.user32.SetLayeredWindowAttributes(
                hwnd, 0x00010101, 0, LWA_COLORKEY,
            )
            logger.info("Click-through enabled via win32.")
        except Exception:
            logger.warning("Click-through setup failed.", exc_info=True)

    def _fetch_status(self):
        """Polls the main Flask application to sync state and notifications."""
        try:
            with urllib.request.urlopen(STATUS_URL, timeout=1) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            new_state    = data.get("state", "idle")
            new_brain    = data.get("current_brain", "gemma2")
            new_response = data.get("last_response", "")
            new_gesture  = data.get("last_gesture", "")
            self.speaking    = data.get("speaking", False)
            self.mic_unavail = data.get("mic_unavailable", False)

            # Trigger AI Model label if the response source changed
            if new_response and new_response != self.prev_response:
                self.prev_response = new_response
                self.brain_label = f"AI {new_brain.capitalize()}"
                self.brain_label_time = time.time()

            # Trigger Gesture notification
            if new_gesture and new_gesture != self.prev_gesture:
                self.prev_gesture = new_gesture
                gesture_names = {
                    "open_palm": "Palm Detected",
                    "index_finger": "Volume Up",
                    "peace_sign": "Switch User",
                    "fist": "Stop TTS",
                }
                self.gesture_label = gesture_names.get(new_gesture, new_gesture)
                self.gesture_label_time = time.time()

            # Update target width for the pill expansion animation
            if new_state != self.state or new_brain != self.brain:
                self.state = new_state
                self.brain = new_brain
                if self.state == "thinking":
                    tw = THINKING_WIDTHS.get(self.brain, 210)
                else:
                    tw = STATE_WIDTHS.get(self.state, 52)
                    
                if tw != self.target_width:
                    self.transition_from  = self.current_width
                    self.transition_start = time.time()
                    self.target_width     = float(tw)
            else:
                self.state = new_state
                self.brain = new_brain

        except Exception:
            pass  # Resilient to server downtime

    def _rounded_rect(self, x1, y1, x2, y2, r, **kw):
        """Draws a rounded rectangle on the canvas using polygon smoothing."""
        points = [
            x1 + r, y1,
            x2 - r, y1,
            x2, y1, x2, y1 + r,
            x2, y2 - r,
            x2, y2, x2 - r, y2,
            x1 + r, y2,
            x1, y2, x1, y2 - r,
            x1, y1 + r,
            x1, y1, x1 + r, y1,
        ]
        return self.canvas.create_polygon(points, smooth=True, **kw)

    def _accent_color(self):
        """Returns the appropriate color for the current state's visual elements."""
        if self.state == "thinking":
            key = f"thinking_{self.brain}"
        else:
            key = self.state
        return STATE_ACCENT.get(key, "#FFFFFF")

    def _draw_frame(self):
        """Main rendering pass for a single animation frame."""
        now  = time.time()
        t    = now - self.start_time
        c    = self.canvas
        c.delete("all")

        # ── Interpolate Pill Width (Spring Animation) ────────
        elapsed = now - self.transition_start
        if elapsed < SPRING_DURATION and self.target_width != self.transition_from:
            progress = min(elapsed / SPRING_DURATION, 1.0)
            eased    = _spring_ease(progress)
            self.current_width = self.transition_from + (self.target_width - self.transition_from) * eased
        else:
            self.current_width = self.target_width

        pw = self.current_width
        ph = PILL_HEIGHT
        cx = MAX_CANVAS_W / 2.0
        cy = MAX_CANVAS_H / 2.0

        pill_x1 = cx - pw / 2
        pill_y1 = cy - ph / 2
        pill_x2 = cx + pw / 2
        pill_y2 = cy + ph / 2

        accent = self._accent_color()
        glow_color = self._fade_hex(accent, 0.35)

        # ── Render Layers ────────────────────────────────────
        
        # 1. Outer dropshadow layers
        for i in range(3):
            offset = 3 - i
            alpha_hex = ["#2A2A2A", "#1E1E1E", "#141414"][i]
            self._rounded_rect(
                pill_x1 - offset, pill_y1 - offset + 2,
                pill_x2 + offset, pill_y2 + offset + 2,
                CORNER_RADIUS + offset,
                fill=alpha_hex, outline="",
            )

        # 2. Outer glow ring
        self._rounded_rect(
            pill_x1 - 2, pill_y1 - 2, pill_x2 + 2, pill_y2 + 2,
            CORNER_RADIUS + 2,
            fill="",
            outline=glow_color,
            width=2,
        )

        # 3. Main frosted glass body
        self._rounded_rect(
            pill_x1, pill_y1, pill_x2, pill_y2,
            CORNER_RADIUS,
            fill=GLASS_FILL,
            outline="",
        )

        # 4. Glass inner shimmer (stippled)
        self._rounded_rect(
            pill_x1 + 1, pill_y1 + 1, pill_x2 - 1, pill_y2 - 1,
            CORNER_RADIUS - 1,
            fill=GLASS_INNER,
            outline="",
            stipple="gray50",
        )

        # 5. Frosted edge border
        self._rounded_rect(
            pill_x1, pill_y1, pill_x2, pill_y2,
            CORNER_RADIUS,
            fill="",
            outline=GLASS_BORDER,
            width=1.5,
        )

        # 6. Top-edge reflection highlight
        self._rounded_rect(
            pill_x1 + 4, pill_y1 + 2, pill_x2 - 4, pill_y1 + ph * 0.38,
            CORNER_RADIUS - 4,
            fill="#8892A3",
            outline="",
            stipple="gray25",
        )

        # ── Dynamic Visual States ────────────────────────────
        dot_base_r = 6
        
        if self.state == "idle" and not self.speaking:
            # Idle "Breathing" animation
            breath = 0.85 + 0.30 * (0.5 + 0.5 * math.sin(2 * math.pi * t / BREATHE_PERIOD))
            dot_r = dot_base_r * breath
            c.create_oval(
                cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                fill=DOT_COLOR_WHITE, outline="",
            )
            # Soft pulsing halo
            halo_r = dot_r + 4 + math.sin(t * 1.5) * 2
            c.create_oval(
                cx - halo_r, cy - halo_r, cx + halo_r, cy + halo_r,
                fill="", outline="#BCC6D5", width=1, stipple="gray25",
            )

        elif self.state == "listening":
            # Active listening waveform
            self._draw_waveform(c, cx - 35, cy, t, accent, bar_count=5)
            self._draw_pill_text(c, cx - 10, cy, "Listening", accent)

        elif self.state == "waiting":
            # Subtle waiting pulse
            self._draw_waveform(c, cx - 35, cy, t, accent, bar_count=4)
            self._draw_pill_text(c, cx - 10, cy, "Waiting", accent)

        elif self.state == "thinking" and not self.speaking:
            # AI Processing visualization
            c.create_oval(cx - dot_base_r, cy - dot_base_r, cx + dot_base_r, cy + dot_base_r,
                          fill=accent, outline="")
            if self.brain == "gemma2":
                # Geometric orbit for Gemma 2
                self._draw_orbiting_particles(c, cx, cy, t, accent, count=3)
                self._draw_pill_text(c, cx, cy, "Processing", accent)
            else:
                # Fallback pulse for other brains
                pulse_r = 8 + 6 * abs(math.sin(t * 2 * math.pi / 0.5))
                c.create_oval(
                    cx - pulse_r, cy - pulse_r, cx + pulse_r, cy + pulse_r,
                    fill="", outline=accent, width=2,
                )
                self._draw_pill_text(c, cx, cy, "Processing", accent)

        elif self.state == "speaking" or self.speaking:
            # Speech synthesis waveform
            self._draw_waveform(c, cx - 30, cy, t, DOT_COLOR_WHITE, bar_count=5)
            self._draw_pill_text(c, cx - 5, cy, "Speaking", "#D7DDE7")

        # ── Notifications & Overlays ─────────────────────────
        
        # Bottom Label (AI Model Information)
        if self.brain_label and (now - self.brain_label_time) < BRAIN_LABEL_DUR:
            fade = max(0.0, 1.0 - (now - self.brain_label_time) / BRAIN_LABEL_DUR)
            label_color = self._fade_hex("#5B9BD5", fade) if "Gemma" in self.brain_label else self._fade_hex("#B0B0B0", fade)
            c.create_text(
                cx, pill_y2 + 14,
                text=self.brain_label,
                fill=label_color,
                font=LABEL_FONT,
                anchor="center",
            )

        # Top Label (Hand Gesture Events)
        if self.gesture_label and (now - self.gesture_label_time) < GESTURE_LABEL_DUR:
            fade = max(0.0, 1.0 - (now - self.gesture_label_time) / GESTURE_LABEL_DUR)
            label_color = self._fade_hex("#FFD700", fade)
            c.create_text(
                cx, pill_y1 - 12,
                text=self.gesture_label,
                fill=label_color,
                font=LABEL_FONT,
                anchor="center",
            )

    def _draw_waveform(self, c, cx, cy, t, color, bar_count=5):
        """Draws animated vertical bars representing voice activity."""
        bar_w   = 3
        bar_gap = 6
        total_w = bar_count * bar_w + (bar_count - 1) * bar_gap
        start_x = cx + 14
        for i in range(bar_count):
            phase   = t * 5.0 + i * 1.2
            bar_h   = 6 + 12 * abs(math.sin(phase))
            bx      = start_x + i * (bar_w + bar_gap) - total_w / 2
            by_top  = cy - bar_h / 2
            by_bot  = cy + bar_h / 2
            c.create_rectangle(bx, by_top, bx + bar_w, by_bot, fill=color, outline="")

    def _draw_orbiting_particles(self, c, cx, cy, t, color, count=3):
        """Draws small particles orbiting the central dot for thinking state."""
        orbit_r = 16
        period  = 2.0
        for i in range(count):
            angle = (2 * math.pi * t / period) + i * (2 * math.pi / count)
            px = cx + orbit_r * math.cos(angle)
            py = cy + orbit_r * math.sin(angle)
            pr = 3
            c.create_oval(px - pr, py - pr, px + pr, py + pr, fill=color, outline="")

    def _draw_pill_text(self, c, cx, cy, text, color):
        """Renders status text inside the expanded pill."""
        c.create_text(
            cx + 46, cy,
            text=text,
            fill=color,
            font=STATUS_FONT,
            anchor="w",
        )

    @staticmethod
    def _fade_hex(hex_color: str, fade: float) -> str:
        """Utility to calculate a faded color (interpolating toward black)."""
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        r = int(r * fade)
        g = int(g * fade)
        b = int(b * fade)
        return f"#{r:02x}{g:02x}{b:02x}"

    def run(self):
        """Starts the Tkinter main loop and scheduling of ticks."""
        self._poll_counter = 0

        def _tick():
            self._poll_counter += 1
            # Poll server state at a lower frequency than drawing
            if self._poll_counter % 8 == 0:
                self._fetch_status()
            self._draw_frame()
            self.root.after(POLL_MS, _tick)

        self.root.after(POLL_MS, _tick)
        logger.info("Pill overlay started (tkinter).")
        self.root.mainloop()


if __name__ == "__main__":
    PillOverlay().run()
