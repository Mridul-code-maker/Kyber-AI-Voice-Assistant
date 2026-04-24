"""
command_router.py — Routes voice commands to specific local tools or the AI brain.
Handles browser automation, system control, memory persistence, and user management.
"""

import logging
import os
import re
import urllib.parse

import ai_brain
import local_tools
import speech_engine
from auth import voice_auth
from config import ADMIN_RECORD_SECONDS, WAKE_WORD
from db import db_engine
from ui.status_manager import status_manager

logger = logging.getLogger(__name__)


def normalize_command(command):
    """
    Cleans up the command string for better pattern matching.
    Removes punctuation and normalizes whitespace.
    """
    lowered = command.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def extract_query(command, remove_words):
    """
    Strips stop-words from a command to isolate the core search query.
    """
    words = normalize_command(command).split()
    filtered = [word for word in words if word not in remove_words]
    return " ".join(filtered).strip()


def _open_url(url):
    """Utility to open a URL in the default system browser."""
    import webbrowser
    webbrowser.open(url)


def _get_first_youtube_video(query):
    """
    Scrapes YouTube search results to find the ID of the first video.
    Returns the full watch URL if found, otherwise the search results URL.
    """
    import requests
    import re
    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    try:
        response = requests.get(search_url, headers=headers, timeout=5)
        video_ids = re.findall(r"/watch\?v=([a-zA-Z0-9_-]{11})", response.text)
        if video_ids:
            # Find first unique ID
            unique_ids = []
            for vid in video_ids:
                if vid not in unique_ids:
                    unique_ids.append(vid)
            if unique_ids:
                return f"https://www.youtube.com/watch?v={unique_ids[0]}"
    except Exception as e:
        logger.error("Failed to scrape first YouTube video ID: %s", e)
    
    return search_url


def _handle_browser_commands(normalized):
    """
    Processes browser-related commands like 'open youtube', 'search for...', etc.
    """
    remove_words = [
        WAKE_WORD.lower(), "play", "open", "on", "youtube", "music", "please", 
        "can", "you", "for", "me", "search", "and", "the", "a", "an", "something", "song",
    ]
    
    # Specific YouTube integration (Auto-play first result)
    if "open youtube and play" in normalized:
        query = normalized.split("open youtube and play", 1)[1].strip()
        if not query:
            _open_url("https://www.youtube.com")
            return "Opening YouTube."
        url = _get_first_youtube_video(query)
        _open_url(url)
        return f"Playing {query} on YouTube."

    # YouTube Music handling
    if "play" in normalized and "youtube music" in normalized:
        query = extract_query(normalized, remove_words)
        if not query:
            _open_url("https://music.youtube.com")
            return "Opening YouTube Music."
        _open_url(f"https://music.youtube.com/search?q={urllib.parse.quote_plus(query)}")
        return f"Playing {query} on YouTube Music."

    # General YouTube play command
    if "play" in normalized and "youtube" in normalized:
        query = extract_query(normalized, remove_words)
        if not query:
            _open_url("https://www.youtube.com")
            return "Opening YouTube."
        url = _get_first_youtube_video(query)
        _open_url(url)
        return f"Playing {query} on YouTube."

    if "open youtube" in normalized:
        _open_url("https://www.youtube.com")
        return "Opening YouTube."

    # Search queries
    if "search youtube for " in normalized:
        query = normalized.split("search youtube for ", 1)[1].strip()
        if not query:
            return None
        _open_url(f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}")
        return f"Searching YouTube for {query}."

    if "search for " in normalized:
        query = normalized.split("search for ", 1)[1].strip()
        if not query:
            return None
        _open_url(f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}")
        return f"Searching for {query}."

    # Generic site opening with fallback to search
    if "open " in normalized:
        site_name = normalized.split("open ", 1)[1].strip()
        site_name = site_name.split(" and ")[0].strip()
        if not site_name:
            return None
        mapping = {
            "google": "https://www.google.com",
            "gmail": "https://mail.google.com",
            "github": "https://www.github.com",
            "spotify": "https://open.spotify.com",
            "netflix": "https://www.netflix.com",
            "twitter": "https://www.twitter.com",
            "x": "https://www.twitter.com",
            "reddit": "https://www.reddit.com",
            "wikipedia": "https://www.wikipedia.org",
        }
        url = mapping.get(site_name, f"https://www.google.com/search?q={urllib.parse.quote_plus(site_name)}")
        _open_url(url)
        return f"Opening {site_name}."

    return None


def _register_guest(name):
    """
    Interactive guest voice registration flow.
    Requires the guest to repeat the wake word for a biometric sample.
    """
    speech_engine.speak(f"Waiting for voice authentication. {name}, please say {WAKE_WORD} followed by a short sentence.")
    try:
        audio = speech_engine.capture_audio(timeout=10, phrase_time_limit=5)
        heard = speech_engine.audio_to_text(audio)
        if normalize_command(WAKE_WORD) not in normalize_command(heard):
            speech_engine.speak(f"I didn't hear the wake word. Registration cancelled.")
            return
    except Exception:
        speech_engine.speak(f"Registration timed out.")
        return

    speech_engine.speak("Recording your voice now. Keep speaking.")
    audio = speech_engine.capture_audio(timeout=3, phrase_time_limit=ADMIN_RECORD_SECONDS)
    temp_file = speech_engine.save_audio_to_temp_file(audio)
    try:
        voice_auth.register_speaker(name, temp_file)
        db_engine.ensure_user(name, role="guest")
    finally:
        try:
            os.remove(temp_file)
        except OSError:
            logger.error("Could not remove temporary enrollment file.", exc_info=True)


def _register_admin_from_voice():
    """
    Biometric registration for the system administrator.
    Only triggered during bootstrap or via explicit command.
    """
    speech_engine.speak(f"Waiting for admin voice authentication. Please say {WAKE_WORD} followed by a short sentence.")
    try:
        audio = speech_engine.capture_audio(timeout=10, phrase_time_limit=5)
        heard = speech_engine.audio_to_text(audio)
        if normalize_command(WAKE_WORD) not in normalize_command(heard):
            speech_engine.speak(f"I didn't hear the wake word. Admin registration cancelled.")
            return
    except Exception:
        speech_engine.speak(f"Registration timed out.")
        return

    speech_engine.speak("Recording admin voice now. Keep speaking.")
    audio = speech_engine.capture_audio(timeout=3, phrase_time_limit=ADMIN_RECORD_SECONDS)
    temp_file = speech_engine.save_audio_to_temp_file(audio)
    try:
        voice_auth.register_speaker("admin", temp_file)
        db_engine.ensure_user("admin", role="admin")
    finally:
        try:
            os.remove(temp_file)
        except OSError:
            logger.error("Could not remove temporary admin enrollment file.", exc_info=True)


def _flip_pronouns(text):
    """
    Swaps 'my' for 'your', 'I' for 'you', etc. for natural responses.
    Ensures the assistant speaks about the user correctly.
    """
    replacements = {
        "my": "your",
        "mine": "yours",
        "i": "you",
        "me": "you",
        "am": "are"
    }
    words = text.split()
    result = []
    for word in words:
        lower = word.lower()
        if lower in replacements:
            result.append(replacements[lower])
        else:
            result.append(word)
    return " ".join(result)


def _choose_brain(command: str) -> str:
    """Logic to decide which AI model/brain to use for a specific command."""
    return "gemma2"


def route_command(command, current_user, on_response_chunk=None):
    """
    Main entry point for command routing.
    Matches commands against local tools first, falling back to the AI brain.
    """
    normalized = normalize_command(command)

    # Emergency / Control commands
    if normalized in {"stop", "stop speaking", "be quiet", "quiet", "cancel speech"}:
        speech_engine.stop_speaking()
        return "Speech stopped.", False

    if normalized in {"forget everything", "clear history", "start fresh", "forget our conversation"}:
        db_engine.clear_conversation_history(current_user)
        return "I have cleared our conversation history. Starting fresh now.", False

    # -- System Features & Vision --
    try:
        if "enable virtual mouse" in normalized:
            from vision.hand_tracking import start_virtual_mouse
            success, message = start_virtual_mouse()
            return message, False
        if "disable virtual mouse" in normalized:
            from vision.hand_tracking import stop_virtual_mouse
            stop_virtual_mouse()
            return "Virtual mouse disabled.", False

        if "enable gesture control" in normalized:
            from gesture_controller import start_gesture_daemon
            start_gesture_daemon()
            return "Gesture control enabled. You can now use hand signals for volume and state control.", False

        if "disable gesture control" in normalized:
            from gesture_controller import stop_gesture_daemon
            stop_gesture_daemon()
            return "Gesture control disabled. Camera released.", False
        
        import re as _re
        _app_match = _re.match(r"^open\s+(notepad|calculator|browser)\b", normalized)
        if _app_match:
            return local_tools.open_application(_app_match.group(1)), False

        if "system status" in normalized or "how is the computer" in normalized:
            return local_tools.get_system_status(), False
    except Exception:
        logger.error("Feature command failed.", exc_info=True)
        return "Sorry, that command failed. Try again.", False

    # -- Memory Retrieval & Persistence --
    try:
        if "remember" in normalized and " is " in normalized:
            fact = normalized.split("remember", 1)[1].strip()
            key, value = fact.split(" is ", 1)
            key = key.replace("that", "").strip()
            db_engine.save_memory(current_user, key, value.strip())
            display_key = _flip_pronouns(key)
            return f"Got it. I will remember that {display_key} is {value.strip()}.", False

        if "what is my" in normalized or "what is the" in normalized:
            for prefix in ("what is my ", "what is the "):
                if prefix in normalized:
                    query = normalized.split(prefix, 1)[1].strip()
                    break
            else:
                query = ""
            if query:
                result = db_engine.search_memory(current_user, query)
                if result:
                    fact_key, fact_value = result
                    display_key = _flip_pronouns(fact_key)
                    return f"Based on your profile, {display_key} is {fact_value}.", False
    except Exception:
        logger.error("Memory command failed.", exc_info=True)
        return "Sorry, that command failed. Try again.", False

    # -- Admin / User Management --
    if "make me admin" in normalized or "set me as admin" in normalized:
        if db_engine.has_admin_user():
            return "Admin is already configured. Only one admin is allowed.", False
        _register_admin_from_voice()
        return "You are now registered as admin.", False

    if normalized.startswith("register user"):
        if current_user != "admin":
            return "Only the admin can register new users.", False
        remainder = normalized.split("register user", 1)[1].strip()
        name = remainder.split()[0] if remainder else ""
        if not name:
            return "Tell me the user name to register.", False
        _register_guest(name)
        return f"{name} has been registered as a guest.", False

    if normalized.startswith("delete memory"):
        if current_user != "admin":
            return "Only admin can delete memory.", False
        parts = normalized.split("delete memory", 1)[1].strip()
        if not parts:
            return "Tell me which memory key to delete.", False
        deleted = db_engine.delete_memory(current_user, parts)
        return (f"Deleted memory for key {parts}." if deleted else "I could not find that memory key."), False

    # -- Browser Automation Fallback --
    try:
        browser_response = _handle_browser_commands(normalized)
        if browser_response:
            return browser_response, False
    except Exception:
        logger.error("Browser command failed.", exc_info=True)
        return "Sorry, that command failed. Try again.", False

    # -- General Intelligence (AI Fallback) --
    brain = _choose_brain(command)
    status_manager.set_current_brain(brain)
    response = ai_brain.generate_response(
        command, current_user=current_user, preferred_brain=brain, on_chunk=on_response_chunk
    )
    return response, True

