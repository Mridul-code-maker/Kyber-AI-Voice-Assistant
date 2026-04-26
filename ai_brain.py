"""
ai_brain.py — Local AI logic for Kira using Ollama (Gemma 2).

Kira now uses a 100% local, API-key-free implementation backed by Ollama.
This module handles communication with the local Ollama API.
"""

import logging
import json
import re
import requests
from config import CONVERSATION_CONTEXT_TURNS, OLLAMA_NUM_GPU
from db import db_engine

logger = logging.getLogger(__name__)


def _clean_for_tts(text):
    """
    Removes markdown symbols and extra whitespace to prepare text for TTS engines.
    """
    return text.replace("*", "").replace("#", "").strip()


def _pop_speakable_segments(buffer, force=False):
    """
    Splits a streaming buffer into complete sentences (segments) ready for immediate speech.
    Returns (list of segments, remaining buffer).
    """
    if force:
        segment = buffer.strip()
        return ([segment] if segment else []), ""

    # Split by sentence-ending punctuation followed by whitespace
    parts = re.split(r"(?<=[.!?])\s+", buffer)
    if len(parts) <= 1:
        return [], buffer
    complete = [part.strip() for part in parts[:-1] if part.strip()]
    return complete, parts[-1]


def _build_ollama_messages(prompt, current_user):
    """
    Constructs the message payload for Ollama, including system instructions and context history.
    """
    system_instructions = (
        "You are Kira, a helpful, concise AI voice assistant. "
        "Keep your responses short so they are easy to read aloud by a Text-to-Speech engine. "
        "Do not use markdown formatting like asterisks or hash symbols. Don't use emojis also. Be conversational."
    )
    messages = [{"role": "system", "content": system_instructions}]
    
    # Inject conversational history from the database
    for user_input, ai_response in db_engine.list_context_messages(current_user, CONVERSATION_CONTEXT_TURNS):
        if user_input:
            messages.append({"role": "user", "content": user_input})
        if ai_response:
            messages.append({"role": "assistant", "content": ai_response})
            
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    return messages


def query_local_ollama(prompt, current_user="guest"):
    """
    Sends a non-streaming request to the local Ollama API.
    """
    url = "http://localhost:11434/api/chat"
    messages = _build_ollama_messages(prompt, current_user)
    
    payload = {
        "model": "gemma2:latest",
        "messages": messages,
        "stream": False,
        "options": {
            "num_gpu": OLLAMA_NUM_GPU
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=45)
        response.raise_for_status()
        text = response.json()["message"]["content"]
        
        return _clean_for_tts(text)
    except requests.exceptions.ConnectionError:
        logger.error("Ollama connection refused. Is Ollama running?")
        return None
    except Exception:
        logger.error("Local Ollama request failed.", exc_info=True)
        return None


def query_local_ollama_stream(prompt, current_user="guest", on_chunk=None):
    """
    Sends a streaming request to Ollama and invokes on_chunk for every complete sentence.
    """
    url = "http://localhost:11434/api/chat"
    messages = _build_ollama_messages(prompt, current_user)

    payload = {
        "model": "gemma2:latest",
        "messages": messages,
        "stream": True,
        "options": {
            "num_gpu": OLLAMA_NUM_GPU
        }
    }

    full_text = []
    speak_buffer = ""

    try:
        with requests.post(url, json=payload, timeout=45, stream=True) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    event = json.loads(raw_line)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed Ollama stream line: %s", raw_line[:120])
                    continue

                token = event.get("message", {}).get("content", "")
                if token:
                    full_text.append(token)
                    if on_chunk:
                        speak_buffer += token
                        # Try to extract complete sentences to speak while still generating
                        segments, speak_buffer = _pop_speakable_segments(speak_buffer)
                        for segment in segments:
                            clean_segment = _clean_for_tts(segment)
                            if clean_segment:
                                on_chunk(clean_segment)

                if event.get("done"):
                    break

        # Flush any remaining text in the buffer
        if on_chunk:
            segments, speak_buffer = _pop_speakable_segments(speak_buffer, force=True)
            for segment in segments:
                clean_segment = _clean_for_tts(segment)
                if clean_segment:
                    on_chunk(clean_segment)

        return _clean_for_tts("".join(full_text))
    except requests.exceptions.ConnectionError:
        logger.error("Ollama connection refused. Is Ollama running?")
        return None
    except Exception:
        logger.error("Local Ollama streaming request failed.", exc_info=True)
        return None


def generate_response(prompt, current_user="guest", preferred_brain="gemma2", on_chunk=None):
    """
    Unified entry point for AI response generation. 
    Routes requests to the local Gemma 2 model via Ollama.
    """
    logger.info("Generating response via local Gemma 2 model.")
    if on_chunk:
        result = query_local_ollama_stream(prompt, current_user, on_chunk=on_chunk)
    else:
        result = query_local_ollama(prompt, current_user)
    
    if result:
        return result
        
    # Fallback message if Ollama service is unavailable or overloaded
    return "My local neural network is having trouble starting up. This usually happens if your computer is low on RAM. Please try closing other apps or restarting Ollama."

