import os
import pickle
import logging

import numpy as np

from config import EMBEDDINGS_PATH, SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except Exception:  # pragma: no cover
    VoiceEncoder = None
    preprocess_wav = None

encoder = VoiceEncoder() if VoiceEncoder else None


def get_embedding(audio_path: str) -> np.ndarray:
    if not encoder or not preprocess_wav:
        raise RuntimeError("Voice authentication is unavailable. Install resemblyzer.")
    wav = preprocess_wav(audio_path)
    return encoder.embed_utterance(wav)


def load_embeddings() -> dict:
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as file:
            return pickle.load(file)
    return {}


def save_embeddings(embeddings: dict):
    with open(EMBEDDINGS_PATH, "wb") as file:
        pickle.dump(embeddings, file)


def identify_speaker(audio_path: str) -> str:
    "Returns admin, a guest username, or unknown."
    if not audio_path or not os.path.exists(audio_path):
        return "unknown"

    embeddings = load_embeddings()
    if not embeddings:
        return "unknown"
    candidate = get_embedding(audio_path)
    best_match, best_score = None, 0.0
    for name, emb in embeddings.items():
        score = float(np.dot(candidate, emb))
        if score > best_score:
            best_score, best_match = score, name
    logger.info("Voice match for %s: score=%.3f (threshold=%.2f)", best_match, best_score, SIMILARITY_THRESHOLD)
    return best_match if best_score > SIMILARITY_THRESHOLD else "unknown"


def register_speaker(name: str, audio_path: str):
    embeddings = load_embeddings()
    embeddings[name] = get_embedding(audio_path)
    save_embeddings(embeddings)
    logger.info("Registered speaker embedding for %s.", name)


def is_admin_registered() -> bool:
    return "admin" in load_embeddings()
