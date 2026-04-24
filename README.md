# Kyber AI Assistant

Kyber is a high-performance, local-first Windows voice assistant. It features voice biometric authentication, persistent long-term memory, an Ollama-backed AI brain, and camera-driven gesture/virtual mouse controls.

## ✨ Key Features
- **Local AI (Ollama)**: 100% private conversations using the `gemma2` model.
- **Offline Speech-to-Text**: Powered by `faster-whisper` for near-instant transcription.
- **Voice Biometrics**: Identifies speakers (Admin vs. Guests) using speaker embeddings.
- **Persistent Memory**: Stores conversational context and user-specific facts in a local SQLite database.
- **Glassmorphism Overlay**: A sleek, always-on-top `tkinter` visualizer for assistant states.
- **Vision Controls**: Hand gesture shortcuts and a virtual mouse powered by MediaPipe.

## 📂 Project Structure
- `app.py`: Main entry point (Flask & WebSocket server).
- `ai_brain.py`: Local Ollama integration and stream processing.
- `speech_engine.py`: Handles TTS and STT (Whisper & pyttsx3).
- `gesture_controller.py`: Background camera processing for hand signs.
- `orb_overlay.py`: Visual state representation (frosted-glass pill).
- `core/`: Assistant logic and command routing.
- `db/`: Database schema and persistent storage logic.
- `auth/`: Voice authentication and user registration.

## 🚀 Developer Setup

### 1. Prerequisites
- **Python 3.10+**
- **Ollama**: [Download Ollama](https://ollama.com/) and run `ollama pull gemma2`.
- **FFmpeg**: Required for audio processing.

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/Mridul-code-maker/Kyber-AI-Voice-Assistant
cd kyber-ai-assistant

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory to customize settings (optional):
```env
STT_ENGINE=faster-whisper
WHISPER_MODEL_SIZE=base.en
```

### 4. Running Kyber
```bash
python app.py
```
- The **Orb Overlay** will appear at the bottom of your screen.
- Access the **Web Dashboard** at `http://localhost:5000`.

## 🛠️ Commands
- "Kyber, open YouTube and play [video]"
- "Kyber, remember that my favorite color is blue"
- "Kyber, what is my favorite color?"
- "Kyber, enable virtual mouse"
- "Kyber, enable gesture control"

