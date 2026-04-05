# Nova - Database-Driven AI Voice Assistant

Nova is an advanced AI Voice Assistant featuring a stunning, vibrant Glassmorphism Web interface. It uniquely bridges Artificial Intelligence with core Database Management Systems (DBMS) engineering, including implementations of Row Level Security (RLS).

## Key Features
- **Lightning Fast AI Logic**: Powered by Groq (LLaMA-3) for instantaneous voice conversation processing.
- **Glassmorphism Web Dashboard**: A dynamic, animated frontend built securely on a Flask web server.
- **Local Voice Engine**: Uses Python's SpeechRecognition to listen, and pyttsx3 to speak back out loud.
- **Full DBMS Integration (SQLite)**: Logs all conversations, commands, and saves unique facts to the database.
- **Row Level Security (RLS) Simulation**: Strict multi-user context enforcement at the database level. Switch between Admin and Guest profiles in the UI to demonstrate how the DBMS physically walls off data access!

## Developer Setup Instructions
1. Clone this repository to your local machine.
2. Create a `.env` file in the root directory and add your Groq API key: 
   `GROQ_API_KEY=your_key_here`
3. Install all the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the Application Server:
   ```bash
   python app.py
   ```
5. Open your web browser and navigate to `http://localhost:5000`.
