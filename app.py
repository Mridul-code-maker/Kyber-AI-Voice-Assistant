from flask import Flask, render_template, request, jsonify
import threading
import time
import speech_engine
import ai_brain
import local_tools
import db_engine

app = Flask(__name__)

WAKE_WORD = "computer"
assistant_status = f"Waiting for '{WAKE_WORD}'"
current_user_id = "admin" # Row Level Context

def respond(text):
    speech_engine.speak(text)

def assistant_loop():
    global assistant_status, current_user_id
    print("Starting background assistant loop...")
    db_engine.init_db()
    
    while True:
        command = speech_engine.listen()
        
        if not command:
            continue
            
        if WAKE_WORD not in command:
            continue
            
        assistant_status = f"Processing ({current_user_id.upper()})..."
        command = command.replace(WAKE_WORD, "").strip()
        
        if not command:
            respond("Yes? I am listening.")
            assistant_status = f"Waiting for '{WAKE_WORD}'"
            continue
            
        if "open" in command and ("notepad" in command or "calculator" in command or "browser" in command):
            if "notepad" in command:
                response = local_tools.open_application("notepad")
            elif "calculator" in command:
                response = local_tools.open_application("calculator")
            elif "browser" in command:
                response = local_tools.open_application("browser")
            
            respond(response)
            db_engine.log_conversation(current_user_id, command, response)
            assistant_status = f"Waiting for '{WAKE_WORD}'"
            continue
            
        if "system status" in command:
            response = local_tools.get_system_status()
            respond(response)
            db_engine.log_conversation(current_user_id, command, response)
            assistant_status = f"Waiting for '{WAKE_WORD}'"
            continue
            
        if "remember" in command and " is " in command:
            fact = command.split("remember")[1].strip()
            key, value = fact.split(" is ", 1)
            key = key.replace("that", "").strip()
            db_engine.save_memory(current_user_id, key, value.strip())
            response = f"Got it. I will remember that {key} is {value.strip()} for user {current_user_id}."
            respond(response)
            db_engine.log_conversation(current_user_id, command, response)
            assistant_status = f"Waiting for '{WAKE_WORD}'"
            continue
            
        if "what is my" in command or "what is the" in command:
            words = command.split()
            found = False
            for word in words:
                mem = db_engine.get_memory(current_user_id, word)
                if mem:
                    response = f"Based on your profile, {word} is {mem}."
                    respond(response)
                    db_engine.log_conversation(current_user_id, command, response)
                    found = True
                    break
            if found:
                assistant_status = f"Waiting for '{WAKE_WORD}'"
                continue
                
        assistant_status = "Thinking (using Groq)..."
        ai_response = ai_brain.generate_response(command)
        respond(ai_response)
        db_engine.log_conversation(current_user_id, command, ai_response)
        
        assistant_status = f"Waiting for '{WAKE_WORD}'"
        time.sleep(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/set_user', methods=['POST'])
def set_user():
    global current_user_id
    data = request.json
    if "user_id" in data:
        current_user_id = data["user_id"]
    return jsonify({"success": True, "current_user": current_user_id})

@app.route('/api/status', methods=['GET'])
def get_status():
    global current_user_id
    import sqlite3
    try:
        conn = sqlite3.connect("assistant_brain.db")
        cursor = conn.cursor()
        # Row level restriction!
        cursor.execute("SELECT user_input, ai_response, timestamp FROM conversations WHERE user_id = ? ORDER BY id DESC LIMIT 5", (current_user_id,))
        history = cursor.fetchall()
        conn.close()
    except:
        history = []
        
    return jsonify({
        "status": f"{assistant_status} [{current_user_id.upper()}]",
        "current_user_id": current_user_id,
        "history": [{"user": row[0], "ai": row[1], "time": row[2]} for row in history]
    })

if __name__ == '__main__':
    threading.Thread(target=assistant_loop, daemon=True).start()
    app.run(debug=False, port=5000, host="0.0.0.0")
