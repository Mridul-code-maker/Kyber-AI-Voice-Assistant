import time
import speech_engine
import ai_brain
import local_tools
import db_engine

WAKE_WORD = "computer" # You can change this to Jarvis, Nova, etc.

def respond(text):
    "Speak and log the response."
    speech_engine.speak(text)

def main():
    print("="*40)
    print(f"Starting Assistant...")
    print(f"Waiting for wake word: '{WAKE_WORD}'")
    print("="*40)
    
    # Initialize the database automatically!
    db_engine.init_db()
    
    while True:
        # 1. Listen for voice
        command = speech_engine.listen()
        
        if not command:
            continue
            
        # 2. Check Wake Word 
        if WAKE_WORD not in command:
            # Did not hear the wake word, ignore this audio
            continue
            
        # Remove wake word from command to get the actual instruction
        command = command.replace(WAKE_WORD, "").strip()
        if not command:
            respond("Yes? I am listening.")
            continue
            
        # 3. Check for Local Hardcoded Commands First
        if "open" in command and ("notepad" in command or "calculator" in command or "browser" in command):
            if "notepad" in command:
                response = local_tools.open_application("notepad")
            elif "calculator" in command:
                response = local_tools.open_application("calculator")
            elif "browser" in command:
                response = local_tools.open_application("browser")
            
            respond(response)
            db_engine.log_conversation(command, response)
            continue
            
        if "system status" in command or "how is the computer" in command:
            response = local_tools.get_system_status()
            respond(response)
            db_engine.log_conversation(command, response)
            continue
            
        if "remember" in command:
            fact = command.split("remember")[1].strip()
            # Simple parsing: remember that [key] is [value]
            if " is " in fact:
                key, value = fact.split(" is ", 1)
                key = key.replace("that", "").strip()
                db_engine.save_memory(key, value.strip())
                response = f"Got it. I will remember that {key} is {value.strip()}."
                respond(response)
                db_engine.log_conversation(command, response)
                continue
        
        # Check memory
        if "what is my" in command or "what is the" in command:
             # Very basic retrieval logic
             for word in command.split():
                 mem = db_engine.get_memory(word)
                 if mem:
                     respond(f"You told me that {word} is {mem}.")
                     db_engine.log_conversation(command, mem)
                     break
        
        # 4. If nothing else, pass it to the AI Brain
        print("Thinking...")
        ai_response = ai_brain.generate_response(command)
        respond(ai_response)
        db_engine.log_conversation(command, ai_response)
        
        # Give it a second before listening again
        time.sleep(1)

if __name__ == "__main__":
    main()
