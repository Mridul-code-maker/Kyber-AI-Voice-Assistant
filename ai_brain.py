import os
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

def has_real_key():
    return API_KEY and API_KEY != "your_api_key_here"

def generate_local_response(prompt):
    "Fallback local chat logic when no API key is present."
    prompt = prompt.lower()
    if "hello" in prompt or "hi" in prompt or "hey" in prompt:
        return "Hello! I am operating in local mode."
    elif "who are you" in prompt:
        return "I am Nova, your local AI assistant."
    elif "joke" in prompt:
        return "Why do programmers prefer dark mode? Because light attracts bugs!"
    elif "time" in prompt:
        import datetime
        return f"The current time is {datetime.datetime.now().strftime('%H:%M')}."
    else:
        responses = [
            "I'm currently in local mode, so my knowledge is limited.",
            "I am waiting for connection to my neural network."
        ]
        return random.choice(responses)

def generate_response(prompt):
    "Processes a prompt through the Groq AI model or falls back to local."
    if not has_real_key():
        return generate_local_response(prompt)
    
    try:
        from groq import Groq
        if not hasattr(generate_response, "client"):
            generate_response.client = Groq(api_key=API_KEY)
            
        system_instructions = (
            "You are a helpful, concise AI voice assistant. "
            "Keep your responses short so they are easy to read aloud by a Text-to-Speech engine. "
            "Do not use markdown formatting like asterisks or hash symbols."
        )
        
        chat_completion = generate_response.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_instructions,
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama3-8b-8192",
        )
        
        response_text = chat_completion.choices[0].message.content
        clean_text = response_text.replace('*', '').replace('#', '').strip()
        return clean_text
    
    except Exception as e:
        return f"I ran into an error connecting to my brain. Error: {str(e)}"
