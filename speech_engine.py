import speech_recognition as sr
import pyttsx3

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Try to set it to a female voice if available, or just the default
if len(voices) > 1:
    engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 160) # Speed of speech

def speak(text):
    "Speaks the given text out loud."
    print(f"Assistant: {text}")
    engine.say(text)
    engine.runAndWait()

def listen():
    "Listens to the microphone and converts speech to text."
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            # Listen for a maximum of 5 seconds
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            print("Recognizing...")
            # Use Google's free speech recognition
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text.lower()
            
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("Sorry, I could not understand.")
            return ""
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            return ""
        except Exception as e:
            print(f"Microphone error: {e}")
            return ""

if __name__ == "__main__":
    speak("Hello, the speech engine is working.")
    result = listen()
    if result:
        speak(f"I heard you say: {result}")
