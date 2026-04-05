import os
import subprocess
import webbrowser
import psutil
import platform

def get_system_status():
    "Returns a string about the current system status."
    cpu_usage = psutil.cpu_percent(interval=1)
    ram = psutil.virtual_memory()
    ram_usage = ram.percent
    system_os = platform.system()
    
    status = f"System is running {system_os}. CPU usage is at {cpu_usage} percent, and RAM usage is at {ram_usage} percent."
    return status

def open_website(url):
    "Opens a website in the default browser."
    if not url.startswith("http"):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opening {url}"

def open_application(app_name):
    "Attempts to open a common Windows application."
    app_name = app_name.lower()
    
    apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "cmd": "cmd.exe",
        "browser": "start msedge",
        "explorer": "explorer.exe"
    }
    
    if app_name in apps:
        try:
            # Using subprocess to open without blocking the python script
            subprocess.Popen(apps[app_name], shell=True)
            return f"Opened {app_name}."
        except Exception as e:
            return f"Failed to open {app_name}. Error: {e}"
    else:
        return f"I don't know how to open {app_name} yet."

if __name__ == "__main__":
    print(get_system_status())
