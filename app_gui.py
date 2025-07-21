import tkinter as tk
import threading
import subprocess

def start_detection():
    subprocess.run(["python", "realtime_detection.py"])

root = tk.Tk()
root.title("Mask Detection App")
root.geometry("400x200")

label = tk.Label(root, text="Click the button to start real-time mask detection.", font=("Arial", 12))
label.pack(pady=20)

btn = tk.Button(root, text="Start Detection", command=lambda: threading.Thread(target=start_detection).start(), font=("Arial", 12))
btn.pack(pady=10)

root.mainloop()
