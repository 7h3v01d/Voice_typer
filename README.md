# Voice Typewriter (Windows)

**Voice Typewriter** is a fast, accurate, system-wide **voice-to-text keyboard replacement** for Windows 10/11.

It allows you to dictate text **into any application** simply by placing the cursor and speaking — no commands, no training, no cloud dependency.

Designed for real daily use.

---

## ✨ Key Features

### 🎙️ System-Wide Voice Typing
- Dictate into **any app**: browsers, editors, chat apps, terminals, office software
- Cursor-based injection (paste or type)
- No app-specific plugins required

### 🧠 High-Accuracy Transcription
- Powered by **faster-whisper**
- Local processing (no internet required)
- Voice Activity Detection (VAD) to remove silence
- Optional spoken punctuation (e.g. “comma”, “new line”)

---

## 🖥️ Two Interface Modes

### 🔹 Full Mode
- Complete control panel
- Settings, status, transcription preview
- Best for configuration and monitoring

### 🔹 Widget (Tile) Mode
- Compact always-on-screen tile
- Single **Start / Stop** button
- Live microphone level indicator
- Ideal for minimal distraction workflows

Switch instantly via the **View** menu.

---

## ⌨️ Flexible Controls

### 🎯 Hotkey Control
- Default hotkey: **F8**
- Fully user-configurable
- Button label updates automatically to match

### 🔁 Recording Modes
- **Toggle mode**: press once to start, press again to stop
- **Press-and-hold mode**: hold key to record, release to stop  
  *(Available for single-key hotkeys)*

---

## 🖱️ Injection Modes

### 📋 Normal Mode
- Automatically injects text after transcription

### 🖱️ Click-to-Paste Mode
- After transcription, click anywhere to paste
- Countdown timer with cancel support
- Extremely reliable for browsers and Electron apps

The app automatically selects the most reliable method per target app.

---

## 🔊 Gentle Audio Feedback

Subtle, non-intrusive system beeps provide confidence without distraction:

- 🎙️ Recording start
- 🛑 Recording stop
- ✅ Successful text injection

Uses Windows system sounds (no audio files).

---

## 🧳 System Integration

### 🚀 Auto-Start with Windows
- Optional setting
- Uses user registry (no admin required)
- Starts silently using `pythonw` when available

### 🪟 Minimize to System Tray
- Close = hide to tray
- Tray menu:
  - Show / Hide
  - Help / About
  - Quit

Standard taskbar minimization also supported.

---

## ℹ️ Help & About

Built-in **Help / About** window with:
- Feature overview
- Usage instructions
- Version and system info

Accessible from:
- Main UI
- System tray menu

---

## ⚙️ Requirements

- Windows 10 or 11
- Python 3.9+
- Microphone

---

## 📦 Dependencies

See `requirements.txt`. Core dependencies include:

- `faster-whisper`
- `sounddevice`, `soundfile`, `numpy`
- `keyboard`, `pyperclip`
- `pywin32`
- `pynput`
- `pystray`, `Pillow` (for system tray)

---

## ▶️ Running the App

```bash
python voice_typewriter.py
```
(Filename may vary if using a patched version)

---
## 🧪 Notes on Reliability
Designed to avoid Windows focus and UIPI issues

Uses clipboard paste where most reliable

Automatically adapts behaviour for browsers

Bounded audio buffers prevent memory issues

Safe failure paths with logging

## 🛠️ Logging
Logs are written to:

```lua
%APPDATA%\VoiceTypewriter\voice_typewriter.log
Useful for diagnostics and tuning.
```

📦 Packaging (EXE)
The app is ready to be packaged into a single executable:

```bash
Copy code
pip install pyinstaller
pyinstaller --onefile --noconsole voice_typewriter.py
```
(Icon, version metadata, and auto-start polish can be added.)

---
❤️ Philosophy
This project prioritizes:
- Reliability over gimmicks
- Local processing over cloud dependence
- Minimal friction for daily use
- It is built to feel like a natural extension of the operating system — not a novelty tool.

## 📜 License
Private / Internal use</br>
(Define license before public release)
