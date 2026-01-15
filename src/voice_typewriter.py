# Voice Typewriter (PySide6 UI) - modern GUI wrapper preserving the working backend behavior.
# Adds back the "v10-era" settings to the PySide6 UI:
# - Microphone selection (stores actual sounddevice device index)
# - Output mode (paste/type)
# - Model size
# - Always-on-top
# - Click-to-paste + timeout
# - Auto-capitalize
# - Punctuation commands
# - Auto-start with Windows
# - Start in widget (tile) mode
# - Hotkey (default F8)
# - Optional hold-to-record (press key = start, release = stop)
# - Minimize-to-tray behavior

from __future__ import annotations

import os
import sys
import time
import json
import re
import queue
import threading
import tempfile
import traceback
import logging
import logging.handlers
import subprocess
from pathlib import Path
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────
# Global UI style (readability)
# ──────────────────────────────────────────────────────────────
def apply_readable_style(app: "QtWidgets.QApplication") -> None:
    """Force readable dark text on light UI across Full + Widget modes."""
    try:
        app.setStyle("Fusion")
        palette = QtGui.QPalette()

        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("#f6f7fb"))
        palette.setColor(QtGui.QPalette.Base, QtGui.QColor("#ffffff"))
        palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor("#f2f3f8"))

        palette.setColor(QtGui.QPalette.Text, QtGui.QColor("#1f2937"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("#111827"))
        palette.setColor(QtGui.QPalette.ButtonText, QtGui.QColor("#111827"))
        palette.setColor(QtGui.QPalette.ToolTipText, QtGui.QColor("#111827"))

        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("#4c9aff"))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))

        app.setPalette(palette)

        # Extra safety: ensure common widgets inherit a readable foreground.
        # (Some themes / drivers can produce low-contrast default text.)
        app.setStyleSheet(app.styleSheet() + """

            * { color: #111827; }

            QToolTip { color: #111827; background: #ffffff; border: 1px solid #d0d3dd; }

            QLabel { color: #111827; }

            QCheckBox { color: #111827; }

            QGroupBox { color: #111827; }

            QTabBar::tab { color: #111827; }

            QMenuBar { color: #111827; }

            QMenu { color: #111827; }

            QStatusBar { color: #111827; }

        """)
    except Exception:
        # Never allow styling to break startup.
        pass

# ──────────────────────────────────────────────────────────────
# Guarded imports with clear dependency messages
# ──────────────────────────────────────────────────────────────
_sd_import_error = None
_np_import_error = None
_sf_import_error = None
_pyperclip_import_error = None
_keyboard_import_error = None
_whisper_import_error = None
_qt_import_error = None

try:
    import sounddevice as sd
except Exception as e:  # noqa: BLE001
    sd = None
    _sd_import_error = e

try:
    import numpy as np
except Exception as e:  # noqa: BLE001
    np = None
    _np_import_error = e

try:
    import soundfile as sf
except Exception as e:  # noqa: BLE001
    sf = None
    _sf_import_error = e

try:
    import pyperclip
except Exception as e:  # noqa: BLE001
    pyperclip = None
    _pyperclip_import_error = e

try:
    import keyboard
except Exception as e:  # noqa: BLE001
    keyboard = None
    _keyboard_import_error = e

try:
    from faster_whisper import WhisperModel
except Exception as e:  # noqa: BLE001
    WhisperModel = None
    _whisper_import_error = e

try:
    from PySide6 import QtCore, QtGui, QtWidgets
except Exception as e:  # noqa: BLE001
    QtCore = QtGui = QtWidgets = None
    _qt_import_error = e

# Mouse click-to-paste listener
try:
    from pynput import mouse as pynput_mouse
except Exception:
    pynput_mouse = None

# Windows focus control (pywin32)
try:
    import win32gui
    import win32con
    import win32api
    import win32process
except Exception:
    win32gui = win32con = win32api = win32process = None


APP_TITLE = "Voice Typewriter"
APP_NAME = "VoiceTypewriter"
DEFAULT_HOTKEY = "f8"
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCKSIZE = 1024
MODEL_SIZE = "small"
MAX_RECORDING_SECONDS = 60
AUDIO_QUEUE_MAX_CHUNKS = int((MAX_RECORDING_SECONDS * SAMPLE_RATE) / BLOCKSIZE) + 64
DEFAULT_CLICK_PASTE_TIMEOUT = 30


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def is_windows() -> bool:
    return sys.platform.startswith("win")


def get_app_dir(app_name: str = APP_NAME) -> Path:
    appdata = os.getenv("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    d = Path(appdata) / app_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(app_name: str = APP_NAME) -> Path:
    app_dir = get_app_dir(app_name)
    log_path = app_dir / "voice_typewriter.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return log_path

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return log_path


LOG_PATH = setup_logging()


def gentle_beep(kind: str) -> None:
    """Gentle audible cues using Windows system beep. No-op on non-Windows."""
    if not is_windows():
        return
    try:
        import winsound  # type: ignore

        if kind == "start":
            winsound.Beep(1200, 60)
        elif kind == "stop":
            winsound.Beep(700, 60)
        elif kind == "success":
            winsound.Beep(900, 40)
    except Exception:
        pass


def _module_error_hint(mod: str, err: Exception | None) -> str:
    if err is None:
        return ""
    return f"\n- {mod}: {type(err).__name__}: {err}"


def ensure_dependencies_or_exit() -> None:
    missing = []
    if QtWidgets is None:
        missing.append("PySide6")
    if sd is None:
        missing.append("sounddevice")
    if np is None:
        missing.append("numpy")
    if sf is None:
        missing.append("soundfile")
    if pyperclip is None:
        missing.append("pyperclip")
    if keyboard is None:
        missing.append("keyboard")
    if WhisperModel is None:
        missing.append("faster-whisper")
    if pynput_mouse is None:
        missing.append("pynput")
    if is_windows() and (win32gui is None):
        missing.append("pywin32")

    if not missing:
        return

    details = (
        _module_error_hint("PySide6", _qt_import_error)
        + _module_error_hint("sounddevice", _sd_import_error)
        + _module_error_hint("numpy", _np_import_error)
        + _module_error_hint("soundfile", _sf_import_error)
        + _module_error_hint("pyperclip", _pyperclip_import_error)
        + _module_error_hint("keyboard", _keyboard_import_error)
        + _module_error_hint("faster-whisper", _whisper_import_error)
    )

    msg = (
        "Voice Typewriter (PySide6) is missing required dependencies.\n\n"
        "Install them in your venv, then re-run:\n\n"
        "  pip install -U PySide6 sounddevice soundfile numpy pyperclip keyboard pynput pywin32 faster-whisper\n\n"
        f"Missing: {', '.join(missing)}\n"
        f"Import details:{details if details else ' (no details)'}\n"
    )
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def get_foreground_window() -> int:
    if not is_windows() or win32gui is None:
        return 0
    try:
        return int(win32gui.GetForegroundWindow())
    except Exception:
        return 0


def get_window_title(hwnd: int) -> str:
    if not is_windows() or win32gui is None:
        return ""
    try:
        return win32gui.GetWindowText(hwnd) or ""
    except Exception:
        return ""


def bring_window_to_front(hwnd: int) -> bool:
    if not is_windows() or win32gui is None or win32con is None or win32api is None or win32process is None:
        return False
    try:
        if not win32gui.IsWindow(hwnd):
            return False

        # Restore only if minimized; do NOT change maximized/normal state.
        if win32gui.IsIconic(hwnd):
            try:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            except Exception:
                pass

        try:
            fg = win32gui.GetForegroundWindow()
        except Exception:
            fg = 0

        try:
            target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
            fg_tid, _ = win32process.GetWindowThreadProcessId(fg) if fg else (0, 0)
            cur_tid = win32api.GetCurrentThreadId()

            if fg_tid:
                win32process.AttachThreadInput(cur_tid, fg_tid, True)
            win32process.AttachThreadInput(cur_tid, target_tid, True)

            try:
                win32gui.SetForegroundWindow(hwnd)
                win32gui.BringWindowToTop(hwnd)
                win32gui.SetActiveWindow(hwnd)
            finally:
                win32process.AttachThreadInput(cur_tid, target_tid, False)
                if fg_tid:
                    win32process.AttachThreadInput(cur_tid, fg_tid, False)
        except Exception:
            try:
                win32gui.SetForegroundWindow(hwnd)
            except Exception:
                pass

        return True
    except Exception:
        logging.debug("bring_window_to_front failed:\n" + traceback.format_exc())
        return False


def _send_ctrl_v() -> None:
    """Send Ctrl+V using Win32 when available; fall back to 'keyboard'."""
    if is_windows() and win32api is not None and win32con is not None:
        try:
            win32api.keybd_event(win32con.VK_CONTROL, 0, 0, 0)
            win32api.keybd_event(ord("V"), 0, 0, 0)
            time.sleep(0.01)
            win32api.keybd_event(ord("V"), 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
            return
        except Exception:
            pass

    if keyboard is None:
        raise RuntimeError("keyboard dependency missing for Ctrl+V")
    keyboard.press_and_release("ctrl+v")


def safe_clipboard_paste(text: str, *, retries: int = 3, restore_clipboard: bool = True) -> bool:
    """Paste via clipboard + Ctrl+V with retries."""
    if pyperclip is None:
        logging.error("Paste failed: missing 'pyperclip'.")
        return False

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            old = None
            if restore_clipboard:
                try:
                    old = pyperclip.paste()
                except Exception:
                    old = None

            pyperclip.copy(text)
            time.sleep(0.08)
            _send_ctrl_v()

            if restore_clipboard and old is not None:
                try:
                    time.sleep(0.03)
                    pyperclip.copy(old)
                except Exception:
                    pass

            return True
        except Exception as e:
            last_err = e
            logging.warning(f"Paste attempt {attempt}/{retries} failed: {e}")
            time.sleep(0.08 * attempt)

    logging.error(f"Paste failed after {retries} attempts: {last_err}")
    return False


def safe_direct_type(text: str, *, retries: int = 3) -> bool:
    """Type characters directly with retries."""
    if keyboard is None:
        logging.error("Direct typing failed: missing 'keyboard'.")
        return False

    CHUNK = 200
    delay = 0.02
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            for i in range(0, len(text), CHUNK):
                keyboard.write(text[i : i + CHUNK], delay=delay)
                time.sleep(0.01)
            return True
        except Exception as e:
            last_err = e
            logging.warning(f"Direct type attempt {attempt}/{retries} failed: {e}")
            time.sleep(0.08 * attempt)

    logging.error(f"Direct typing failed after {retries} attempts: {last_err}")
    return False


def auto_capitalize(text: str) -> str:
    if not text.strip():
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if s[0].islower():
            s = s[0].upper() + s[1:]
        if s[-1] not in ".!?":
            s += "."
        out.append(s)
    result = " ".join(out)
    result = re.sub(r"\s+([.!?])", r"\1", result)
    return re.sub(r"\s+", " ", result).strip()


def apply_punctuation_commands(text: str) -> str:
    commands = {
        r"\bperiod\b": ".",
        r"\bcomma\b": ",",
        r"\bquestion\s*mark\b": "?",
        r"\bexclamation\s*(mark|point)\b": "!",
        r"\bnew\s*line\b": "\n",
        r"\bnew\s*paragraph\b": "\n\n",
        r"\bcolon\b": ":",
        r"\bsemicolon\b": ";",
        r"\bquote\b": '"',
        r"\bclose\s*quote\b": '"',
    }
    for pat, rep in commands.items():
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()


# ──────────────────────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────────────────────
@dataclass
class AppSettings:
    output_mode: str = "paste"             # "paste" or "type"
    model_size: str = MODEL_SIZE
    input_device_index: int | None = None  # sounddevice device index (actual)
    always_on_top: bool = False
    click_to_paste: bool = False
    click_paste_timeout: int = DEFAULT_CLICK_PASTE_TIMEOUT
    auto_capitalize: bool = True
    punctuation_commands: bool = True
    autostart_with_windows: bool = False
    hotkey: str = DEFAULT_HOTKEY
    hold_to_record: bool = False
    start_in_widget_mode: bool = False
    
    widget_x: int | None = None
    widget_y: int | None = None

    @staticmethod
    def path() -> Path:
        return get_app_dir() / "settings.json"

    @classmethod
    def load(cls) -> "AppSettings":
        p = cls.path()
        if p.exists():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                s = cls()
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
                # normalize
                if isinstance(s.input_device_index, str) and s.input_device_index.strip().isdigit():
                    s.input_device_index = int(s.input_device_index.strip())
                if s.input_device_index in ("default", "", -1):
                    s.input_device_index = None
                if isinstance(s.click_paste_timeout, str) and str(s.click_paste_timeout).isdigit():
                    s.click_paste_timeout = int(s.click_paste_timeout)
                return s
            except Exception:
                return cls()
        return cls()

    def save(self) -> None:
        self.path().write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# Auto-start with Windows (HKCU Run)
# ──────────────────────────────────────────────────────────────
def set_windows_autostart(app_name: str, enabled: bool, command: str) -> tuple[bool, str]:
    if not is_windows():
        return False, "Auto-start is only available on Windows."
    try:
        import winreg  # type: ignore

        run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_SET_VALUE) as k:
            if enabled:
                winreg.SetValueEx(k, app_name, 0, winreg.REG_SZ, command)
            else:
                try:
                    winreg.DeleteValue(k, app_name)
                except FileNotFoundError:
                    pass
        return True, "OK"
    except Exception as e:
        return False, str(e)


def current_autostart_enabled(app_name: str) -> bool:
    if not is_windows():
        return False
    try:
        import winreg  # type: ignore

        run_key = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, run_key, 0, winreg.KEY_READ) as k:
            winreg.QueryValueEx(k, app_name)
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────
# Core engine (UI-agnostic)
# ──────────────────────────────────────────────────────────────
class VoiceTypewriterCore(QtCore.QObject):
    statusChanged = QtCore.Signal(str)
    durationChanged = QtCore.Signal(str)
    levelChanged = QtCore.Signal(float)   # 0..1
    transcriptionReady = QtCore.Signal(str)
    injected = QtCore.Signal(bool)

    def __init__(self, settings: AppSettings):
        super().__init__()
        self.settings = settings

        self._recording = False
        self._record_start_time = 0.0
        self._max_duration_reached = False

        self._audio_q: queue.Queue | None = None
        self._audio_drop_count = 0
        self._audio_lock = threading.Lock()
        self._latest_audio_chunk = None

        self.stream = None
        self.whisper_model = None

        # click-to-paste
        self._waiting_for_click = False
        self._mouse_listener = None
        self._click_timeout_thread = None

        # hotkey state
        self._hold_pressing = False

        self._load_model()

        self._ui_timer = QtCore.QTimer(self)
        self._ui_timer.setInterval(250)
        self._ui_timer.timeout.connect(self._tick_ui)
        self._ui_timer.start()

    # ── model ──────────────────────────────────────────────
    def _load_model(self) -> None:
        if WhisperModel is None:
            self.whisper_model = None
            self.statusChanged.emit("Missing faster-whisper")
            return
        try:
            self.whisper_model = WhisperModel(self.settings.model_size, device="cpu", compute_type="int8")
            logging.info(f"Loaded model: {self.settings.model_size}")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            self.whisper_model = None
            self.statusChanged.emit(f"Model error: {e}")

    def set_model_size(self, size: str) -> None:
        size = (size or "").strip()
        if size and size != self.settings.model_size:
            self.settings.model_size = size
            self.settings.save()
            self._load_model()

    # ── audio ──────────────────────────────────────────────
    @QtCore.Slot()
    def toggle_record(self) -> None:
        if self._recording:
            self.stop_recording()
        else:
            self.start_recording()

    @QtCore.Slot()
    def start_recording(self) -> None:
        if self._recording:
            return
        if sd is None:
            self.statusChanged.emit("Audio error: sounddevice missing")
            return

        self._recording = True
        self._max_duration_reached = False
        self._audio_q = queue.Queue(maxsize=AUDIO_QUEUE_MAX_CHUNKS)
        self._audio_drop_count = 0

        self._record_start_time = time.time()
        self.statusChanged.emit(f"Recording... (max {MAX_RECORDING_SECONDS}s)")
        gentle_beep("start")

        device = self.settings.input_device_index
        if device in ("default", "", -1):
            device = None
        try:
            if isinstance(device, str) and device.strip().isdigit():
                device = int(device.strip())
        except Exception:
            device = None

        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                device=device,
                blocksize=BLOCKSIZE,
                callback=self._audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self._recording = False
            self.statusChanged.emit("Audio error (see log)")
            logging.error(f"Failed to start audio stream (device={device}): {e}")

    @QtCore.Slot()
    def stop_recording(self) -> None:
        if not self._recording:
            return
        self._recording = False
        gentle_beep("stop")

        if self._max_duration_reached:
            self.statusChanged.emit(f"Max {MAX_RECORDING_SECONDS}s reached — processing...")
        else:
            self.statusChanged.emit("Stopping / processing...")

        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
        except Exception as e:
            logging.error(f"Error stopping stream: {e}")

        threading.Thread(target=self._finalize_transcription, daemon=True).start()

    def _audio_callback(self, indata, frames, time_info, status):
        chunk = indata.copy()
        with self._audio_lock:
            self._latest_audio_chunk = chunk[:, 0] if chunk.ndim == 2 and chunk.shape[1] >= 1 else chunk

        q = self._audio_q
        if q is not None:
            try:
                q.put_nowait(chunk)
            except queue.Full:
                try:
                    q.get_nowait()
                except Exception:
                    pass
                try:
                    q.put_nowait(chunk)
                except Exception:
                    pass
                self._audio_drop_count += 1
                if self._audio_drop_count % 50 == 0:
                    logging.warning(f"Audio queue full; dropped {self._audio_drop_count} chunks so far")

        try:
            if self._record_start_time:
                elapsed = time.time() - self._record_start_time
                if elapsed >= MAX_RECORDING_SECONDS and not self._max_duration_reached:
                    self._max_duration_reached = True
                    QtCore.QMetaObject.invokeMethod(self, "stop_recording", QtCore.Qt.QueuedConnection)
        except Exception:
            pass

    def _tick_ui(self) -> None:
        if self._recording:
            elapsed = max(0.0, time.time() - self._record_start_time)
            m, s = divmod(int(elapsed), 60)
            self.durationChanged.emit(f"{m:02d}:{s:02d}")
        else:
            self.durationChanged.emit("")

        if self._latest_audio_chunk is not None and np is not None:
            try:
                arr = np.array(self._latest_audio_chunk).astype(np.float32)
                if len(arr) > 0:
                    rms = float(np.sqrt(np.mean(arr**2)))
                    self.levelChanged.emit(min(1.0, rms / 0.1))
                else:
                    self.levelChanged.emit(0.0)
            except Exception:
                self.levelChanged.emit(0.0)
        else:
            self.levelChanged.emit(0.0)

    # ── transcription ──────────────────────────────────────────────
    def _finalize_transcription(self) -> None:
        try:
            q = self._audio_q
            frames_list = []
            if q is not None:
                while True:
                    try:
                        frames_list.append(q.get_nowait())
                    except queue.Empty:
                        break

            if not frames_list:
                self.statusChanged.emit("Ready (no audio captured)")
                return

            if np is None or sf is None:
                raise RuntimeError("Missing numpy/soundfile; cannot finalize audio.")

            audio_data = np.concatenate(frames_list, axis=0)
            with tempfile.TemporaryDirectory(prefix="voice_typewriter_") as td:
                wav_path = os.path.join(td, "capture.wav")
                sf.write(wav_path, audio_data, SAMPLE_RATE)
                text = self.transcribe_audio_file(wav_path).strip()

            if self.settings.punctuation_commands:
                text = apply_punctuation_commands(text)
            if self.settings.auto_capitalize:
                text = auto_capitalize(text)

            self.transcriptionReady.emit(text if text else "[No speech detected]")
            self.statusChanged.emit("Ready")

            if text:
                if self.settings.click_to_paste:
                    self._begin_click_to_paste(text)
                else:
                    time.sleep(0.12)
                    ok = self.send_text_to_target(text)
                    self.injected.emit(ok)
                    if ok:
                        gentle_beep("success")
                        self.statusChanged.emit("Injected text")
                    else:
                        self.statusChanged.emit("Injection failed")
        except Exception:
            logging.error("Transcription failed:\n" + traceback.format_exc())
            self.statusChanged.emit("Ready (error — check log)")
            self.transcriptionReady.emit("[Error — check log]")

    def transcribe_audio_file(self, wav_path: str) -> str:
        if self.whisper_model is None:
            return ""
        segments, info = self.whisper_model.transcribe(wav_path, vad_filter=True)
        return " ".join(seg.text for seg in segments)

    # ── injection ──────────────────────────────────────────────
    def send_text_to_target(self, text: str) -> bool:
        hwnd = get_foreground_window()
        title = get_window_title(hwnd).lower()

        if hwnd == 0:
            logging.error("No target window found.")
            return False

        bring_window_to_front(hwnd)
        time.sleep(0.12)

        is_browser = any(b in title for b in ["vivaldi", "chrome", "edge", "firefox", "opera", "brave"])
        mode = "paste" if is_browser else self.settings.output_mode

        if mode == "paste":
            ok = safe_clipboard_paste(text, retries=3, restore_clipboard=True)
            if not ok:
                ok = safe_direct_type(text, retries=3)
        else:
            ok = safe_direct_type(text, retries=3)
            if not ok:
                ok = safe_clipboard_paste(text, retries=3, restore_clipboard=True)

        return ok

    # ── click-to-paste ─────────────────────────────────────────
    def _begin_click_to_paste(self, text: str) -> None:
        if pynput_mouse is None:
            self.statusChanged.emit("Click-to-paste unavailable (missing pynput)")
            return

        timeout_sec = int(self.settings.click_paste_timeout or DEFAULT_CLICK_PASTE_TIMEOUT)
        self._waiting_for_click = True
        self.statusChanged.emit(f"Click to paste... {timeout_sec}s left (right-click cancels)")

        def on_click(x, y, button, pressed):
            if not self._waiting_for_click:
                return False

            if button == pynput_mouse.Button.left and pressed:
                try:
                    time.sleep(0.12)
                    ok = safe_clipboard_paste(text, retries=3, restore_clipboard=False)
                    if not ok:
                        ok = safe_direct_type(text, retries=2)

                    self.injected.emit(ok)
                    if ok:
                        gentle_beep("success")
                        self.statusChanged.emit("Pasted successfully")
                    else:
                        self.statusChanged.emit("Paste failed")
                except Exception as e:
                    logging.error(f"Click-paste failed: {e}")
                    self.statusChanged.emit("Paste failed")

                self._waiting_for_click = False
                return False

            if button == pynput_mouse.Button.right and pressed:
                self.statusChanged.emit("Paste canceled")
                self._waiting_for_click = False
                return False

            return True

        self._mouse_listener = pynput_mouse.Listener(on_click=on_click)
        self._mouse_listener.start()

        self._click_timeout_thread = threading.Thread(
            target=self._click_timeout_monitor, args=(timeout_sec,), daemon=True
        )
        self._click_timeout_thread.start()

    def _click_timeout_monitor(self, timeout_sec: int) -> None:
        start = time.time()
        while self._waiting_for_click:
            remaining = max(0, timeout_sec - int(time.time() - start))
            self.statusChanged.emit(f"Click to paste... {remaining}s left (right-click cancels)")
            if (time.time() - start) >= timeout_sec:
                self._waiting_for_click = False
                self.statusChanged.emit("Paste timed out")
                break
            time.sleep(0.5)

    # ── hotkeys ────────────────────────────────────────────────
    def register_hotkey(self) -> None:
        if keyboard is None:
            self.statusChanged.emit("Hotkey disabled (missing keyboard)")
            return

        self.unregister_hotkey()
        hk = (self.settings.hotkey or DEFAULT_HOTKEY).strip().lower()

        try:
            if self.settings.hold_to_record:
                def _on_press(e):
                    if self._hold_pressing:
                        return
                    self._hold_pressing = True
                    QtCore.QMetaObject.invokeMethod(self, "start_recording", QtCore.Qt.QueuedConnection)

                def _on_release(e):
                    if not self._hold_pressing:
                        return
                    self._hold_pressing = False
                    QtCore.QMetaObject.invokeMethod(self, "stop_recording", QtCore.Qt.QueuedConnection)

                keyboard.on_press_key(hk, _on_press, suppress=False)
                keyboard.on_release_key(hk, _on_release, suppress=False)
            else:
                keyboard.add_hotkey(
                    hk,
                    lambda: QtCore.QMetaObject.invokeMethod(self, "toggle_record", QtCore.Qt.QueuedConnection),
                )

            logging.info(f"Registered hotkey: {hk} (hold={self.settings.hold_to_record})")
            self.statusChanged.emit(f"Ready (hotkey: {hk.upper()})")
        except Exception as e:
            logging.error(f"Failed to register hotkey: {e}")
            self.statusChanged.emit("Hotkey registration failed (see log)")

    def unregister_hotkey(self) -> None:
        if keyboard is None:
            return
        try:
            keyboard.unhook_all_hotkeys()
            keyboard.unhook_all()
        except Exception:
            pass
        self._hold_pressing = False

    def shutdown(self) -> None:
        self._waiting_for_click = False
        try:
            if self._mouse_listener:
                self._mouse_listener.stop()
        except Exception:
            pass
        self.unregister_hotkey()


# ──────────────────────────────────────────────────────────────
# UI - Tile (widget) mode
# ──────────────────────────────────────────────────────────────
class TileWidget(QtWidgets.QWidget):
    showFullRequested = QtCore.Signal()

    def __init__(self, core: VoiceTypewriterCore, settings: AppSettings):
        super().__init__()
        self.core = core
        self.settings = settings

        self.setWindowTitle("Voice Typewriter — Widget")
        # Frameless, floating widget
        self.setWindowFlags(
            QtCore.Qt.Tool
            | QtCore.Qt.FramelessWindowHint
            | QtCore.Qt.WindowStaysOnTopHint
        )
        self.setFixedSize(310, 65)
        # Prevent any "ghosting" / overdraw issues on some Windows GPU drivers:
        # ensure the widget repaints its background every update.
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)


        # Resolve icon paths relative to this script
        base = Path(__file__).resolve().parent
        self._icon_stopped_path = str(base / "stopped50.png")
        self._icon_recording_path = str(base / "recording50.png")

        self._icon_stopped = QtGui.QIcon(self._icon_stopped_path)
        self._icon_recording = QtGui.QIcon(self._icon_recording_path)

        # Theme
        self.setObjectName("VTWidget")
        self.setStyleSheet("""
            QWidget#VTWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ffffff, stop:1 #eef4ff);
                border-radius: 16px;
                border: 1px solid #cfd6e6;
            }
            QLabel#VTTitle {
                color: #0f172a;
                font-weight: 800;
                font-size: 13px;
            }
            QLabel#VTStatus {
                color: #111827;
                font-size: 11px;
                background: transparent;
            }
            QToolButton#VTIconBtn {
                background: transparent;
                border: none;
            }
            QToolButton#VTIconBtn:hover {
                background: rgba(76,154,255,0.12);
                border-radius: 12px;
            }
            QPushButton#VTFullBtn {
                background: rgba(255,255,255,0.92);
                color: #111827;
                border: 1px solid #cfd6e6;
                border-radius: 10px;
                padding: 4px 10px;
                font-weight: 700;
            }
            QPushButton#VTFullBtn:hover { background: #f2f5fb; }
        """)

        # Soft shadow
        try:
            shadow = QtWidgets.QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(22)
            shadow.setOffset(0, 7)
            shadow.setColor(QtGui.QColor(0, 0, 0, 55))
            self.setGraphicsEffect(shadow)
        except Exception:
            pass

        self._drag_offset = None

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(12)

        # Left: icon button (Start/Stop)
        self.btn_icon = QtWidgets.QToolButton()
        self.btn_icon.setObjectName("VTIconBtn")
        self.btn_icon.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self.btn_icon.setIconSize(QtCore.QSize(50, 50))
        self.btn_icon.clicked.connect(self.core.toggle_record)
        root.addWidget(self.btn_icon, 0, QtCore.Qt.AlignVCenter)

        # Right: title + status, plus Full toggle
        right = QtWidgets.QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(4)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        self.title = QtWidgets.QLabel("Voice Typewriter")
        self.title.setObjectName("VTTitle")
        top.addWidget(self.title, 1)

        self.btn_full = QtWidgets.QPushButton("Full")
        self.btn_full.setObjectName("VTFullBtn")
        self.btn_full.setFixedHeight(26)
        self.btn_full.clicked.connect(self.showFullRequested.emit)
        top.addWidget(self.btn_full, 0, QtCore.Qt.AlignRight)

        right.addLayout(top)

        self.lbl = QtWidgets.QLabel("Ready")
        self.lbl.setObjectName("VTStatus")
        self.lbl.setWordWrap(True)
        self.lbl.setMinimumHeight(32)
        right.addWidget(self.lbl, 1)

        root.addLayout(right, 1)
        self._restore_position()

        # Wire core status updates
        core.statusChanged.connect(self._on_status)
        self._update_icon()

    def _on_status(self, s: str) -> None:
        s = (s or "").strip()
        if len(s) > 92:
            s = s[:89] + "…"
        self.lbl.setText(s if s else "Ready")
        # Force refresh (avoids rare ghosting in frameless widgets)
        self.lbl.update()
        self.update()
        self._update_icon()

    def _update_icon(self) -> None:
        # Fallback if PNGs are missing
        if self.core._recording:
            icon = self._icon_recording if not self._icon_recording.isNull() else QtGui.QIcon.fromTheme("media-playback-stop")
            self.btn_icon.setToolTip("Stop recording")
        else:
            icon = self._icon_stopped if not self._icon_stopped.isNull() else QtGui.QIcon.fromTheme("media-record")
            self.btn_icon.setToolTip("Start recording")
        self.btn_icon.setIcon(icon)

    # Drag anywhere to reposition
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_offset is not None and (event.buttons() & QtCore.Qt.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.LeftButton:
            self._drag_offset = None
            self._persist_position()
            event.accept()
        else:
            super().mouseReleaseEvent(event)
                      
    def moveEvent(self, event):
        super().moveEvent(event)
        self._persist_position()

    def _restore_position(self):
        try:
            x = self.settings.widget_x
            y = self.settings.widget_y
            if x is None or y is None:
                return

            pt = QtCore.QPoint(int(x), int(y))
            screen = QtGui.QGuiApplication.screenAt(pt)
            if screen is None:
                screen = QtGui.QGuiApplication.primaryScreen()
            if screen is None:
                return

            geom = screen.availableGeometry()
            nx = max(geom.left(), min(pt.x(), geom.right() - self.width()))
            ny = max(geom.top(),  min(pt.y(), geom.bottom() - self.height()))
            self.move(nx, ny)
        except Exception:
            pass

    def _persist_position(self):
        try:
            pos = self.pos()
            self.settings.widget_x = int(pos.x())
            self.settings.widget_y = int(pos.y())
            self.settings.save()
        except Exception:
            pass

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, controller: "AppController"):
        super().__init__()
        self.ctrl = controller
        self.core = controller.core
        self.settings = controller.settings

        self.setWindowTitle(APP_TITLE)
        self.resize(860, 610)

        tabs = QtWidgets.QTabWidget()
        self.setCentralWidget(tabs)

        # ── Main tab
        main = QtWidgets.QWidget()
        tabs.addTab(main, "Main")
        main_layout = QtWidgets.QVBoxLayout(main)
        main_layout.setContentsMargins(14, 14, 14, 14)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Status:"))
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("font-weight: 600;")
        top_row.addWidget(self.lbl_status, 1)
        self.lbl_timer = QtWidgets.QLabel("")
        self.lbl_timer.setMinimumWidth(70)
        top_row.addWidget(self.lbl_timer)
        main_layout.addLayout(top_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_toggle = QtWidgets.QPushButton(f"Start ({self.settings.hotkey.upper()})")
        self.btn_toggle.setMinimumHeight(36)
        self.btn_toggle.clicked.connect(self.core.toggle_record)
        btn_row.addWidget(self.btn_toggle)

        self.btn_widget = QtWidgets.QPushButton("Widget mode")
        self.btn_widget.setMinimumHeight(36)
        self.btn_widget.clicked.connect(self.ctrl.show_widget)
        btn_row.addWidget(self.btn_widget)

        self.btn_open_log = QtWidgets.QPushButton("Open log")
        self.btn_open_log.setMinimumHeight(36)
        self.btn_open_log.clicked.connect(self.open_log)
        btn_row.addWidget(self.btn_open_log)

        btn_row.addStretch(1)
        main_layout.addLayout(btn_row)

        level_row = QtWidgets.QHBoxLayout()
        level_row.addWidget(QtWidgets.QLabel("Mic level:"))
        self.level = QtWidgets.QProgressBar()
        self.level.setRange(0, 100)
        self.level.setTextVisible(False)
        level_row.addWidget(self.level, 1)
        main_layout.addLayout(level_row)

        self.txt_last = QtWidgets.QPlainTextEdit()
        self.txt_last.setReadOnly(True)
        self.txt_last.setPlaceholderText("Last transcription will appear here…")
        self.txt_last.setMinimumHeight(260)
        main_layout.addWidget(self.txt_last, 1)

        tip = QtWidgets.QLabel("Tip: Click into any app, place cursor, then use your hotkey or Start/Stop.")
        tip.setStyleSheet("color: #777;")
        main_layout.addWidget(tip)

        # ── Settings tab
        settings_tab = QtWidgets.QWidget()
        tabs.addTab(settings_tab, "Settings")
        form = QtWidgets.QFormLayout(settings_tab)
        form.setContentsMargins(14, 14, 14, 14)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(12)

        # Microphone
        self.cmb_mic = QtWidgets.QComboBox()
        self._mic_items = self._build_mic_list()
        for item in self._mic_items:
            self.cmb_mic.addItem(item["label"], userData=item["sd_index"])
        if self.settings.input_device_index is None:
            ix = self.cmb_mic.findData(None)
            if ix >= 0:
                self.cmb_mic.setCurrentIndex(ix)
        else:
            ix = self.cmb_mic.findData(self.settings.input_device_index)
            if ix >= 0:
                self.cmb_mic.setCurrentIndex(ix)
        form.addRow("Microphone:", self.cmb_mic)

        # Model size
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.cmb_model.setCurrentText(self.settings.model_size)
        form.addRow("Model size:", self.cmb_model)

        # Output mode
        self.cmb_output = QtWidgets.QComboBox()
        self.cmb_output.addItems(["paste", "type"])
        self.cmb_output.setCurrentText(self.settings.output_mode)
        form.addRow("Output mode:", self.cmb_output)

        # Hotkey
        self.ed_hotkey = QtWidgets.QLineEdit(self.settings.hotkey)
        self.ed_hotkey.setPlaceholderText("e.g. f8, f9, ctrl+alt+v")
        form.addRow("Hotkey:", self.ed_hotkey)

        self.chk_hold = QtWidgets.QCheckBox("Hold-to-record (press & hold hotkey; release to stop)")
        self.chk_hold.setChecked(self.settings.hold_to_record)
        form.addRow("", self.chk_hold)

        # Click-to-paste
        self.chk_click = QtWidgets.QCheckBox("Click-to-paste mode (left click to paste, right click to cancel)")
        self.chk_click.setChecked(self.settings.click_to_paste)
        form.addRow("", self.chk_click)

        self.spin_click = QtWidgets.QSpinBox()
        self.spin_click.setRange(10, 120)
        self.spin_click.setSingleStep(5)
        self.spin_click.setValue(int(self.settings.click_paste_timeout))
        form.addRow("Click-to-paste timeout (seconds):", self.spin_click)

        # Text post-processing
        self.chk_cap = QtWidgets.QCheckBox("Auto-capitalize sentences")
        self.chk_cap.setChecked(self.settings.auto_capitalize)
        form.addRow("", self.chk_cap)

        self.chk_punct = QtWidgets.QCheckBox("Punctuation commands (say 'comma', 'period', etc.)")
        self.chk_punct.setChecked(self.settings.punctuation_commands)
        form.addRow("", self.chk_punct)

        # Window options
        self.chk_top = QtWidgets.QCheckBox("Always on top")
        self.chk_top.setChecked(self.settings.always_on_top)
        form.addRow("", self.chk_top)

        self.chk_widget_start = QtWidgets.QCheckBox("Start in widget mode")
        self.chk_widget_start.setChecked(self.settings.start_in_widget_mode)
        form.addRow("", self.chk_widget_start)

        self.chk_autostart = QtWidgets.QCheckBox("Auto-start with Windows")
        self.chk_autostart.setChecked(current_autostart_enabled(APP_NAME) or bool(self.settings.autostart_with_windows))
        form.addRow("", self.chk_autostart)

        btn_save = QtWidgets.QPushButton("Save settings")
        btn_save.clicked.connect(self.save_settings)
        form.addRow("", btn_save)

        # ── About tab
        about = QtWidgets.QWidget()
        tabs.addTab(about, "About")
        about_l = QtWidgets.QVBoxLayout(about)
        about_l.setContentsMargins(18, 18, 18, 18)
        about_text = QtWidgets.QLabel(
            "<h2>Voice Typewriter</h2>"
            "<p>Fast, accurate voice-to-text anywhere in Windows.<br>"
            "Hotkey-driven recording, click-to-paste, and tray/widget modes.</p>"
            "<p><b>Privacy:</b> Transcription runs locally using <i>faster-whisper</i>.</p>"
        )
        about_text.setWordWrap(True)
        about_l.addWidget(about_text)
        about_l.addStretch(1)

        # ── Help tab
        help_tab = QtWidgets.QWidget()
        tabs.addTab(help_tab, "Help")
        help_l = QtWidgets.QVBoxLayout(help_tab)
        help_l.setContentsMargins(18, 18, 18, 18)
        help_txt = QtWidgets.QPlainTextEdit()
        help_txt.setReadOnly(True)
        help_txt.setPlainText(
            "Quick start\n"
            "1) Click into any app (Word, browser, email), place the cursor.\n"
            "2) Press the hotkey (default F8) or click Start.\n"
            "3) Speak. Press again (or release if hold-to-record) to stop.\n"
            "4) Text is pasted/typed into the focused app.\n\n"
            "Click-to-paste\n"
            "- Enable in Settings.\n"
            "- After transcription, click where you want the text.\n"
            "- Right-click cancels.\n\n"
            "If text won't paste into a specific app\n"
            "- If the target app is running as Administrator, Windows may block injection.\n"
            "- Fix: run Voice Typewriter as Administrator too, or run target app non-elevated.\n\n"
            f"Log file:\n{LOG_PATH}\n"
        )
        help_l.addWidget(help_txt, 1)

        # Menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction("Widget mode").triggered.connect(self.ctrl.show_widget)
        view_menu.addAction("Full mode").triggered.connect(self.ctrl.show_full)
        self.menuBar().addAction("Exit").triggered.connect(self.ctrl.exit_app)

        # Connect core -> UI
        self.core.statusChanged.connect(self.on_status)
        self.core.durationChanged.connect(self.lbl_timer.setText)
        self.core.levelChanged.connect(lambda v: self.level.setValue(int(max(0.0, min(1.0, v)) * 100)))
        self.core.transcriptionReady.connect(self.txt_last.setPlainText)

        # modern styling
        self._apply_style()

        # Always-on-top (initial)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, bool(self.settings.always_on_top))

    def _apply_style(self) -> None:
        QtWidgets.QApplication.setStyle("Fusion")
        self.setStyleSheet("""
            QMainWindow { background: #f6f7fb; }
            QTabWidget::pane { border: 1px solid #d9dbe2; border-radius: 10px; background: white; }
            QTabBar::tab { padding: 8px 14px; border: 1px solid #d9dbe2; border-bottom: none; border-top-left-radius: 10px; border-top-right-radius: 10px; background: #eef0f7; margin-right: 4px; }
            QTabBar::tab:selected { background: white; }
            QPushButton { padding: 7px 12px; border-radius: 10px; background: white; border: 1px solid #d0d3dd; }
            QPushButton:hover { background: #f2f3f8; }
            QProgressBar { border: 1px solid #d0d3dd; border-radius: 8px; background: white; height: 14px; }
            QProgressBar::chunk { background: #4c9aff; border-radius: 8px; }
            QPlainTextEdit { border-radius: 12px; border: 1px solid #d0d3dd; background: white; padding: 8px; }
            QLineEdit, QComboBox, QSpinBox { border-radius: 10px; border: 1px solid #d0d3dd; background: white; padding: 6px; }
        
            QLabel { color: #111827; }
            QCheckBox { color: #111827; }
            QRadioButton { color: #111827; }
            QComboBox { color: #111827; }
            QSpinBox { color: #111827; }
            QLineEdit { color: #111827; }
            QPlainTextEdit { color: #111827; }
            QTabBar::tab { color: #111827; }
""")

    def _build_mic_list(self) -> list[dict]:
        out: list[dict] = []
        if sd is None:
            return [{"label": "Default", "sd_index": None}]
        try:
            devices = sd.query_devices()
            out.append({"label": "Default", "sd_index": None})
            for idx, d in enumerate(devices):
                try:
                    if int(d.get("max_input_channels", 0)) > 0:
                        name = d.get("name", f"Device {idx}")
                        label = name if len(name) <= 60 else (name[:60] + "…")
                        out.append({"label": f"{idx}: {label}", "sd_index": idx})
                except Exception:
                    continue
        except Exception:
            return [{"label": "Default", "sd_index": None}]
        return out or [{"label": "Default", "sd_index": None}]

    def on_status(self, s: str) -> None:
        self.lbl_status.setText(s)
        self.btn_toggle.setText(f"{'Stop' if self.core._recording else 'Start'} ({self.settings.hotkey.upper()})")

    def open_log(self) -> None:
        try:
            if LOG_PATH.exists():
                os.startfile(LOG_PATH)  # type: ignore[attr-defined]
        except Exception:
            subprocess.run(["notepad.exe", str(LOG_PATH)], check=False)

    def save_settings(self) -> None:
        # collect
        self.settings.input_device_index = self.cmb_mic.currentData()
        self.settings.model_size = self.cmb_model.currentText().strip() or MODEL_SIZE
        self.settings.output_mode = self.cmb_output.currentText().strip() or "paste"
        self.settings.hotkey = (self.ed_hotkey.text().strip().lower() or DEFAULT_HOTKEY)
        self.settings.hold_to_record = self.chk_hold.isChecked()
        self.settings.click_to_paste = self.chk_click.isChecked()
        self.settings.click_paste_timeout = int(self.spin_click.value())
        self.settings.auto_capitalize = self.chk_cap.isChecked()
        self.settings.punctuation_commands = self.chk_punct.isChecked()
        self.settings.always_on_top = self.chk_top.isChecked()
        self.settings.start_in_widget_mode = self.chk_widget_start.isChecked()
        self.settings.autostart_with_windows = self.chk_autostart.isChecked()

        # persist
        self.settings.save()

        # apply always-on-top live
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint, bool(self.settings.always_on_top))
        self.show()

        # apply to core
        self.core.settings = self.settings
        self.core.set_model_size(self.settings.model_size)
        self.core.register_hotkey()

        # autostart
        cmd = f'"{sys.executable}" "{os.path.abspath(sys.argv[0])}"'
        ok, msg = set_windows_autostart(APP_NAME, self.settings.autostart_with_windows, cmd)
        if not ok:
            QtWidgets.QMessageBox.warning(self, "Auto-start", f"Failed to update auto-start:\n{msg}")

        QtWidgets.QMessageBox.information(self, "Saved", "Settings saved.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        event.ignore()
        self.hide()
        self.ctrl.show_tray_message("Voice Typewriter", "Still running in the system tray.")


# ──────────────────────────────────────────────────────────────
# App controller (tray, mode switching)
# ──────────────────────────────────────────────────────────────
class AppController(QtCore.QObject):
    def __init__(self):
        super().__init__()
        self.settings = AppSettings.load()
        self.core = VoiceTypewriterCore(self.settings)

        self.main = MainWindow(self)
        self.tile = TileWidget(self.core, self.settings)
        self.tile.showFullRequested.connect(self.show_full)

        self._setup_tray()

        self.core.register_hotkey()

        if self.settings.start_in_widget_mode:
            self.show_widget()
        else:
            self.show_full()

    def _setup_tray(self) -> None:
        # Use custom tray icon (avoid dark system theme icons)
        base = Path(__file__).resolve().parent
        icon_path = base / "tray.ico"   # or "tray.png"

        if icon_path.exists():
            icon = QtGui.QIcon(str(icon_path))
        else:
            # Safe fallback if file missing
            pm = QtGui.QPixmap(64, 64)
            pm.fill(QtGui.QColor("#4c9aff"))
            icon = QtGui.QIcon(pm)

        self.tray = QtWidgets.QSystemTrayIcon(icon)
        self.tray.setToolTip("Voice Typewriter")

        menu = QtWidgets.QMenu()
        menu.addAction("Show (Full)").triggered.connect(self.show_full)
        menu.addAction("Show (Widget)").triggered.connect(self.show_widget)
        menu.addSeparator()
        menu.addAction("Start/Stop Recording").triggered.connect(self.core.toggle_record)
        menu.addSeparator()
        menu.addAction("Exit").triggered.connect(self.exit_app)

        self.tray.setContextMenu(menu)
        self.tray.activated.connect(self._tray_activated)
        self.tray.show()

    def _tray_activated(self, reason):
        if reason == QtWidgets.QSystemTrayIcon.Trigger:
            if self.main.isVisible():
                self.main.hide()
            else:
                self.show_full()

    def show_tray_message(self, title: str, msg: str) -> None:
        try:
            self.tray.showMessage(title, msg, QtWidgets.QSystemTrayIcon.Information, 2500)
        except Exception:
            pass

    def show_full(self) -> None:
        self.tile.hide()
        self.main.show()
        self.main.raise_()
        self.main.activateWindow()

    def show_widget(self) -> None:
        self.main.hide()
        self.tile.show()
        self.tile.raise_()
        self.tile.activateWindow()

    def exit_app(self) -> None:
        self.core.shutdown()
        try:
            self.tray.hide()
        except Exception:
            pass
        QtWidgets.QApplication.quit()


def main():
    ensure_dependencies_or_exit()
    app = QtWidgets.QApplication(sys.argv)
    apply_readable_style(app)

    _ = AppController()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
