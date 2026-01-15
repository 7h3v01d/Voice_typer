# VoiceTypewriter.spec
# Build: pyinstaller VoiceTypewriter.spec --clean

import os
from PyInstaller.utils.hooks import collect_submodules

script_path = os.path.abspath("voice_typewriter_pyside6_v9.py")
project_dir = os.path.dirname(script_path)

hiddenimports = []
hiddenimports += collect_submodules("pynput")
hiddenimports += collect_submodules("sounddevice")
hiddenimports += collect_submodules("soundfile")
hiddenimports += collect_submodules("faster_whisper")
hiddenimports += collect_submodules("ctranslate2")
hiddenimports += collect_submodules("tokenizers")

datas = [
    (os.path.join(project_dir, "stopped50.png"), "."),
    (os.path.join(project_dir, "recording50.png"), "."),
]

block_cipher = None

a = Analysis(
    [script_path],
    pathex=[project_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VoiceTypewriter",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app (no console window)
    icon=os.path.join(project_dir, "tray.ico"),
    version=os.path.join(project_dir, "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VoiceTypewriter",
)
