# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for CTXAgent: one onedir bundle, two executables.

    CTXAgent.exe     windowed: the desktop app (no console window)
    CTXAgentCLI.exe  console: --server/--browser, workers (--run-module) and the
                     MCP server over stdio. PowerShell waits only for console
                     programs, so delegated agents and jobs need this one.

Build through packaging/build.ps1 (it prepares build/tiktoken_cache first):

    .venv\\Scripts\\python -m PyInstaller --noconfirm --distpath dist --workpath build\\pyinstaller packaging\\ctxagent.spec

Linux (packaging/build_linux.sh): one executable `CTXAgent`; `CTXAgentCLI` is a
symlink to it (Linux has no console/GUI split, the name only keeps workers
recognisable). The window is Qt WebEngine (PyQt6); the desktop's own libraries
(graphics stack, D-Bus, fontconfig, NSS, libstdc++) are left to the system and
unused Qt parts are pruned, see the Linux block after Analysis.
"""
import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

WINDOWS = sys.platform == "win32"
if WINDOWS:
    from PyInstaller.utils.win32 import versioninfo as vi

ROOT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(ROOT))
from ctxagent.version import APP_NAME, PUBLISHER, __version__  # noqa: E402
from ctxagent.paths import CLI_EXE_STEM  # noqa: E402

TIKTOKEN_CACHE = ROOT / "build" / "tiktoken_cache"
if not any(TIKTOKEN_CACHE.glob("*")):
    raise SystemExit(f"{TIKTOKEN_CACHE} is empty: run packaging/prepare_tiktoken.py first")
ICON = ROOT / "packaging" / "ctxagent.ico"

# A venv made from an Anaconda Python: _sqlite3/_ctypes/_lzma link against
# sqlite3.dll/ffi.dll/liblzma.dll in <base>\Library\bin, which PyInstaller
# only finds on PATH. (No-op for a python.org interpreter.)
_conda_bin = Path(sys.base_prefix) / "Library" / "bin"
if _conda_bin.is_dir():
    os.environ["PATH"] = str(_conda_bin) + os.pathsep + os.environ.get("PATH", "")

hiddenimports = (
    collect_submodules("ctxagent")          # workers run by name (--run-module)
    + collect_submodules("uvicorn")         # protocols/loops are imported by string
    + collect_submodules("anthropic")       # imported inside functions, parts only when a request runs
    + ["tiktoken_ext", "tiktoken_ext.openai_public"]
)

excludes = [
    # research tooling and tests are not part of the product
    "tests", "bench", "paper", "tools_scripts",
    # optional / development-only heavyweights
    "playwright", "tkinter", "_tkinter", "IPython", "jupyter", "notebook", "matplotlib",
    "pytest", "sphinx", "PyQt5", "PyQt6", "PySide2", "PySide6", "gi", "cefpython3",
    "pandas.tests", "numpy.tests",
    # pywin32's MFC GUI layer (pulled in by its hooks, unused)
    "pythonwin", "win32ui", "win32uiole", "dde",
]
datas = [
    (str(ROOT / "web"), "web"),
    (str(TIKTOKEN_CACHE), "tiktoken_cache"),
]
if not WINDOWS:
    excludes.remove("PyQt6")  # the Linux window: pywebview's Qt backend
    datas.append((str(ROOT / "packaging" / "ctxagent.png"), "."))  # window icon

a = Analysis(
    [str(ROOT / "packaging" / "ctxagent_main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

if not WINDOWS:
    import re
    import subprocess

    # Libraries of the desktop session and the graphics stack come from the
    # system: they must match the running X11/Wayland session, Mesa drivers,
    # D-Bus and fontconfig setup, and a copy from an older build host breaks
    # them (an old libstdc++ breaks Mesa's LLVM drivers). NSS loads its crypto
    # modules from its own folder.
    SYSTEM_LIB = re.compile(
        r"lib(nss3|nssutil3|smime3|ssl3|nspr4|plc4|plds4|softokn3|freebl\w*|nssckbi"
        r"|stdc\+\+|gcc_s|gbm|drm|EGL\w*|GL\w*|OpenGL|vulkan|wayland-[\w-]+|fontconfig|freetype"
        r"|dbus-1|glib-2\.0|gio-2\.0|gobject-2\.0|gmodule-2\.0|gthread-2\.0|systemd|udev|asound)\.so[.\d]*$")
    # Qt parts the window does not use: QML (the view is a widget), the GTK3 theme
    # (it would pull in the build host's whole GTK), printing, positioning,
    # network-information and evdev plugins, platforms other than X11 and Wayland.
    QT = ("PyQt6", "Qt6")
    UNUSED_QT = ("qml", "plugins/position", "plugins/printsupport", "plugins/networkinformation",
                 "plugins/generic", "plugins/egldeviceintegrations", "plugins/platformthemes/libqgtk3.so")
    PLATFORMS = ("libqxcb.so", "libqwayland.so")
    LOCALES = ("en-US.pak", "en-GB.pak", "de.pak")  # Qt WebEngine UI strings

    def _wanted(entry):
        path = Path(entry[0])
        if SYSTEM_LIB.match(path.name):
            return False
        if path.parts[:2] == QT:
            inner = "/".join(path.parts[2:])
            if inner.startswith(UNUSED_QT):
                return False
            if inner.startswith("plugins/platforms/") and path.name not in PLATFORMS:
                return False
            if inner.startswith("translations/qtwebengine_locales/") and path.name not in LOCALES:
                return False
        return True

    def _is_dependency(entry):
        """A shared library collected only because something links it."""
        path = Path(entry[0])
        return (re.match(r"lib.+\.so[.\d]*$", path.name) and not path.name.startswith("libpython")
                and (len(path.parts) == 1 or path.parts[:3] == QT + ("lib",)))

    def _needed(source):
        dump = subprocess.run(["objdump", "-p", source], capture_output=True, text=True).stdout
        return re.findall(r"^\s*NEEDED\s+(\S+)", dump, re.M)

    # Keep what is wanted, then only the libraries it actually links (DT_NEEDED,
    # transitively): what the dropped parts pulled in goes with them.
    binaries = [b for b in a.binaries if _wanted(b)]
    by_name = {}
    for b in binaries:
        by_name.setdefault(Path(b[0]).name, []).append(b)
    keep = [b for b in binaries if not _is_dependency(b)]
    seen = {b[0] for b in keep}
    queue = list(keep)
    while queue:
        for name in _needed(queue.pop()[1]):
            for lib in by_name.get(name, []):
                if lib[0] not in seen:
                    seen.add(lib[0])
                    keep.append(lib)
                    queue.append(lib)
    dropped = len(a.binaries) - len(keep)
    a.binaries = [b for b in a.binaries if b[0] in seen]
    a.datas = [d for d in a.datas if _wanted(d)]
    print(f"Linux bundle: {len(a.binaries)} binaries kept, {dropped} left to the system or unused")

pyz = PYZ(a.pure)

numbers = tuple(int(part) for part in __version__.split(".")[:3]) + (0,)


def version_resource(description, filename):
    return vi.VSVersionInfo(
        ffi=vi.FixedFileInfo(filevers=numbers, prodvers=numbers, mask=0x3F, flags=0x0,
                             OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
        kids=[
            vi.StringFileInfo([vi.StringTable("040904B0", [
                vi.StringStruct("CompanyName", PUBLISHER),
                vi.StringStruct("FileDescription", description),
                vi.StringStruct("FileVersion", __version__),
                vi.StringStruct("InternalName", filename.rsplit(".", 1)[0]),
                vi.StringStruct("LegalCopyright", f"(c) {PUBLISHER}"),
                vi.StringStruct("OriginalFilename", filename),
                vi.StringStruct("ProductName", APP_NAME),
                vi.StringStruct("ProductVersion", __version__),
            ])]),
            vi.VarFileInfo([vi.VarStruct("Translation", [1033, 1200])]),
        ],
    )


if WINDOWS:
    gui = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=APP_NAME,
        console=False,
        icon=str(ICON),
        version=version_resource(APP_NAME, f"{APP_NAME}.exe"),
        upx=False,
    )
    cli = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name=CLI_EXE_STEM,
        console=True,
        icon=str(ICON),
        version=version_resource(f"{APP_NAME} (console)", f"{CLI_EXE_STEM}.exe"),
        upx=False,
    )
    coll = COLLECT(gui, cli, a.binaries, a.datas, name=APP_NAME, upx=False)
else:
    app = EXE(pyz, a.scripts, [], exclude_binaries=True, name=APP_NAME, console=True, upx=False)
    coll = COLLECT(app, a.binaries, a.datas, name=APP_NAME, upx=False)
    link = Path(DISTPATH) / APP_NAME / CLI_EXE_STEM
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(APP_NAME)
