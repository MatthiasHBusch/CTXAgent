# CTXAgent desktop app

**Formerly PiAgent.** The app was renamed to match the publication (CTXAgent). The
installer replaces a PiAgent install: it installs to
`%LOCALAPPDATA%\Programs\CTXAgent` and removes the old program folder and
shortcuts. On its first start the app moves the old data folder
`%LOCALAPPDATA%\PiAgent` (chats, settings, logs) to `%LOCALAPPDATA%\CTXAgent`;
if the old app is still running from it, the app keeps using it and moves it at
a later start. API keys saved under the old name (Credential Manager service
`PiAgent`) are read from there and copied to `CTXAgent`. Profile exports of the
old app (`format: piagent-profiles`) import as before. `PIAGENT_KEYSTORE` and
`PIAGENT_NO_WEBVIEW` (Linux: `PIAGENT_WEBVIEW_GPU`, `PIAGENT_PREFIX`) are still
read when the `CTXAGENT_` names are not set. On Linux `install.sh` replaces a
PiAgent install and the app moves `~/.local/share/piagent` the same way.

CTXAgent is the desktop shell around `ctxagent`: the same FastAPI server and web
UI, started in-process on `127.0.0.1` and shown in a native window (pywebview
on Edge WebView2; on Linux Qt WebEngine, see [Linux](#linux)). The product name
lives in `ctxagent/version.py` (`APP_NAME`, `__version__`); the Python package
stays `ctxagent`.

## Run from source

```powershell
pip install -r requirements.txt pywebview     # pywebview only for the window
python -m ctxagent.desktop                     # native window
.\run.cmd --desktop                            # the same through the launcher
python -m ctxagent.desktop --browser           # system browser instead of a window
python -m ctxagent                             # server only on :8765 (development, unchanged)
.\run.cmd                                      # server on :8765 + browser (unchanged)
```

Options (`python -m ctxagent.desktop --help`):

| option | meaning |
| --- | --- |
| `--browser` | serve on a free port and open the system browser |
| `--server` | server only, on `--port` or `CTXAGENT_PORT` (default 8765); what `python -m ctxagent` runs |
| `--port N` | fixed port; the window/browser modes otherwise reuse the last port (stable page origin) or pick a free one |
| `--data DIR` | data folder, same as `CTXAGENT_DATA` |
| `--import-data DIR` | copy an existing data folder in before starting (see below) |
| `--replace-data` | with `--import-data`: back up and replace a data folder that already holds chats |
| `--debug` | WebView2 developer tools and verbose logs |

The server binds only to `127.0.0.1`. Without pywebview, or when the Edge
WebView2 runtime is missing, the app falls back to the system browser. With no
console attached (the installed `CTXAgent.exe`) a small message box then stays
open: *Yes* opens the page again, *No* stops CTXAgent.

**One instance per data folder.** The first instance holds an OS lock on
`<data>/instance.lock` and writes `<data>/instance.json` (port, pid, a random
token). Launching again brings the running window to the front (or reopens the
browser page) instead of starting a second server on the same chats. A server
started with `python -m ctxagent` holds the same lock; a second launch then
opens a window on that server. `--server` refuses to start next to a running
instance.

**Closing the window stops CTXAgent.** If a chat turn, a background job or a
delegated agent is still running, CTXAgent asks first. Running turns get five
seconds to wind down, jobs and delegated agents are cancelled (the same
cancellation the UI uses), the server stops and leftover worker or shell
processes are ended; nothing keeps running or billing in the background.
`python -m ctxagent` (server mode) keeps the development behaviour: jobs are
durable and survive a server restart.

## Where data and logs live

| run mode | data folder |
| --- | --- |
| installed app (`CTXAgent.exe`) | `%LOCALAPPDATA%\CTXAgent` (macOS `~/Library/Application Support/CTXAgent`, Linux `$XDG_DATA_HOME/ctxagent`) |
| source checkout | `<checkout>\.ctxagent` (as before, existing chats stay where they are) |
| `pip install` (site-packages) | same as the installed app |
| any mode with `CTXAGENT_DATA` or `--data` | that folder |

Inside the data folder: `ctxagent.db` (chats), `blobs/`, `jobs/`, `artifacts/`,
`desktop.json` (window size/position, last port), `webview/` (the window's
browser profile), `logs/ctxagent.log` (rotated at 2 MB, five kept; workers log
into their job folders), and `logs/console.log` for stray output of the
windowed exe. Logic: `ctxagent/paths.py`.

API keys: the installed app reads environment variables and `<data>\.env`
(`OPENROUTER_API_KEY=...`, `ANTHROPIC_API_KEY=...`); it does not read a `.env`
from the current folder. The app starts without any key; the UI then asks for
one. (Storing keys in Windows Credential Manager from Settings is a separate
workstream, `ctxagent/keys.py`.)

## Import the chats of a checkout

The installed app starts with an empty data folder. To take over the chats of a
source checkout, close CTXAgent and run once:

```powershell
& "$env:LOCALAPPDATA\Programs\CTXAgent\CTXAgentCLI.exe" --import-data "C:\path\to\checkout\.ctxagent"
```

It copies, never moves: the source folder is only read (the database through
SQLite's online backup, so a server still running on the source is fine). The
database, `blobs/`, `jobs/`, artifacts and settings files are copied; `logs/`,
lock files and the webview profile are not. A data folder that already holds
chats is left untouched unless you add `--replace-data`; its database is then
first saved to `<data>\backups\before-import-<time>\ctxagent.db`. Then CTXAgent
starts as usual. `CTXAgent.exe --import-data ...` works too and reports in a
message box.

## Build the installer

```powershell
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

Steps: create `.venv` (git-ignored) with `python` (3.11; `-Python` to choose),
`pip install -r requirements.txt -c packaging\constraints.txt pywebview pyinstaller`,
bundle the tokenizer data (`packaging\prepare_tiktoken.py`, so token counting
works offline), PyInstaller (`packaging\ctxagent.spec`), a self-test of the
bundle, then Inno Setup (`packaging\ctxagent.iss`). Switches: `-Clean` (fresh
venv, build and dist), `-SkipInstaller`, `-Iscc <path>`. Inno Setup 6 per user:
`winget install --id JRSoftware.InnoSetup -e --scope user`.

Output:

* `dist\CTXAgent\` - the onedir bundle: `CTXAgent.exe` (windowed),
  `CTXAgentCLI.exe` (console twin), `_internal\` (Python, `web/`, tokenizer data).
* `dist\installer\CTXAgent-Setup-<version>.exe` - per-user installer, no admin
  rights: installs to `%LOCALAPPDATA%\Programs\CTXAgent`, Start-menu entry,
  optional desktop shortcut. The uninstaller keeps your data unless you tick
  *Also remove my data*; silently: `unins000.exe /VERYSILENT /REMOVEDATA=1`.
  Setup and uninstall ask to close a running CTXAgent first.

The icon is drawn by `packaging\make_icon.py` (Pillow, no artwork files);
`packaging\ctxagent.ico` is its committed output. Dependency pins live in
`packaging\constraints.txt`: the versions the offline tests run against.

Check an installed copy without a model call:

```powershell
& "$env:LOCALAPPDATA\Programs\CTXAgent\CTXAgentCLI.exe" --run-module ctxagent.selftest
```

It reports the static files, the offline tokenizer, the optional document
readers, a background job round trip, and the delegated-agent worker started
both directly and through a job exactly as delegation launches it.

## Frozen helper processes

The harness runs helpers as separate processes: the job supervisor
(`ctxagent.job_worker`), delegated agents (`ctxagent.agent_worker`, through a
job) and the MCP server. In a checkout they start as `python -u <script>`; in the
bundle `sys.executable` is the app and has no `-m`, so they start as
`CTXAgentCLI.exe --run-module ctxagent.<module> ...`. Every spawn site goes through
`paths.module_command()`. The console exe is needed because PowerShell waits
only for console programs and the MCP server speaks stdio. Windowless processes
(the windowed app, detached job supervisors) start console children with
`CREATE_NO_WINDOW`, so no console windows flash up.

MCP server for other agents, from the installed app:

```json
{ "command": "C:\\Users\\<you>\\AppData\\Local\\Programs\\CTXAgent\\CTXAgentCLI.exe",
  "args": ["--run-module", "ctxagent.mcp_server"] }
```

## Linux

A tarball for x86_64 desktops, installed per user without root, built by
`packaging/build_linux.sh`.

### Install, start, update

```bash
tar xzf CTXAgent-0.1.0-linux-x86_64.tar.gz
cd CTXAgent-0.1.0-linux-x86_64
./install.sh                 # --desktop-icon also puts a launcher on the desktop
```

`install.sh` copies the bundle to `~/.local/opt/CTXAgent` (`CTXAGENT_PREFIX`
overrides), links `~/.local/bin/ctxagent` and writes the app-menu entry
`~/.local/share/applications/ctxagent.desktop`. Start CTXAgent from the app menu
or with `ctxagent`; the options of the table above work the same
(`ctxagent --browser`, `--import-data DIR`, ...). At the end `install.sh` names
any system library the window still lacks.

**Update:** unpack the newer tarball and run its `install.sh`. It replaces
`~/.local/opt/CTXAgent` and leaves the data folder alone; it refuses while
CTXAgent runs. **Uninstall:** `~/.local/opt/CTXAgent/uninstall.sh` removes the
app, the command and the menu entry and keeps your data; `--purge` also deletes
the data folder, `~/.config/ctxagent`, `~/.cache/ctxagent` and the API keys
CTXAgent stored in the keyring (the same as *Also remove my data* on Windows).

### System packages

The bundle carries Python, all Python packages, Qt 6 and Qt WebEngine
(Chromium). From the system it needs glibc 2.35 or newer (built on Ubuntu
22.04: Ubuntu 22.04, Debian 12, Fedora 36, Linux Mint 21 or later), an X11 or
Wayland session, and the libraries that belong to the desktop itself and must
match it: Mesa's EGL/GL/GBM, ALSA, D-Bus, fontconfig and freetype, glib, udev,
the Wayland and XCB client libraries, libstdc++. A desktop installation has all
of them. The one that may be missing is NSS, Chromium's certificate and crypto
library (it loads its own modules from the system, so it is not bundled):

| distribution | package |
| --- | --- |
| Ubuntu, Debian, Mint | `sudo apt install libnss3` |
| Fedora | `sudo dnf install nss` |
| Arch, Manjaro | `sudo pacman -S nss` |
| openSUSE | `sudo zypper install mozilla-nss` |

`install.sh` runs `ldd` on the window's libraries and names anything still
missing. Optional: a Secret Service keyring for API keys (below), `zenity` or
`kdialog` for messages when CTXAgent was started from the menu, `xdg-utils` for
*Open data folder* and the browser fallback.

Without a display (`DISPLAY`/`WAYLAND_DISPLAY` unset), or when Qt WebEngine
cannot load (a missing library), CTXAgent serves the same UI in the default web
browser and writes the reason to `logs/ctxagent.log`. Started from the app menu,
a zenity/kdialog question then keeps it stoppable (*Yes* opens the page again,
*No* stops CTXAgent); from a terminal, Ctrl+C stops it.

### Where data and keys live

| what | where |
| --- | --- |
| app | `~/.local/opt/CTXAgent` (`CTXAgent`, `CTXAgentCLI`, `_internal/`) |
| data folder | `$XDG_DATA_HOME/ctxagent`, default `~/.local/share/ctxagent`: chats, settings, logs, the window's browser profile; the same layout as on Windows |
| `.env` (optional) | `<data>/.env` or `$XDG_CONFIG_HOME/ctxagent/.env` (default `~/.config/ctxagent/.env`) |
| API keys | the Secret Service (GNOME Keyring, KWallet, KeePassXC) through `keyring`, service `CTXAgent` |

Keys entered in Settings go to the Secret Service of the desktop session. When
none answers (no keyring daemon, a bare window manager, an SSH session), they go
to `<data>/credentials.json` instead, mode 0600 in a 0700 folder: plain text,
protected by the user account. Settings > API keys then says *File in the data
folder (credentials.json, user-only permissions); no Secret Service keyring
(GNOME Keyring, KWallet) is running*. `CTXAGENT_KEYSTORE=keyring|file|env`
forces a backend (`ctxagent/keys.py`).

Import the chats of a checkout, check an install without a model call, MCP
server:

```bash
ctxagent --import-data ~/path/to/checkout/.ctxagent
~/.local/opt/CTXAgent/CTXAgentCLI --run-module ctxagent.selftest
~/.local/opt/CTXAgent/CTXAgentCLI --run-module ctxagent.mcp_server     # command for an MCP client
```

### How it differs from Windows

* **Window:** pywebview's Qt backend (`pywebview[qt]`: PyQt6 and PyQt6-WebEngine
  wheels, nothing from a system Qt). `desktop.gui_backend()` picks it in the
  bundle; from source pywebview chooses (GTK or Qt, whichever is installed).
* **Chromium sandbox:** it needs unprivileged user namespaces. Ubuntu 24.04 and
  later restrict them with AppArmor for programs without a profile, and some
  hardened kernels switch them off; Chromium then aborts and the window stays
  blank. CTXAgent checks `/proc/sys` at start and, only there, sets
  `QTWEBENGINE_DISABLE_SANDBOX=1`. The page is CTXAgent's own UI on loopback and
  external links open in the system browser.
* **Software rendering:** Chromium's GPU compositing leaves the window white on
  some driver stacks (it did under WSLg, and blank Qt WebEngine windows are known
  on some NVIDIA/Wayland setups). A chat page does not need the GPU, so CTXAgent
  passes `--disable-gpu`. To try the GPU anyway, start it with
  `CTXAGENT_WEBVIEW_GPU=1 ctxagent`, or put `CTXAGENT_WEBVIEW_GPU=1` into
  `~/.config/ctxagent/.env`.
* **Qt identity:** the Qt application is named `ctxagent` whichever way CTXAgent
  was started, so docks match the window to `ctxagent.desktop` (Wayland app id,
  X11 `WM_CLASS`), and Qt WebEngine's own folders are
  `~/.local/share/ctxagent/QtWebEngine` and `~/.cache/ctxagent`.
* **Single instance:** `fcntl.flock` on `<data>/instance.lock`, otherwise as on
  Windows. A second launch un-minimizes and raises the window (a maximized
  window stays maximized); Wayland compositors may only flag it as wanting
  attention instead of raising it.
* **One executable:** Linux has no console/GUI split, so `CTXAgentCLI` is a
  symlink to `CTXAgent`. Workers still start as `CTXAgentCLI --run-module ...`, so
  the exit cleanup recognises them by name. Jobs and the shell tool run
  `bash -lc`.
* **Library path:** the PyInstaller bootloader puts the bundle's libraries on
  `LD_LIBRARY_PATH`. `paths.prepare_runtime()` gives child processes (bash,
  git, `xdg-open`, the browser) the user's value back, so they do not load the
  bundled OpenSSL or glib. The self-test checks this (`child_library_path`).
* **Messages** that Windows shows in a message box (already running, import
  results, the browser fallback) use zenity or kdialog when CTXAgent was started
  from the menu, and the terminal otherwise.

### Build the tarball

```bash
packaging/build_linux.sh --python python3.12      # --clean, --skip-tarball
```

Steps: `.venv-linux` (git-ignored) from the given interpreter, `pip install -r
requirements.txt -c packaging/constraints.txt -c packaging/constraints-linux.txt
"pywebview[qt]" pyinstaller`, the tokenizer data, PyInstaller with the same
`packaging/ctxagent.spec` (its Linux branch: one executable plus the
`CTXAgentCLI` symlink and the PNG window icon; the desktop's own libraries -
graphics stack, D-Bus, glib, fontconfig, NSS, libstdc++ - are left to the
system; Qt's QML, printing, positioning and GTK-theme parts are dropped, and
then every library nothing kept links to; Qt WebEngine translations limited to
English and German), the bundle self-test, then
`dist/linux/CTXAgent-<version>-linux-x86_64.tar.gz` (bundle, `install.sh`,
`uninstall.sh`, `README.txt` from `packaging/linux/`). No root and no system
packages are needed to build; PyInstaller needs `objdump` and `ldd` (binutils,
libc), as it always does on Linux.

Size: the bundle is about 600 MB unpacked, the tarball about 245 MB. Most of
it is Qt WebEngine (Chromium: 195 MB in one library, plus ICU and the Qt
libraries it links); Windows uses the system's WebView2 instead.

Use a distribution, deadsnakes or python.org Python 3.11/3.12 with the `venv`
module and a shared `libpython`. A venv on top of a conda Python does not work:
PyInstaller then pairs conda's `pyexpat` with the system's older `libexpat` and
the bundle fails its self-test. The bundle runs on the glibc of the build host
and newer, so build on the oldest distribution you want to support.

## Known limits

* Windows and Linux (x86_64) are built and tested. The macOS code paths exist
  (data folder, `fcntl` lock, pywebview's own backends), but there is no build
  script or installer for it.
* The Linux build was tested under WSLg (Ubuntu 22.04 on Windows 11), with
  NSS unpacked from Ubuntu's packages because that system lacks it, not yet on
  a GNOME or KDE desktop. Checked there, on Wayland and on X11: `install.sh`,
  the bundle self-test, the window with a chat against a local fake model
  endpoint, a second launch handing over, the close question while a job runs,
  closing (no process left behind), the browser fallback without NSS, the key
  store fallback and `uninstall.sh --purge`. Not checked: the Secret Service
  backend (no keyring daemon there), the Ubuntu 24.04 sandbox path (unit test
  only), the GPU path on real drivers, and zenity/kdialog messages.
* The Linux bundle includes PyQt6, which is GPLv3: fine for personal use; a
  public binary release would have to follow the GPL (or move to PySide6,
  LGPL).
* There is no AppImage: it would need FUSE (`libfuse2`, not installed by
  default on current Ubuntu) and an extra build tool, and installs no menu
  entry by itself.
* JavaScript page rendering and screenshots of web pages (`playwright`) are not
  bundled. Those browser actions report "JS browsing needs playwright"; plain
  HTTP fetching (`web_open`, `web_find`, search) works.
* The installer is not code-signed, so SmartScreen warns on first run.
* The WebView2 runtime must be present (it ships with Windows 11 and current
  Windows 10). Without it CTXAgent uses the default browser.
* The window keeps its own browser profile in `<data>\webview`; localStorage is
  tied to the page origin, so it survives restarts only while the remembered
  port stays free (otherwise a new port is picked).
* `--import-data` copies whole folders and has no merge: importing into a data
  folder with chats replaces its database (after a backup).
* The uninstaller's *remove data* removes `%LOCALAPPDATA%\CTXAgent` only, not a
  folder chosen with `CTXAGENT_DATA`, and not keys stored in Windows Credential
  Manager.
