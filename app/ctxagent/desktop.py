"""CTXAgent desktop shell: the server in-process on loopback, a native window on top.

    CTXAgent.exe                         native window (pywebview: Edge WebView2; Linux: Qt WebEngine)
    CTXAgent.exe --browser               the system browser instead of a window
    CTXAgent.exe --server                server only (what `python -m ctxagent` runs)
    CTXAgent.exe --import-data DIR       copy an existing data folder in, then start
    CTXAgentCLI.exe --run-module M ...   run helper module M as __main__ (frozen workers)

From a checkout: `python -m ctxagent.desktop [options]`.

One instance per data folder. The first instance holds an OS lock on
<data>/instance.lock and writes <data>/instance.json (port, pid, a random
token). A second launch finds it there and asks it to come to the front (or to
open the browser again) instead of starting a second server on the same data.

Closing the window stops the app: running turns get a few seconds to wind down,
background jobs and delegated agents are cancelled (after a confirmation), then
the server stops and no helper process is left behind. `--server` keeps the
development semantics: jobs are durable and survive a restart.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import logging.handlers
import os
import secrets
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

from . import paths
from .version import APP_NAME, APP_USER_MODEL_ID, __version__

log = logging.getLogger("ctxagent.desktop")

HOST = "127.0.0.1"   # loopback only, never 0.0.0.0
DEFAULT_SIZE = (1280, 860)
MIN_SIZE = (720, 480)
TURN_GRACE_SECONDS = 5.0
JOB_GRACE_SECONDS = 5.0
TERMINAL_JOB_STATES = {"succeeded", "failed", "cancelled", "timed_out", "lost"}
# Direct children we may stop at exit: our workers and the shells they run.
# Never browsers, WebView2 or conhost.
HELPER_PROCESS_NAMES = {"python.exe", "pythonw.exe", "python", "python3", "powershell.exe",
                        "pwsh.exe", "pwsh", "cmd.exe", "bash", "bash.exe", "sh",
                        (paths.CLI_EXE_STEM + ".exe").lower(), paths.CLI_EXE_STEM.lower()}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


# The windowed exe (and pythonw) start without stdout/stderr; setup_logging
# later points them at a log file, so remember what we started with.
_HAD_STREAMS = sys.stdout is not None and sys.stderr is not None


def has_console() -> bool:
    """Whether text printed now reaches a person (a console or a capturing parent)."""
    return _HAD_STREAMS


def _xdg_desktop() -> bool:
    """Linux/BSD (not Windows, not macOS)."""
    return os.name != "nt" and sys.platform != "darwin"


def has_display() -> bool:
    return not _xdg_desktop() or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _without_terminal() -> bool:
    """Linux bundle started from the app menu: stdout exists, but nobody reads it."""
    if not (_xdg_desktop() and paths.is_frozen()):
        return False
    try:
        return not (sys.stderr is not None and sys.stderr.isatty())
    except (AttributeError, ValueError):
        return True


def _desktop_dialog(text: str, kind: str = "info") -> bool | None:
    """Linux: a zenity or kdialog box; kind is info, error or question (True = Yes).
    None when neither tool or no graphical session is there."""
    if not has_display():
        return None
    import subprocess
    if shutil.which("zenity"):
        argv = ["zenity", f"--{kind}", f"--title={APP_NAME}", f"--text={text}", "--no-markup"]
    elif shutil.which("kdialog"):
        flag = {"info": "--msgbox", "error": "--error", "question": "--yesno"}[kind]
        argv = ["kdialog", "--title", APP_NAME, flag, text]
    else:
        return None
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0
    except OSError:
        return None


def notify(text: str, error: bool = False, title: str = APP_NAME) -> None:
    """Tell the user something: the console if there is one, else a message box."""
    (log.error if error else log.info)(text.replace("\n", " "))
    if _without_terminal() and _desktop_dialog(text, "error" if error else "info") is not None:
        return
    if has_console() or os.name != "nt":
        stream = sys.stderr if error else sys.stdout
        if stream is not None:
            print(text, file=stream, flush=True)
        return
    import ctypes
    ctypes.windll.user32.MessageBoxW(None, text, title, 0x10 if error else 0x40)


def _message_box(text: str, flags: int, owner: int | None = None) -> int:
    import ctypes
    return ctypes.windll.user32.MessageBoxW(owner, text, APP_NAME, flags)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(data: Path, console: bool, verbose: bool = False) -> Path:
    """<data>/logs/ctxagent.log, rotated at 2 MB (5 kept). Windowed runs have no
    stdout/stderr; stray prints and tracebacks go to logs/console.log."""
    logs = paths.log_dir(data)
    logs.mkdir(parents=True, exist_ok=True)
    logfile = logs / "ctxagent.log"
    root = logging.getLogger()
    if any(getattr(h, "_ctxagent", False) for h in root.handlers):
        return logfile
    handler = logging.handlers.RotatingFileHandler(logfile, maxBytes=2_000_000, backupCount=5,
                                                   encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(process)d %(levelname)s %(name)s: %(message)s"))
    handler._ctxagent = True
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if console and sys.stderr is not None:
        echo = logging.StreamHandler(sys.stderr)
        echo.setLevel(logging.INFO if verbose else logging.WARNING)
        echo.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        echo._ctxagent = True
        root.addHandler(echo)
    if sys.stdout is None or sys.stderr is None:
        stray = logs / "console.log"
        if stray.exists() and stray.stat().st_size > 2_000_000:
            os.replace(stray, logs / "console.log.1")
        stream = open(stray, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = stream
        if sys.stderr is None:
            sys.stderr = stream
    threading.excepthook = lambda a: log.error("thread %s crashed", a.thread and a.thread.name,
                                               exc_info=(a.exc_type, a.exc_value, a.exc_traceback))
    logging.captureWarnings(True)
    return logfile


# ---------------------------------------------------------------------------
# Single instance
# ---------------------------------------------------------------------------

class InstanceLock:
    """Exclusive, non-blocking OS lock on <data>/instance.lock for the process
    lifetime. The OS drops it when the process dies, so it cannot go stale."""

    def __init__(self, data: Path):
        self.path = Path(data) / "instance.lock"
        self._stream = None

    def acquire(self) -> bool:
        if self._stream is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self._stream = stream
        return True

    def release(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()

    @property
    def held(self) -> bool:
        return self._stream is not None


def instance_file(data: Path) -> Path:
    return Path(data) / "instance.json"


def write_instance(data: Path, info: dict) -> None:
    _write_json(instance_file(data), info)


def remove_instance(data: Path, token: str) -> None:
    if _read_json(instance_file(data)).get("token") == token:
        instance_file(data).unlink(missing_ok=True)


def probe_health(url: str, timeout: float = 2.0) -> dict | None:
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/api/health", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def _post(url: str, token: str, timeout: float = 5.0) -> dict | None:
    request = urllib.request.Request(url, data=b"{}", method="POST",
                                     headers={"X-CTXAgent-Token": token, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None


def find_running(data: Path, lock: InstanceLock, wait: float = 15.0) -> dict | None:
    """The live instance that holds the lock, or None once the lock is ours.

    Waits while the other instance is still starting (lock held, no answer yet)."""
    deadline = time.monotonic() + wait
    while True:
        info = _read_json(instance_file(data))
        if info.get("url"):
            health = probe_health(info["url"])
            if health and health.get("server_pid") == info.get("pid"):
                return info
        if lock.acquire():
            return None
        if time.monotonic() >= deadline:
            return {}
        time.sleep(0.3)


def hand_over(info: dict, mode: str) -> int:
    """Second launch: bring the running instance forward instead of starting."""
    url = info["url"]
    if os.name == "nt" and info.get("pid"):
        try:
            import ctypes
            ctypes.windll.user32.AllowSetForegroundWindow(int(info["pid"]))
        except (AttributeError, OSError, ValueError):
            pass
    if info.get("token") and info.get("mode") in ("window", "browser"):
        answer = _post(url.rstrip("/") + "/api/desktop/focus", info["token"])
        if answer and answer.get("ok"):
            log.info("focused running instance pid %s", info.get("pid"))
            return 0
    # A plain server instance (python -m ctxagent): show it without a second server.
    log.info("opening a view on running server %s (pid %s)", url, info.get("pid"))
    if mode == "window" and webview_available():
        return 0 if run_window(url, None, None, None) else 1
    webbrowser.open(url)
    return 0


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------

def bind_loopback(port: int = 0) -> socket.socket:
    """A socket bound to 127.0.0.1 (0 = any free port), not yet listening.

    Windows: SO_EXCLUSIVEADDRUSE, so no other process can bind the same port
    alongside us; elsewhere SO_REUSEADDR so a restart is not blocked by TIME_WAIT."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((HOST, int(port)))
    except OSError:
        sock.close()
        raise
    return sock


def pick_free_port() -> int:
    with bind_loopback(0) as sock:
        return sock.getsockname()[1]


def preferred_port(data: Path) -> int:
    """Stable per data folder, so the page origin (and with it the webview's
    localStorage) stays the same between launches. Falls back when taken."""
    remembered = _read_json(Path(data) / "desktop.json").get("port")
    if isinstance(remembered, int) and 1024 < remembered < 65536:
        return remembered
    digest = hashlib.sha256(str(Path(data).resolve()).lower().encode("utf-8")).digest()
    return 20000 + int.from_bytes(digest[:4], "big") % 10000


def bind_preferred(port: int) -> socket.socket:
    try:
        return bind_loopback(port)
    except OSError:
        log.info("port %s is taken, using a free one", port)
        return bind_loopback(0)


# ---------------------------------------------------------------------------
# Importing an existing data folder
# ---------------------------------------------------------------------------

class ImportDataError(Exception):
    pass


DB_NAME = "ctxagent.db"
_IMPORT_SKIP = {"logs", "instance.lock", "instance.json", "webview", "failed-test-data"}


def count_chats(db: Path) -> int:
    if not Path(db).is_file():
        return 0
    try:
        conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
        try:
            return conn.execute("SELECT count(*) FROM chats").fetchone()[0]
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def _sqlite_copy(source: Path, dest: Path) -> None:
    """Consistent copy (includes the WAL) even while another process uses it."""
    try:
        src = sqlite3.connect(Path(source).resolve().as_uri() + "?mode=ro", uri=True)
        src.execute("SELECT count(*) FROM sqlite_master").fetchone()
    except sqlite3.Error:
        src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(dest))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def import_data(source: Path, target: Path, replace: bool = False) -> dict:
    """Copy an existing data folder (chats, blobs, jobs, settings) into `target`.

    Copies, never moves: `source` is only read. A target that already holds
    chats is left alone unless `replace`; then its database is first backed up
    to <target>/backups/before-import-<time>/."""
    source = Path(source).expanduser().resolve()
    target = Path(target).expanduser().resolve()
    if not (source / DB_NAME).is_file():
        raise ImportDataError(f"{source} is not a {APP_NAME} data folder (no {DB_NAME} in it).")
    if source == target or source in target.parents or target in source.parents:
        raise ImportDataError(f"Source {source} and data folder {target} overlap.")
    existing = count_chats(target / DB_NAME)
    backup = None
    if existing and not replace:
        raise ImportDataError(
            f"{target} already holds {existing} chat(s). Nothing was changed.\n"
            f"Re-run with --replace-data to back that database up to "
            f"{target / 'backups'} and import anyway.")
    target.mkdir(parents=True, exist_ok=True)
    if (target / DB_NAME).exists():
        if existing:
            backup = target / "backups" / time.strftime("before-import-%Y%m%d-%H%M%S")
            backup.mkdir(parents=True, exist_ok=False)
            _sqlite_copy(target / DB_NAME, backup / DB_NAME)
    staging = target / (DB_NAME + ".importing")
    staging.unlink(missing_ok=True)
    _sqlite_copy(source / DB_NAME, staging)
    for suffix in ("-wal", "-shm", "-journal"):
        (target / (DB_NAME + suffix)).unlink(missing_ok=True)
    os.replace(staging, target / DB_NAME)
    copied = 0
    for item in sorted(source.iterdir()):
        name = item.name
        if name in _IMPORT_SKIP or name.startswith(DB_NAME) or name.endswith(".lock"):
            continue
        dest = target / name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            copied += sum(1 for p in item.rglob("*") if p.is_file())
        elif item.is_file():
            shutil.copy2(item, dest)
            copied += 1
    result = {"source": str(source), "target": str(target), "chats": count_chats(target / DB_NAME),
              "files": copied, "backup": str(backup) if backup else None}
    log.info("imported data %s", result)
    return result


# ---------------------------------------------------------------------------
# Background work: turns, jobs, helper processes
# ---------------------------------------------------------------------------

def running_jobs(data: Path) -> list[dict]:
    """Jobs (including delegated agents) whose supervisor is still alive."""
    root = Path(data) / "jobs"
    if not root.is_dir():
        return []
    import psutil
    out = []
    for status_file in root.glob("*/status.json"):
        status = _read_json(status_file)
        if not status or status.get("state") in TERMINAL_JOB_STATES:
            continue
        pid = status.get("worker_pid")
        if pid:
            try:
                proc = psutil.Process(pid)
                if abs(proc.create_time() - status.get("worker_created", 0)) < 0.1 and proc.is_running():
                    out.append({"dir": status_file.parent, "process": proc})
            except psutil.Error:
                pass
        elif time.time() - status.get("created_at", 0) < 30:
            out.append({"dir": status_file.parent, "process": None})
    return out


def stop_jobs(jobs: list[dict], timeout: float = JOB_GRACE_SECONDS) -> int:
    """Ask each supervisor to cancel (the marker file jobs.cancel uses) and wait."""
    import psutil
    for job in jobs:
        try:
            (job["dir"] / "cancel").touch()
        except OSError:
            pass
    procs = [j["process"] for j in jobs if j["process"] is not None]
    if procs:
        _, alive = psutil.wait_procs(procs, timeout=timeout)
        return len(alive)
    return 0


def kill_helper_children(timeout: float = 3.0) -> int:
    """Kill leftover worker/shell child trees of this process (not browsers,
    not WebView2, not conhost). Returns how many processes were killed."""
    import psutil
    victims = []
    for child in psutil.Process().children(recursive=False):
        try:
            name = child.name().lower()
            if name in HELPER_PROCESS_NAMES or name.startswith("python"):
                victims.extend(child.children(recursive=True))
                victims.append(child)
        except psutil.Error:
            continue
    for proc in victims:
        try:
            proc.kill()
        except psutil.Error:
            pass
    if victims:
        psutil.wait_procs(victims, timeout=timeout)
        log.info("killed %d helper process(es) at exit", len(victims))
    return len(victims)


# ---------------------------------------------------------------------------
# The in-process server
# ---------------------------------------------------------------------------

class ServerHost:
    """uvicorn on a pre-bound loopback socket, in a background thread."""

    def __init__(self, app, sock: socket.socket):
        import uvicorn
        self.sock = sock
        self.port = sock.getsockname()[1]
        self.url = f"http://{HOST}:{self.port}/"
        self.config = uvicorn.Config(app, log_config=None, log_level="info", access_log=False,
                                     timeout_graceful_shutdown=3, lifespan="on")
        self.server = uvicorn.Server(self.config)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, name="ctxagent-server", daemon=True)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            loop.run_until_complete(self.server.serve(sockets=[self.sock]))
        except BaseException as exc:  # noqa: BLE001 - reported to the user
            self.error = exc
            log.exception("server stopped with an error")
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()

    def start(self, timeout: float = 60.0) -> None:
        self.thread.start()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.thread.is_alive():
                raise RuntimeError(f"server failed to start: {self.error!r}")
            if self.server.started and probe_health(self.url, timeout=2.0):
                log.info("server ready on %s", self.url)
                return
            time.sleep(0.05)
        raise RuntimeError(f"server did not answer on {self.url} within {timeout:.0f}s")

    def running_turns(self) -> int:
        try:
            from . import agent
            sessions = list(getattr(agent, "_SESSIONS", {}).values())
            return sum(1 for s in sessions if s.turn_lock.locked())
        except Exception:  # noqa: BLE001 - best effort
            return 0

    async def _wind_down(self, grace: float) -> tuple[int, int]:
        from . import agent
        busy = [s for s in list(getattr(agent, "_SESSIONS", {}).values()) if s.turn_lock.locked()]
        for s in busy:
            s.cancel.set()
            task = getattr(s, "browser_task", None)
            if task is not None and not task.done():
                task.cancel()
        deadline = time.monotonic() + grace
        while any(s.turn_lock.locked() for s in busy) and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        return len(busy), sum(1 for s in busy if s.turn_lock.locked())

    def stop(self, grace: float = TURN_GRACE_SECONDS) -> None:
        if self.loop is not None and self.thread.is_alive() and grace > 0:
            try:
                future = asyncio.run_coroutine_threadsafe(self._wind_down(grace), self.loop)
                busy, left = future.result(grace + 2)
                if busy:
                    log.info("stopped %d running turn(s), %d still busy after %.0fs", busy, left, grace)
            except Exception:  # noqa: BLE001
                log.exception("could not wind down running turns")
        self.server.should_exit = True
        self.thread.join(10)
        if self.thread.is_alive():
            self.server.force_exit = True
            self.thread.join(5)
        try:
            self.sock.close()
        except OSError:
            pass


class Shell:
    """What the focus endpoint acts on: the window, or the browser fallback."""

    def __init__(self, mode: str, url: str, token: str):
        self.mode = mode
        self.url = url
        self.token = token
        self.window = None
        self.state: WindowState | None = None

    def focus(self) -> None:
        window = self.window
        if window is None:
            if self.mode == "browser":
                webbrowser.open(self.url)
            return
        try:
            window.show()
            if os.name == "nt" and window.native is not None:
                import ctypes
                hwnd = int(window.native.Handle.ToInt64())
                if ctypes.windll.user32.IsIconic(hwnd):
                    ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE keeps maximized state
                ctypes.windll.user32.SetForegroundWindow(hwnd)
            else:
                # Qt's restore() also un-maximizes; maximize again where it was.
                maximized = self.state is not None and self.state.maximized
                window.restore()
                if maximized:
                    window.maximize()
        except Exception:  # noqa: BLE001
            log.exception("could not focus the window")


def install_routes(app, shell: Shell) -> None:
    """POST /api/desktop/focus, guarded by the per-launch token in instance.json
    (a web page cannot read it, and the custom header forces a CORS preflight)."""
    from fastapi import HTTPException, Request
    app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/api/desktop/focus"]

    async def focus(request):
        if not secrets.compare_digest(request.headers.get("x-ctxagent-token", ""), shell.token):
            raise HTTPException(403, "bad token")
        await asyncio.to_thread(shell.focus)
        return {"ok": True, "mode": shell.mode}

    # Real objects, not strings: this module uses postponed annotations and
    # Request is imported locally, so FastAPI could not resolve "Request".
    focus.__annotations__ = {"request": Request, "return": dict}
    app.add_api_route("/api/desktop/focus", focus, methods=["POST"], include_in_schema=False)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

def webview2_available() -> bool:
    """Edge WebView2 runtime >= 86 installed (per machine or per user)."""
    if os.name != "nt":
        return True
    import winreg
    key = r"Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    for hive, path in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\\" + key),
                       (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\\" + key),
                       (winreg.HKEY_CURRENT_USER, r"SOFTWARE\\" + key)):
        try:
            with winreg.OpenKey(hive, path) as handle:
                version, _ = winreg.QueryValueEx(handle, "pv")
            if int(str(version).split(".")[0]) >= 86:
                return True
        except (OSError, ValueError):
            continue
    return False


def gui_backend() -> str | None:
    """pywebview GUI: Edge WebView2 on Windows; the Linux bundle ships Qt WebEngine
    (PyQt6) only. None lets pywebview pick (macOS, Linux from source)."""
    if os.name == "nt":
        return "edgechromium"
    if _xdg_desktop() and paths.is_frozen():
        return "qt"
    return None


def webview_available() -> bool:
    if os.environ.get("CTXAGENT_NO_WEBVIEW") or os.environ.get("PIAGENT_NO_WEBVIEW"):
        return False
    if not has_display():
        log.warning("no graphical session (DISPLAY and WAYLAND_DISPLAY unset); using the system browser")
        return False
    try:
        import webview  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - ImportError, pythonnet/clr failures
        log.warning("pywebview unavailable (%s); using the system browser", exc)
        return False
    if not webview2_available():
        log.warning("Edge WebView2 runtime not found; using the system browser")
        return False
    if gui_backend() == "qt":
        try:
            import PyQt6.QtWebEngineWidgets  # noqa: F401 - must load before the QApplication
        except Exception as exc:  # noqa: BLE001 - ImportError for a missing system library
            log.warning("Qt WebEngine unavailable (%s); using the system browser. Missing system "
                        "libraries (libnss3, ...)? See README.txt of the Linux tarball.", exc)
            return False
    return True


def userns_restricted(proc: Path = Path("/proc/sys")) -> bool:
    """Whether unprivileged user namespaces are off (Ubuntu 24.04+ AppArmor policy,
    hardened kernels): Chromium's sandbox, and with it Qt WebEngine, cannot start."""
    for name, blocked in (("kernel/apparmor_restrict_unprivileged_userns", "1"),
                          ("kernel/unprivileged_userns_clone", "0"),
                          ("user/max_user_namespaces", "0")):
        try:
            if (proc / name).read_text().strip() == blocked:
                return True
        except OSError:
            continue
    return False


def prepare_qtwebengine(env: dict | None = None) -> None:
    """Linux: two ways Qt WebEngine shows a blank window.

    * Without user namespaces Chromium's sandbox aborts. The page is the app's own
      UI on loopback (external links open in the system browser), so there it
      runs without the sandbox.
    * Chromium's GPU compositing fails on some driver stacks (WSLg's dma-buf path,
      some NVIDIA/Wayland setups) and leaves the window white. A chat UI does not
      need the GPU: software rendering unless CTXAGENT_WEBVIEW_GPU=1 (or the former
      PIAGENT_WEBVIEW_GPU=1)."""
    env = os.environ if env is None else env
    if "QTWEBENGINE_DISABLE_SANDBOX" not in env and userns_restricted():
        env["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
        log.info("user namespaces are restricted: Qt WebEngine runs without the Chromium sandbox")
    flags = env.get("QTWEBENGINE_CHROMIUM_FLAGS", "").split()
    gpu = env.get("CTXAGENT_WEBVIEW_GPU") or env.get("PIAGENT_WEBVIEW_GPU")
    if gpu != "1" and "--disable-gpu" not in flags:
        env["QTWEBENGINE_CHROMIUM_FLAGS"] = " ".join(flags + ["--disable-gpu"])


def qt_identity() -> None:
    """Name the Qt application after the desktop entry (ctxagent.desktop) instead of
    argv[0], which is `ctxagent` or `CTXAgent` depending on how it was started: docks
    match the window by it (Wayland app_id, X11 WM_CLASS), and Qt WebEngine's own
    folders become ~/.local/share/ctxagent and ~/.cache/ctxagent."""
    from PyQt6.QtCore import QCoreApplication
    from PyQt6.QtGui import QGuiApplication
    QCoreApplication.setApplicationName(APP_NAME.lower())
    QGuiApplication.setDesktopFileName(APP_NAME.lower())


class WindowState:
    """Size, position and maximized flag in <data>/desktop.json (logical pixels)."""

    def __init__(self, data: Path | None):
        self.path = Path(data) / "desktop.json" if data is not None else None
        saved = _read_json(self.path).get("window", {}) if self.path else {}
        self.width = int(saved.get("width") or DEFAULT_SIZE[0])
        self.height = int(saved.get("height") or DEFAULT_SIZE[1])
        self.x = saved.get("x")
        self.y = saved.get("y")
        self.maximized = bool(saved.get("maximized"))
        self.width = max(self.width, MIN_SIZE[0])
        self.height = max(self.height, MIN_SIZE[1])

    def clamp_to_screens(self, screens) -> None:
        """Forget a position that is no longer on any monitor."""
        if self.x is None or self.y is None:
            return
        for s in screens or []:
            if (s.x - 50 <= self.x < s.x + s.width - 100) and (s.y - 10 <= self.y < s.y + s.height - 50):
                return
        self.x = self.y = None

    def attach(self, window) -> None:
        def resized(width, height):
            if not self.maximized and width >= MIN_SIZE[0] // 2:
                self.width, self.height = width, height

        def moved(x, y):
            if not self.maximized and x > -10000 and y > -10000:  # minimized windows sit at -32000
                self.x, self.y = x, y

        def maximized():
            self.maximized = True

        def restored():
            self.maximized = False

        window.events.resized += resized
        window.events.moved += moved
        window.events.maximized += maximized
        window.events.restored += restored

    def capture(self, window) -> None:
        """Read the final geometry (a window that was never moved fired no event)."""
        if self.maximized:
            return
        try:
            x, y, width, height = window.x, window.y, window.width, window.height
        except Exception:  # noqa: BLE001
            return
        if x > -10000 and y > -10000 and width >= MIN_SIZE[0] // 2:
            self.x, self.y, self.width, self.height = x, y, width, height

    def save(self, port: int | None = None) -> None:
        if self.path is None:
            return
        data = _read_json(self.path)
        data["window"] = {"width": self.width, "height": self.height, "x": self.x, "y": self.y,
                          "maximized": self.maximized}
        if port:
            data["port"] = port
        try:
            _write_json(self.path, data)
        except OSError:
            log.exception("could not save window state")


SPLASH = """<!doctype html><html><head><meta charset="utf-8"><style>
:root{color-scheme:light dark}body{margin:0;height:100vh;display:grid;place-items:center;
font:15px system-ui,sans-serif;background:Canvas;color:CanvasText}
.b{width:28px;height:28px;border:3px solid #8884;border-top-color:#1a96a0;border-radius:50%;
animation:s 1s linear infinite;margin:0 auto 14px}@keyframes s{to{transform:rotate(360deg)}}
p{opacity:.7;margin:0;text-align:center}</style></head><body><div><div class="b"></div>
<p>Starting __APP__ ...</p></div></body></html>""".replace("__APP__", APP_NAME)


def _error_page(message: str, logfile: Path | None) -> str:
    import html
    where = f"<p>Log: <code>{html.escape(str(logfile))}</code></p>" if logfile else ""
    return (f"<!doctype html><meta charset='utf-8'><body style='font:15px system-ui;margin:40px'>"
            f"<h2>{APP_NAME} could not start</h2><pre style='white-space:pre-wrap'>"
            f"{html.escape(message)}</pre>{where}</body>")


def _window_icon() -> str | None:
    if os.name == "nt":
        if paths.is_frozen():
            return None  # the exe's own icon
        icon = paths.source_root() / "packaging" / "ctxagent.ico"
    elif paths.is_frozen():
        icon = paths.app_root() / "ctxagent.png"  # Linux executables carry no icon
    else:
        icon = paths.source_root() / "packaging" / "ctxagent.png"
    return str(icon) if icon.is_file() else None


def run_window(url: str | None, state: WindowState | None, shell: Shell | None,
               boot=None, on_close=None, data: Path | None = None, debug: bool = False) -> bool:
    """Show the native window; blocks until it is closed. `boot(window)` runs in
    a thread once the GUI is up (start the server, then window.load_url).
    `on_close()` -> False keeps the window open. Returns False when the
    webview could not start at all (the caller then falls back to the browser)."""
    if _xdg_desktop():
        prepare_qtwebengine()
        if gui_backend() == "qt":
            qt_identity()
    import webview
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except (AttributeError, OSError):
            pass
    state = state or WindowState(None)
    try:
        state.clamp_to_screens(webview.screens)
    except Exception:  # noqa: BLE001
        state.x = state.y = None
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    kwargs: dict[str, Any] = dict(width=state.width, height=state.height, min_size=MIN_SIZE,
                                  maximized=state.maximized, text_select=True, zoomable=True)
    if state.x is not None and state.y is not None:
        kwargs.update(x=int(state.x), y=int(state.y))
    if url:
        kwargs["url"] = url
    else:
        kwargs["html"] = SPLASH
    window = webview.create_window(APP_NAME, **kwargs)
    if shell is not None:
        shell.window = window
    state.attach(window)

    def closing():
        if on_close is not None and on_close(window) is False:
            return False
        state.capture(window)
        state.save()
        log.info("window closing")
        return True

    window.events.closing += closing
    storage = None
    if data is not None:
        storage = Path(data) / "webview"
        storage.mkdir(parents=True, exist_ok=True)
    try:
        webview.start(boot, window if boot else None, gui=gui_backend(),
                      debug=debug, private_mode=False,
                      storage_path=str(storage) if storage else None, icon=_window_icon())
    except Exception:  # noqa: BLE001
        log.exception("the native window failed")
        return False
    finally:
        if shell is not None:
            shell.window = None
    return True


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def _busy_text(turns: int, jobs: int) -> str:
    parts = []
    if turns:
        parts.append(f"{turns} chat turn{'s' if turns != 1 else ''} still running")
    if jobs:
        parts.append(f"{jobs} background job{'s' if jobs != 1 else ''} or delegated agent{'s' if jobs != 1 else ''} active")
    return " and ".join(parts)


class App:
    def __init__(self, data: Path, mode: str, args: argparse.Namespace, logfile: Path):
        self.data = data
        self.mode = mode
        self.args = args
        self.logfile = logfile
        self.token = secrets.token_urlsafe(24)
        self.host: ServerHost | None = None
        self.shell: Shell | None = None
        self.confirmed_close = False

    # -- server ----------------------------------------------------------
    def bind(self) -> socket.socket:
        if self.mode == "server":
            port = self.args.port or int(os.environ.get("CTXAGENT_PORT", "8765"))
            return bind_loopback(port)
        return bind_loopback(self.args.port) if self.args.port else bind_preferred(preferred_port(self.data))

    def start_server(self, sock: socket.socket) -> str:
        from . import server
        self.host = ServerHost(server.app, sock)
        self.shell = self.shell or Shell(self.mode, self.host.url, self.token)
        self.shell.url = self.host.url
        install_routes(server.app, self.shell)
        self.host.start()
        write_instance(self.data, {"pid": os.getpid(), "port": self.host.port, "url": self.host.url,
                                   "token": self.token, "mode": self.mode, "version": __version__,
                                   "data": str(self.data), "started_at": time.time()})
        return self.host.url

    def shutdown(self, stop_background: bool) -> None:
        started = time.monotonic()
        if self.host is not None:
            self.host.stop(TURN_GRACE_SECONDS)
        if stop_background:
            jobs = running_jobs(self.data)
            if jobs:
                left = stop_jobs(jobs)
                log.info("cancelled %d background job(s), %d did not stop in time", len(jobs), left)
            kill_helper_children()
        remove_instance(self.data, self.token)
        log.info("shut down in %.1fs", time.monotonic() - started)

    # -- window ------------------------------------------------------------
    def confirm_close(self, window) -> bool:
        if self.confirmed_close or self.host is None:
            return True
        text = _busy_text(self.host.running_turns(), len(running_jobs(self.data)))
        if not text:
            return True
        message = (f"{text[0].upper() + text[1:]}.\n\nClosing {APP_NAME} stops them. "
                   f"Close anyway?")
        if os.name == "nt":
            owner = None
            try:
                owner = int(window.native.Handle.ToInt64())
            except Exception:  # noqa: BLE001
                pass
            ok = _message_box(message, 0x1 | 0x30 | 0x100, owner) == 1  # OKCANCEL|WARNING|DEFBUTTON2
        else:
            ok = bool(window.create_confirmation_dialog(APP_NAME, message))
        self.confirmed_close = ok
        return ok

    def run_window(self, sock: socket.socket) -> int:
        state = WindowState(self.data)
        self.shell = Shell("window", "", self.token)
        self.shell.state = state
        failure: list[str] = []
        booted = threading.Event()

        def boot(window):
            try:
                url = self.start_server(sock)
                state.save(self.host.port)
                window.load_url(url)
            except Exception as exc:  # noqa: BLE001
                log.exception("startup failed")
                failure.append(str(exc))
                try:
                    window.load_html(_error_page(f"{type(exc).__name__}: {exc}", self.logfile))
                except Exception:  # noqa: BLE001 - window already closed
                    pass
            finally:
                booted.set()

        shown = run_window(None, state, self.shell, boot=boot, on_close=self.confirm_close,
                           data=self.data, debug=self.args.debug)
        log.info("window loop ended (shown=%s)", shown)
        if shown:
            booted.wait(70)  # closed while starting: let start_server finish, then stop it
        if not shown and self.host is None:
            log.warning("falling back to the system browser")
            self.mode = "browser"
            return self.run_browser(sock)
        if not shown:
            # Window died after the server was up: keep serving in the browser.
            self.mode = self.shell.mode = "browser"
            webbrowser.open(self.host.url)
            self.wait_browser()
        self.shutdown(stop_background=True)
        return 1 if failure else 0

    # -- browser / server ----------------------------------------------------
    def wait_browser(self) -> None:
        url = self.host.url
        # No console, no tray: a message box is the whole UI of the fallback.
        text = (f"{APP_NAME} is running in your web browser at\n{url}\n\n"
                f"Yes: open it in the browser again.\nNo: stop {APP_NAME} "
                f"(running turns and background jobs stop too).")
        if os.name == "nt" and not has_console():
            while self.host.thread.is_alive():
                if _message_box(text, 0x4 | 0x40) == 6:  # YESNO|INFORMATION -> IDYES
                    webbrowser.open(url)
                    continue
                break
            return
        if _without_terminal():
            while self.host.thread.is_alive():
                answer = _desktop_dialog(text, "question")
                if answer is None:
                    break  # no zenity/kdialog: serve until the process is ended
                if not answer:
                    return
                webbrowser.open(url)
        print(f"{APP_NAME} is running at {url} (Ctrl+C to stop)", flush=True)
        try:
            while self.host.thread.is_alive():
                self.host.thread.join(0.5)
        except KeyboardInterrupt:
            print("stopping ...", flush=True)

    def run_browser(self, sock: socket.socket) -> int:
        self.shell = Shell("browser", "", self.token)
        self.start_server(sock)
        webbrowser.open(self.host.url)
        self.wait_browser()
        self.shutdown(stop_background=True)
        return 0

    def run_server(self, sock: socket.socket) -> int:
        url = self.start_server(sock)
        print(f"{APP_NAME} server on {url} (Ctrl+C to stop)", flush=True)
        if self.args.open:
            webbrowser.open(url)
        try:
            while self.host.thread.is_alive():
                self.host.thread.join(0.5)
        except KeyboardInterrupt:
            print("stopping ...", flush=True)
        self.shutdown(stop_background=False)
        return 1 if self.host.error else 0


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog=APP_NAME, description=f"{APP_NAME} {__version__} desktop app")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--browser", action="store_true", help="use the system browser instead of a window")
    mode.add_argument("--server", action="store_true", help="run the server only (development)")
    p.add_argument("--open", action="store_true", help="with --server: open the browser once it is up")
    p.add_argument("--port", type=int, default=0, help="fixed port (default: remembered or free)")
    p.add_argument("--data", metavar="DIR", help="data folder (same as CTXAGENT_DATA)")
    p.add_argument("--import-data", metavar="DIR",
                   help="copy an existing data folder (e.g. <checkout>/.ctxagent) into the data folder first")
    p.add_argument("--replace-data", action="store_true",
                   help="with --import-data: back up and replace a data folder that already has chats")
    p.add_argument("--debug", action="store_true", help="enable the webview developer tools and verbose logs")
    p.add_argument("--no-fast-exit", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    if sys.stdout is not None and sys.stderr is not None:
        return p.parse_args(argv)
    # Windowed exe: no streams. Show --help/--version/usage errors in a message box.
    import contextlib
    import io
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            return p.parse_args(argv)
    except SystemExit:
        if os.name == "nt":
            _message_box(buffer.getvalue() or "invalid arguments", 0x40)
        raise


def legacy_server(data: Path) -> dict | None:
    """A server from before the instance lock (an old `python -m ctxagent`)
    serving this checkout's default data folder. Checked from source only."""
    if paths.run_mode() != "source" or os.environ.get("CTXAGENT_DATA"):
        return None
    if Path(data).resolve() != (paths.source_root() / ".ctxagent").resolve():
        return None
    url = f"http://{HOST}:{int(os.environ.get('CTXAGENT_PORT', '8765'))}/"
    health = probe_health(url, timeout=1.0)
    if not health or "backend_build" not in health or not health.get("server_pid"):
        return None
    try:
        import psutil
        cwd = Path(psutil.Process(int(health["server_pid"])).cwd()).resolve()
    except Exception:  # noqa: BLE001 - gone, or not ours to inspect
        return None
    if cwd != paths.source_root().resolve():
        return None
    return {"url": url, "pid": health["server_pid"], "mode": "server"}


_RUNNING_MUTEX = None


def _hold_running_mutex() -> None:
    """Windows: a named mutex for the process lifetime; the installer's AppMutex
    (packaging/ctxagent.iss) uses it to ask the user to close the app first."""
    global _RUNNING_MUTEX
    if os.name == "nt" and _RUNNING_MUTEX is None:
        try:
            import ctypes
            _RUNNING_MUTEX = ctypes.windll.kernel32.CreateMutexW(None, False, f"{APP_NAME}.Running")
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == [paths.RUN_MODULE_FLAG]:
        if len(argv) < 2:
            print(f"usage: {APP_NAME} {paths.RUN_MODULE_FLAG} ctxagent.<module> [args...]", file=sys.stderr)
            return 2
        return paths.run_module(argv[1], argv[2:])
    args = parse_args(argv)
    if args.data:
        os.environ["CTXAGENT_DATA"] = str(Path(args.data).expanduser().resolve())
    paths.prepare_runtime()
    data = paths.data_dir()
    data.mkdir(parents=True, exist_ok=True)
    mode = "server" if args.server else "browser" if args.browser else "window"
    logfile = setup_logging(data, console=has_console(), verbose=args.debug)
    log.info("%s %s starting: mode=%s run=%s data=%s exe=%s", APP_NAME, __version__, mode,
             paths.run_mode(), data, sys.executable)
    lock = InstanceLock(data)
    if not lock.acquire():
        running = find_running(data, lock, wait=0 if args.import_data else 15)
        if running is not None:
            if args.import_data:
                notify(f"Close the running {APP_NAME} first, then import again.", error=True)
                return 1
            if not running:
                notify(f"{APP_NAME} is already running with the data folder\n{data}\n"
                       "but does not answer. Wait a moment, or end it in "
                       + ("Task Manager." if os.name == "nt" else f"the system monitor (pkill -x {APP_NAME})."),
                       error=True)
                return 1
            if mode == "server":
                notify(f"{APP_NAME} is already running at {running['url']} "
                       f"(pid {running.get('pid')}, data {data}).", error=True)
                return 1
            return hand_over(running, mode)
    _hold_running_mutex()
    try:
        from . import config
        config.load_env()
        legacy = legacy_server(data)
        if legacy is not None:
            if mode == "server" or args.import_data:
                notify(f"A {APP_NAME} server for this checkout is already running at {legacy['url']} "
                       f"(pid {legacy['pid']}). Stop it first.", error=True)
                return 1
            return hand_over(legacy, mode)
        if args.import_data:
            try:
                result = import_data(Path(args.import_data), data, replace=args.replace_data)
            except (ImportDataError, OSError, sqlite3.Error) as exc:
                notify(f"Import failed: {exc}", error=True)
                return 1
            notify(f"Imported {result['chats']} chat(s) and {result['files']} file(s) from\n"
                   f"{result['source']}\ninto {result['target']}"
                   + (f"\n(previous database backed up to {result['backup']})" if result["backup"] else ""))
        app = App(data, mode, args, logfile)
        try:
            sock = app.bind()
        except OSError as exc:
            port = args.port or os.environ.get("CTXAGENT_PORT", "8765")
            notify(f"Port {port} is not available ({exc}).", error=True)
            return 1
        if mode == "window" and webview_available():
            code = app.run_window(sock)
            if not args.no_fast_exit:
                # Everything is stopped and saved; skip the multi-second .NET
                # (pythonnet) runtime teardown at interpreter exit.
                lock.release()
                log.info("exit %s", code)
                logging.shutdown()
                for stream in (sys.stdout, sys.stderr):
                    try:
                        stream.flush()
                    except Exception:  # noqa: BLE001
                        pass
                os._exit(code)
            return code
        if mode == "server":
            return app.run_server(sock)
        app.mode = "browser"
        return app.run_browser(sock)
    finally:
        lock.release()
        log.info("exit")


def serve_main(argv: list[str] | None = None) -> int:
    """`python -m ctxagent` / the `ctxagent` console script: server only."""
    return main(["--server", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
