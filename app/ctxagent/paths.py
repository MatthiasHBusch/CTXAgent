"""Where code, static files and user data live, in every way CTXAgent can run.

Three run modes:

* source     a git checkout: `python -m ctxagent[.desktop]`. Data stays in
             <repo>/.ctxagent so existing chats survive.
* frozen     the PyInstaller bundle (installed CTXAgent.exe). Data goes to the
             per-user application-data folder (%LOCALAPPDATA%\\CTXAgent; Linux
             $XDG_DATA_HOME/ctxagent, default ~/.local/share/ctxagent). The folder
             of the app's former name (PiAgent, Linux piagent) is moved there on
             first start.
* installed  a plain `pip install` (package in site-packages). Same data folder
             as frozen: site-packages is no place for chats.

CTXAGENT_DATA overrides the data folder in every mode.

Helper processes (agent_worker, job_worker, mcp_server) are spawned through
module_command(): `python -u <script>` from source, `CTXAgentCLI.exe
--run-module ctxagent.<name>` when frozen, because a frozen sys.executable is the
app itself and has no `-m`.

Standard library only: config.py imports this before anything else.
"""
from __future__ import annotations

import functools
import os
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

from .version import APP_NAME, LEGACY_APP_NAMES

RUN_MODULE_FLAG = "--run-module"
# Console twin of the windowed CTXAgent.exe in the same bundle. Workers need a
# console-subsystem executable: PowerShell does not wait for GUI programs, and
# the MCP server talks over stdio.
CLI_EXE_STEM = APP_NAME + "CLI"
_PACKAGE_DIR = Path(__file__).resolve().parent


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def source_root() -> Path:
    """The directory containing the `ctxagent` package."""
    return _PACKAGE_DIR.parent


def _is_checkout(root: Path) -> bool:
    return (root / "web" / "index.html").is_file() and (root / "ctxagent").is_dir()


def run_mode() -> str:
    if is_frozen():
        return "frozen"
    return "source" if _is_checkout(source_root()) else "installed"


def app_root() -> Path:
    """The directory that holds `web/` (config.APP_DIR)."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    root = source_root()
    if _is_checkout(root):
        return root
    return _PACKAGE_DIR  # wheel install: web/ ships as ctxagent/web


def install_dir() -> Path:
    """Where the executables live (frozen) or the checkout root (source)."""
    return Path(sys.executable).resolve().parent if is_frozen() else source_root()


def user_data_dir(platform: str | None = None, env: Mapping[str, str] | None = None,
                  home: Path | None = None, name: str = APP_NAME) -> Path:
    """Per-user application-data folder, without platformdirs (`name`: the app's name,
    or a former one)."""
    platform = platform or sys.platform
    env = os.environ if env is None else env
    home = Path(home) if home is not None else Path.home()
    if platform == "win32":
        base = env.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(base) / name
    if platform == "darwin":
        return home / "Library" / "Application Support" / name
    base = env.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(base) / name.lower()


def _user_data(env: Mapping[str, str]) -> Path:
    """The per-user data folder. One left by the app under a former name (PiAgent: chats,
    settings, logs, key index) is renamed to it once; if that fails (the old app still
    running from it), the old folder stays in use and the move is retried next start."""
    new = user_data_dir(env=env)
    if new.exists():
        return new
    for old_name in LEGACY_APP_NAMES:
        old = user_data_dir(env=env, name=old_name)
        if old.is_dir() and old != new:
            try:
                old.rename(new)
                return new
            except OSError:
                return old
    return new


def user_config_dir(platform: str | None = None, env: Mapping[str, str] | None = None,
                    home: Path | None = None, name: str = APP_NAME) -> Path:
    """Per-user configuration folder. Only XDG platforms (Linux, BSD) keep it apart
    from the data folder: $XDG_CONFIG_HOME/ctxagent, default ~/.config/ctxagent
    (`name`: the app's name, or a former one)."""
    platform = platform or sys.platform
    env = os.environ if env is None else env
    home = Path(home) if home is not None else Path.home()
    if platform in ("win32", "darwin"):
        return user_data_dir(platform, env, home, name)
    base = env.get("XDG_CONFIG_HOME") or str(home / ".config")
    return Path(base) / name.lower()


def data_dir(env: Mapping[str, str] | None = None, mode: str | None = None) -> Path:
    env = os.environ if env is None else env
    override = (env.get("CTXAGENT_DATA") or "").strip()
    if override:
        return Path(override).expanduser()
    mode = mode or run_mode()
    if mode == "source":
        return source_root() / ".ctxagent"
    return _user_data(env)


def log_dir(data: Path | None = None) -> Path:
    return Path(data if data is not None else data_dir()) / "logs"


def env_files(data: Path, mode: str | None = None) -> list[Path]:
    """.env files config.load_env() reads, first wins per key.

    Frozen/installed apps read only <data>/.env (on Linux/BSD also
    $XDG_CONFIG_HOME/ctxagent/.env, then the former name's ~/.config/piagent/.env): a
    stray .env in whatever folder a shortcut starts in must not redirect keys
    (CTXAGENT_OPENROUTER_URL).
    """
    mode = mode or run_mode()
    candidates = [source_root() / ".env", Path.cwd() / ".env"] if mode == "source" else []
    candidates.append(Path(data) / ".env")
    if mode != "source" and sys.platform not in ("win32", "darwin"):
        candidates.append(user_config_dir() / ".env")
        candidates += [user_config_dir(name=old) / ".env" for old in LEGACY_APP_NAMES]
    out: list[Path] = []
    for path in candidates:
        if path not in out:
            out.append(path)
    return out


# ---------------------------------------------------------------------------
# Spawning helper modules
# ---------------------------------------------------------------------------

def _python_for_workers() -> str:
    exe = Path(sys.executable)
    # pythonw.exe is a GUI program; PowerShell would not wait for it.
    if exe.name.lower() == "pythonw.exe" and exe.with_name("python.exe").exists():
        return str(exe.with_name("python.exe"))
    return sys.executable


def worker_executable() -> str:
    if not is_frozen():
        return _python_for_workers()
    exe = Path(sys.executable).resolve()
    cli = exe.with_name(CLI_EXE_STEM + exe.suffix)
    return str(cli if cli.exists() else exe)


def module_script(module: str) -> Path | None:
    """Source file of a ctxagent module (absolute, resolved), or None."""
    if module == "ctxagent" or not module.startswith("ctxagent."):
        return None
    return _PACKAGE_DIR.joinpath(*module.split(".")[1:]).with_suffix(".py")


def module_command(module: str, *args: str, frozen: bool | None = None) -> list[str]:
    """argv that runs `python -m <module> <args>` in this install.

    From source the script runs by absolute path (no dependence on cwd or
    PYTHONPATH; each worker puts the checkout on sys.path itself).
    """
    frozen = is_frozen() if frozen is None else frozen
    if frozen:
        return [worker_executable(), RUN_MODULE_FLAG, module, *map(str, args)]
    script = module_script(module)
    if script is not None and script.is_file():
        return [_python_for_workers(), "-u", str(script), *map(str, args)]
    return [_python_for_workers(), "-u", "-m", module, *map(str, args)]


def worker_cwd() -> Path:
    """An existing directory to start helpers in."""
    return install_dir()


def is_module_invocation(cmdline: Sequence[str], module: str) -> bool:
    """Whether a process command line runs `module` in any of the forms above."""
    cmdline = list(cmdline)
    script = module_script(module)
    if script is not None:
        target = os.path.normcase(str(script))
        if any(os.path.normcase(part) == target for part in cmdline):
            return True
    for flag in (RUN_MODULE_FLAG, "-m"):
        for i, part in enumerate(cmdline[:-1]):
            if part == flag and cmdline[i + 1] == module:
                return True
    return False


def shell_join(argv: Sequence[str], platform: str | None = None) -> str:
    """One shell command line (jobs run commands through PowerShell / bash)."""
    platform = platform or sys.platform
    if platform == "win32":
        return "& " + " ".join("'" + str(a).replace("'", "''") + "'" for a in argv)
    import shlex
    return " ".join(shlex.quote(str(a)) for a in argv)


_MODULE_NAME = re.compile(r"ctxagent(\.[A-Za-z_][A-Za-z0-9_]*)+")


def run_module(module: str, args: Sequence[str]) -> int:
    """The frozen `--run-module` dispatch: run a ctxagent module as __main__."""
    import runpy
    if not _MODULE_NAME.fullmatch(module or ""):
        print(f"{RUN_MODULE_FLAG} runs ctxagent modules only, not {module!r}", file=sys.stderr)
        return 2
    prepare_runtime()
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


# ---------------------------------------------------------------------------
# Process environment for frozen and console-less runs
# ---------------------------------------------------------------------------

def prepare_runtime() -> None:
    """Call once at process start, before ctxagent.tokens is imported."""
    if is_frozen():
        cache = app_root() / "tiktoken_cache"
        if cache.is_dir():
            os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
        if os.name != "nt":
            restore_library_path()
    hide_child_consoles()


def restore_library_path(env: dict | None = None, bundle: Path | None = None) -> bool:
    """Linux/BSD bundle: the PyInstaller bootloader puts the bundle's own libraries
    first on LD_LIBRARY_PATH (the user's value saved as LD_LIBRARY_PATH_ORIG), and
    every child inherits that: bash, git, xdg-open, the browser. The bundled
    OpenSSL, readline or glib then break system programs. Give children the
    user's value back; this process keeps the libraries it has loaded, and our
    own workers are bundle executables that set the path again themselves."""
    env = os.environ if env is None else env
    if "LD_LIBRARY_PATH_ORIG" in env:
        value = env.pop("LD_LIBRARY_PATH_ORIG")
    elif "LD_LIBRARY_PATH" in env:  # the user had none: drop the bundle's entries
        prefix = str(bundle or app_root())
        value = ":".join(p for p in env["LD_LIBRARY_PATH"].split(":") if p and not p.startswith(prefix))
    else:
        return False
    if value:
        env["LD_LIBRARY_PATH"] = value
    else:
        env.pop("LD_LIBRARY_PATH", None)
    return True


@functools.lru_cache(maxsize=1)
def hide_child_consoles() -> bool:
    """Windows: a process without a console (windowed exe, pythonw, detached
    worker) would open a visible console window for every console program it
    starts (PowerShell for the shell tool and jobs). Default such children to
    CREATE_NO_WINDOW. Processes that have a console share it, as before."""
    if os.name != "nt":
        return False
    import ctypes
    import subprocess
    try:
        if ctypes.windll.kernel32.GetConsoleWindow():
            return False
    except (AttributeError, OSError):
        return False
    original = subprocess.Popen.__init__
    explicit = subprocess.CREATE_NEW_CONSOLE | subprocess.DETACHED_PROCESS

    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        if len(args) <= 13:  # creationflags not passed positionally
            flags = kwargs.get("creationflags") or 0
            if not flags & explicit:
                kwargs["creationflags"] = flags | subprocess.CREATE_NO_WINDOW
        original(self, *args, **kwargs)

    subprocess.Popen.__init__ = __init__
    return True
