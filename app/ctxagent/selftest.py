"""Offline self-check of an install: `CTXAgentCLI.exe --run-module ctxagent.selftest`
(from source: `python -m ctxagent.selftest`). No model calls, no network.

Checks that the static files and the bundled tokenizer data are present, which
optional document readers import, and that helper processes start in this
install mode: a real background job round trip (jobs -> job_worker -> shell)
and the delegated-agent worker entry point; also the single-instance lock, the
key store in use and, on Linux, the Qt WebEngine window backend and that shell
children do not inherit the bundle's LD_LIBRARY_PATH. Uses a temporary data
folder unless --data is given. Prints a JSON report; exit code 1 if a required
check fails.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OPTIONAL = ["fitz", "docx", "pptx", "openpyxl", "pandas", "yaml", "bs4", "chardet", "lxml",
            "PIL", "psutil", "mcp", "jsonschema", "webview"]


def _linux_report(desktop) -> dict:
    """Window backend and folders on Linux. Importing Qt WebEngine needs system
    libraries (libnss3, ...): reported, not required, so a build host without them
    still builds."""
    import importlib.util
    from ctxagent import paths
    info: dict = {"gui": desktop.gui_backend(), "display": desktop.has_display(),
                  "user_namespaces_restricted": desktop.userns_restricted(),
                  "default_data": str(paths.user_data_dir()), "config": str(paths.user_config_dir())}
    try:
        found = importlib.util.find_spec("PyQt6.QtWebEngineWidgets") is not None
    except (ImportError, ValueError):
        found = False
    if found and paths.is_frozen():  # the Chromium helper process ships as a plain file
        found = any((paths.app_root() / "PyQt6").rglob("QtWebEngineProcess"))
    info["qt_webengine_files"] = found
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
        info["qt_webengine_import"] = True
    except Exception as exc:  # noqa: BLE001
        info["qt_webengine_import"] = f"{type(exc).__name__}: {exc}"[:160]
    return info


def main(argv: list[str]) -> int:
    temp = None
    if "--data" in argv:
        os.environ["CTXAGENT_DATA"] = argv[argv.index("--data") + 1]
    else:
        temp = tempfile.mkdtemp(prefix="ctxagent-selftest-")
        os.environ["CTXAGENT_DATA"] = temp
    from ctxagent import paths
    paths.prepare_runtime()
    from ctxagent import config, jobs, tokens
    from ctxagent.version import APP_NAME, __version__
    report: dict = {"app": APP_NAME, "version": __version__, "mode": paths.run_mode(),
                    "executable": sys.executable, "data": str(config.DATA_DIR), "checks": {}}
    checks = report["checks"]
    checks["web"] = (config.APP_DIR / "web" / "index.html").is_file()
    checks["tokenizer"] = tokens._ENC is not None and tokens.count("hello world") == 2
    # Imported lazily by the app, so a bundle without them only fails at first use: the
    # direct Anthropic API (bare Claude ids) and the credential store (Settings > API keys).
    for name, module in (("anthropic_sdk", "anthropic"), ("keyring", "keyring")):
        try:
            __import__(module)
            checks[name] = True
        except Exception:  # noqa: BLE001
            checks[name] = False
    report["optional_modules"] = {}
    for name in OPTIONAL:
        try:
            __import__(name)
            report["optional_modules"][name] = True
        except Exception as exc:  # noqa: BLE001
            report["optional_modules"][name] = f"{type(exc).__name__}: {exc}"[:120]

    # One instance per data folder: the OS lock must be exclusive on this platform.
    from ctxagent import desktop, keys
    first, second = desktop.InstanceLock(config.DATA_DIR), desktop.InstanceLock(config.DATA_DIR)
    try:
        checks["single_instance"] = first.acquire() and not second.acquire()
    finally:
        first.release()
        second.release()
    try:
        report["keystore"] = keys.backend_info()
    except Exception as exc:  # noqa: BLE001 - reported, the env fallback still works
        report["keystore"] = {"error": f"{type(exc).__name__}: {exc}"[:120]}
    if os.name != "nt" and paths.is_frozen():
        # Children (shell commands, xdg-open, the browser) must not load the bundle's libraries.
        bundle = str(paths.app_root())
        checks["child_library_path"] = bundle not in os.environ.get("LD_LIBRARY_PATH", "")
    if os.name != "nt" and sys.platform != "darwin":
        report["linux"] = _linux_report(desktop)
        if paths.is_frozen():
            checks["qt_webengine_bundled"] = report["linux"]["qt_webengine_files"]

    # Background job: jobs.start -> job_worker (via paths.module_command) -> shell.
    work = Path(tempfile.mkdtemp(prefix="ctxagent-selftest-cwd-"))
    try:
        job = jobs.start("echo selftest-ok", work, "selftest", timeout=60)
        deadline = time.monotonic() + 60
        status = jobs.status(job["id"], "selftest")
        while status["state"] not in jobs.TERMINAL and time.monotonic() < deadline:
            time.sleep(0.2)
            status = jobs.status(job["id"], "selftest")
        out = jobs.read_log(job["id"], "selftest")["text"]
        checks["job_round_trip"] = status["state"] == "succeeded" and "selftest-ok" in out
        report["job"] = {"state": status["state"], "stdout": out.strip()[:200],
                         "worker": paths.module_command("ctxagent.job_worker", "<dir>")[:3]}
    except Exception as exc:  # noqa: BLE001
        checks["job_round_trip"] = False
        report["job"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Delegated-agent worker: must start, open the database and reject an unknown id.
    try:
        proc = subprocess.run(paths.module_command("ctxagent.agent_worker", "selftest-unknown-id"),
                              capture_output=True, text=True, timeout=120, cwd=str(paths.worker_cwd()))
        checks["agent_worker_entry"] = proc.returncode != 0 and "No such delegated agent" in (proc.stderr + proc.stdout)
        report["agent_worker"] = {"exit": proc.returncode, "tail": (proc.stderr or proc.stdout).strip()[-200:]}
    except Exception as exc:  # noqa: BLE001
        checks["agent_worker_entry"] = False
        report["agent_worker"] = {"error": f"{type(exc).__name__}: {exc}"}

    # The delegation launch path: a job whose shell command starts the agent
    # worker exactly as delegation._launch builds it (PowerShell must wait for it).
    try:
        command = paths.shell_join(paths.module_command("ctxagent.agent_worker", "selftest-unknown-id"))
        job = jobs.start(command, paths.worker_cwd(), "selftest", timeout=120)
        deadline = time.monotonic() + 120
        status = jobs.status(job["id"], "selftest")
        while status["state"] not in jobs.TERMINAL and time.monotonic() < deadline:
            time.sleep(0.2)
            status = jobs.status(job["id"], "selftest")
        err = jobs.read_log(job["id"], "selftest", stream="stderr")["text"]
        checks["agent_worker_via_job"] = (status["state"] == "failed" and status.get("exit_code") == 1
                                          and "No such delegated agent" in err)
        report["agent_via_job"] = {"state": status["state"], "exit": status.get("exit_code")}
    except Exception as exc:  # noqa: BLE001
        checks["agent_worker_via_job"] = False
        report["agent_via_job"] = {"error": f"{type(exc).__name__}: {exc}"}

    # Canvas script worker: Pillow and numpy in its own process (canvas.run_script).
    try:
        from .canvas import Canvas
        canvas = Canvas()
        canvas.open_blank(64, 48)
        _, output = canvas.run_script("draw.rectangle([0, 0, 9, 9], fill='red')\n"
                                      "print('canvas-ok', W, H, np.__name__)", timeout=120)
        checks["canvas_worker"] = ("canvas-ok 64 48 numpy" in output
                                   and canvas.image().getpixel((5, 5))[:3] == (255, 0, 0))
        report["canvas_worker"] = {"version": canvas.current()["version"], "stdout": output.strip()[-200:]}
    except Exception as exc:  # noqa: BLE001
        checks["canvas_worker"] = False
        report["canvas_worker"] = {"error": f"{type(exc).__name__}: {exc}"}

    report["ok"] = all(checks.values())
    print(json.dumps(report, indent=2))
    import shutil
    for folder in (work, temp):
        if folder:
            shutil.rmtree(folder, ignore_errors=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
