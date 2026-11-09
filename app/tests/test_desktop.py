"""Offline tests of the desktop shell: data paths per run mode, worker spawning
in source and frozen mode, ports, the single-instance lock, data import.

No window is opened and no model is called. Real subprocesses are started for
the job round trip, the lock and the server-mode refusal of a second instance.
"""
import atexit
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PurePosixPath
from unittest.mock import patch

if not os.environ.get("CTXAGENT_DATA"):  # standalone run: never touch <repo>/.ctxagent
    _DATA = tempfile.mkdtemp(prefix="ctxagent-desktop-tests-")
    os.environ["CTXAGENT_DATA"] = _DATA
    atexit.register(shutil.rmtree, _DATA, True)

from ctxagent import desktop, paths  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


class FrozenMixin:
    """Pretend to be the PyInstaller bundle in <tmp>/CTXAgent."""

    def frozen(self, with_cli=True):
        tmp = Path(tempfile.mkdtemp(prefix="piagent-frozen-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        bundle = tmp / "CTXAgent"
        (bundle / "_internal" / "web").mkdir(parents=True)
        exe = bundle / "CTXAgent.exe"
        exe.write_bytes(b"")
        if with_cli:
            (bundle / "CTXAgentCLI.exe").write_bytes(b"")
        for name, value in (("frozen", True), ("_MEIPASS", str(bundle / "_internal")),
                            ("executable", str(exe))):
            p = patch.object(sys, name, value, create=True)
            p.start()
            self.addCleanup(p.stop)
        return bundle


class PathTests(FrozenMixin, unittest.TestCase):
    def test_source_mode(self):
        self.assertEqual(paths.run_mode(), "source")
        self.assertEqual(paths.app_root(), ROOT)
        self.assertTrue((paths.app_root() / "web" / "index.html").is_file())
        self.assertEqual(paths.data_dir(env={}), ROOT / ".ctxagent")
        self.assertEqual(paths.log_dir(Path("x")), Path("x") / "logs")

    def test_override_wins_in_every_mode(self):
        for mode in ("source", "frozen", "installed"):
            self.assertEqual(paths.data_dir(env={"CTXAGENT_DATA": "D:/elsewhere"}, mode=mode),
                             Path("D:/elsewhere"))
        # empty means unset, not the current directory
        self.assertEqual(paths.data_dir(env={"CTXAGENT_DATA": " "}, mode="source"), ROOT / ".ctxagent")

    def test_frozen_mode(self):
        bundle = self.frozen()
        self.assertEqual(paths.run_mode(), "frozen")
        self.assertEqual(paths.app_root(), bundle / "_internal")
        self.assertEqual(paths.install_dir(), bundle)
        self.assertEqual(paths.data_dir(env={"LOCALAPPDATA": "C:/Users/u/AppData/Local"}),
                         Path("C:/Users/u/AppData/Local") / "CTXAgent")
        self.assertEqual(paths.env_files(Path("D:/data")), [Path("D:/data") / ".env"])

    def test_installed_package_uses_user_folder(self):
        self.assertEqual(paths.data_dir(env={"LOCALAPPDATA": "L:/"}, mode="installed"), Path("L:/") / "CTXAgent")

    def test_data_folder_of_the_former_name_is_taken_over(self):
        # The app was called PiAgent: its folder (chats, settings, key index) moves once.
        base = Path(tempfile.mkdtemp(prefix="ctxagent-rename-"))
        self.addCleanup(shutil.rmtree, base, True)
        old = base / "PiAgent"
        (old / "logs").mkdir(parents=True)
        (old / "chats.db").write_bytes(b"db")
        env = {"LOCALAPPDATA": str(base)}
        self.assertEqual(paths.data_dir(env=env, mode="frozen"), base / "CTXAgent")
        self.assertEqual((base / "CTXAgent" / "chats.db").read_bytes(), b"db")
        self.assertFalse(old.exists())
        # Both present (the old app was started again): the new folder wins, the old one stays.
        (old / "logs").mkdir(parents=True)
        self.assertEqual(paths.data_dir(env=env, mode="installed"), base / "CTXAgent")
        self.assertTrue(old.exists())
        # Moving fails (the old app still running from it): keep working in the old folder.
        shutil.rmtree(base / "CTXAgent")
        with patch.object(Path, "rename", side_effect=PermissionError("in use")):
            self.assertEqual(paths.data_dir(env=env, mode="frozen"), old)

    def test_user_data_dir_per_platform(self):
        home = Path("/home/u")
        self.assertEqual(paths.user_data_dir("win32", {"LOCALAPPDATA": "C:/L"}, home), Path("C:/L/CTXAgent"))
        self.assertEqual(paths.user_data_dir("win32", {}, home), home / "AppData" / "Local" / "CTXAgent")
        self.assertEqual(paths.user_data_dir("darwin", {}, home),
                         home / "Library" / "Application Support" / "CTXAgent")
        self.assertEqual(paths.user_data_dir("linux", {}, home), home / ".local" / "share" / "ctxagent")
        self.assertEqual(paths.user_data_dir("linux", {"XDG_DATA_HOME": "/x"}, home), Path("/x/ctxagent"))

    def test_env_files_from_source(self):
        files = paths.env_files(Path("D:/data"), mode="source")
        self.assertEqual(files[0], ROOT / ".env")
        self.assertIn(Path.cwd() / ".env", files)
        self.assertEqual(files[-1], Path("D:/data") / ".env")

    def test_config_uses_paths(self):
        from ctxagent import config
        self.assertEqual(config.APP_DIR, paths.app_root())
        self.assertEqual(config.DATA_DIR, Path(os.environ["CTXAGENT_DATA"]))


class DispatchTests(FrozenMixin, unittest.TestCase):
    def test_source_command_runs_script_by_path(self):
        argv = paths.module_command("ctxagent.agent_worker", "abc")
        self.assertEqual(argv[1:], ["-u", str(ROOT / "ctxagent" / "agent_worker.py"), "abc"])
        self.assertTrue(paths.is_module_invocation(argv, "ctxagent.agent_worker"))
        self.assertFalse(paths.is_module_invocation(argv, "ctxagent.job_worker"))

    def test_pythonw_is_replaced_for_workers(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "pythonw.exe").write_bytes(b"")
        (tmp / "python.exe").write_bytes(b"")
        with patch.object(sys, "executable", str(tmp / "pythonw.exe")):
            self.assertEqual(paths.module_command("ctxagent.job_worker")[0], str(tmp / "python.exe"))

    def test_frozen_command_uses_console_exe(self):
        bundle = self.frozen()
        argv = paths.module_command("ctxagent.job_worker", "D:/jobs/1")
        self.assertEqual(argv, [str((bundle / "CTXAgentCLI.exe").resolve()), "--run-module",
                                "ctxagent.job_worker", "D:/jobs/1"])
        self.assertTrue(paths.is_module_invocation(argv, "ctxagent.job_worker"))
        self.assertTrue(paths.is_module_invocation(["python", "-m", "ctxagent.job_worker"], "ctxagent.job_worker"))
        self.assertEqual(paths.worker_cwd(), bundle.resolve())

    def test_frozen_without_cli_falls_back_to_app_exe(self):
        bundle = self.frozen(with_cli=False)
        self.assertEqual(paths.module_command("ctxagent.mcp_server")[0], str((bundle / "CTXAgent.exe").resolve()))

    def test_shell_join_quotes(self):
        self.assertEqual(paths.shell_join(["C:/a b/x.exe", "it's"], platform="win32"),
                         "& 'C:/a b/x.exe' 'it''s'")
        self.assertEqual(paths.shell_join(["/a b/x", "it's"], platform="linux"), "'/a b/x' 'it'\"'\"'s'")

    def test_delegation_launch_goes_through_paths(self):
        from ctxagent import delegation, jobs
        seen = {}

        def fake_start(command, cwd, owner, timeout, publish=None, preserve_managed_children=False):
            seen.update(command=command, cwd=cwd, preserve=preserve_managed_children)
            return {}

        with patch.object(jobs, "start", fake_start):
            delegation._launch("task123", 60, "root")
            self.assertIn("agent_worker.py", seen["command"])
            self.assertTrue(seen["command"].rstrip().endswith("task123'" if os.name == "nt" else "task123"))
            self.assertTrue(seen["preserve"])
            self.frozen()
            delegation._launch("task456", 60, "root")
            self.assertIn("'--run-module' 'ctxagent.agent_worker' 'task456'" if os.name == "nt"
                          else "--run-module ctxagent.agent_worker task456", seen["command"])
            self.assertTrue(Path(seen["cwd"]).is_dir())

    def test_run_module_rejects_foreign_modules(self):
        self.assertEqual(desktop.main(["--run-module", "os"]), 2)
        self.assertEqual(desktop.main(["--run-module", "ctxagent.x;y"]), 2)

    def test_run_module_dispatch_runs_worker(self):
        # The frozen exe runs exactly this path; here under the source interpreter.
        code = "import sys; from ctxagent.desktop import main; sys.exit(main(sys.argv[1:]))"
        proc = subprocess.run([sys.executable, "-B", "-c", code, "--run-module", "ctxagent.agent_worker",
                               "no-such-task"], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 1, proc.stderr[-500:])
        self.assertIn("No such delegated agent", proc.stderr)


class JobTests(unittest.TestCase):
    def setUp(self):
        from ctxagent import config
        self.data = config.DATA_DIR
        self.cwd = Path(tempfile.mkdtemp(prefix="piagent-job-cwd-"))
        self.addCleanup(shutil.rmtree, self.cwd, True)

    def wait(self, job_id, owner, timeout=60):
        from ctxagent import jobs
        deadline = time.monotonic() + timeout
        status = jobs.status(job_id, owner)
        while status["state"] not in jobs.TERMINAL and time.monotonic() < deadline:
            time.sleep(0.1)
            status = jobs.status(job_id, owner)
        return status

    def test_job_round_trip(self):
        from ctxagent import jobs
        job = jobs.start("echo desktop-ok", self.cwd, "desktop-test", timeout=60)
        status = self.wait(job["id"], "desktop-test")
        self.assertEqual(status["state"], "succeeded")
        self.assertIn("desktop-ok", jobs.read_log(job["id"], "desktop-test")["text"])

    def test_running_jobs_are_found_and_stopped(self):
        from ctxagent import jobs
        sleep = "Start-Sleep -Seconds 60" if os.name == "nt" else "sleep 60"
        job = jobs.start(sleep, self.cwd, "desktop-test", timeout=120)
        deadline = time.monotonic() + 30
        found = []
        while time.monotonic() < deadline:
            found = [j for j in desktop.running_jobs(self.data) if j["dir"].name == job["id"]]
            if found and found[0]["process"] is not None:
                break
            time.sleep(0.1)
        self.assertTrue(found, "running job not detected")
        self.assertEqual(desktop.stop_jobs(found, timeout=20), 0)
        self.assertEqual(self.wait(job["id"], "desktop-test")["state"], "cancelled")
        self.assertFalse([j for j in desktop.running_jobs(self.data) if j["dir"].name == job["id"]])

    def test_kill_helper_children(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: child.poll() is None and child.kill())
        self.assertGreaterEqual(desktop.kill_helper_children(timeout=10), 1)
        self.assertIsNotNone(child.wait(timeout=10))


class PortTests(unittest.TestCase):
    def test_free_port_on_loopback(self):
        port = desktop.pick_free_port()
        self.assertTrue(1024 < port < 65536)
        with desktop.bind_loopback(port) as sock:
            self.assertEqual(sock.getsockname(), ("127.0.0.1", port))

    def test_preferred_port_falls_back_when_taken(self):
        with desktop.bind_loopback(0) as taken:
            taken.listen()
            port = taken.getsockname()[1]
            with desktop.bind_preferred(port) as other:
                self.assertNotEqual(other.getsockname()[1], port)

    def test_preferred_port_is_stable_and_remembered(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        first = desktop.preferred_port(tmp)
        self.assertEqual(first, desktop.preferred_port(tmp))
        self.assertTrue(20000 <= first < 30000)
        (tmp / "desktop.json").write_text(json.dumps({"port": 41234}))
        self.assertEqual(desktop.preferred_port(tmp), 41234)


class _Health(BaseHTTPRequestHandler):
    calls = []

    def do_GET(self):
        self._send({"ok": True, "server_pid": os.getpid(), "backend_build": "x"})

    def do_POST(self):
        _Health.calls.append((self.path, self.headers.get("X-CTXAgent-Token")))
        self._send({"ok": True, "mode": "window"})

    def _send(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class InstanceTests(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp(prefix="piagent-instance-"))
        self.addCleanup(shutil.rmtree, self.data, True)

    def test_lock_is_exclusive(self):
        first, second = desktop.InstanceLock(self.data), desktop.InstanceLock(self.data)
        self.assertTrue(first.acquire())
        self.addCleanup(first.release)
        self.assertFalse(second.acquire())
        first.release()
        self.assertTrue(second.acquire())
        second.release()

    def test_lock_held_by_another_process_is_freed_when_it_dies(self):
        code = ("import sys,time; from pathlib import Path; from ctxagent.desktop import InstanceLock; "
                "l=InstanceLock(Path(sys.argv[1])); print(l.acquire(), flush=True); time.sleep(60)")
        proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(self.data)], cwd=ROOT,
                                stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.assertEqual(proc.stdout.readline().strip(), "True")
        lock = desktop.InstanceLock(self.data)
        self.assertFalse(lock.acquire())
        proc.kill()
        proc.wait(10)
        # Windows drops a dead process's byte-range locks shortly after it exits, not at once.
        deadline = time.monotonic() + 5
        while not lock.acquire() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(lock.acquire())
        lock.release()

    def serve_health(self):
        server = HTTPServer(("127.0.0.1", 0), _Health)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_address[1]}/"

    def test_second_launch_finds_and_focuses_the_first(self):
        url = self.serve_health()
        desktop.write_instance(self.data, {"pid": os.getpid(), "url": url, "token": "t0k", "mode": "window"})
        holder = desktop.InstanceLock(self.data)
        self.assertTrue(holder.acquire())
        self.addCleanup(holder.release)
        info = desktop.find_running(self.data, desktop.InstanceLock(self.data), wait=2)
        self.assertEqual(info["url"], url)
        _Health.calls.clear()
        self.assertEqual(desktop.hand_over(info, "window"), 0)
        self.assertEqual(_Health.calls, [("/api/desktop/focus", "t0k")])

    def test_stale_instance_file_does_not_block(self):
        desktop.write_instance(self.data, {"pid": 1, "url": "http://127.0.0.1:9/", "token": "x"})
        lock = desktop.InstanceLock(self.data)
        self.assertIsNone(desktop.find_running(self.data, lock, wait=1))
        self.assertTrue(lock.held)
        lock.release()
        desktop.remove_instance(self.data, "other-token")
        self.assertTrue(desktop.instance_file(self.data).exists())
        desktop.remove_instance(self.data, "x")
        self.assertFalse(desktop.instance_file(self.data).exists())

    def test_server_mode_refuses_a_second_instance(self):
        env = {**os.environ, "CTXAGENT_DATA": str(self.data), "PYTHONIOENCODING": "utf-8"}
        env.pop("OPENROUTER_API_KEY", None)
        env.pop("ANTHROPIC_API_KEY", None)
        port = str(desktop.pick_free_port())
        first = subprocess.Popen([sys.executable, "-B", "-m", "ctxagent", "--port", port], cwd=ROOT, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: first.poll() is None and first.kill())
        deadline = time.monotonic() + 90
        info = {}
        while time.monotonic() < deadline and not info.get("url"):
            info = desktop._read_json(desktop.instance_file(self.data))
            time.sleep(0.2)
        self.assertEqual(info.get("mode"), "server")
        health = desktop.probe_health(info["url"])
        self.assertTrue(health and health["ok"])
        second = subprocess.run([sys.executable, "-B", "-m", "ctxagent", "--port", port], cwd=ROOT, env=env,
                                capture_output=True, text=True, timeout=90)
        self.assertEqual(second.returncode, 1)
        self.assertIn("already running", second.stderr)
        first.kill()
        first.wait(10)


    def test_browser_mode_opens_and_a_second_launch_reopens(self):
        tools = Path(tempfile.mkdtemp(prefix="piagent-browser-"))
        self.addCleanup(shutil.rmtree, tools, True)
        opened = tools / "opened.txt"
        if os.name == "nt":
            fake = tools / "browser.cmd"
            fake.write_text('@echo %1>>"%~dp0opened.txt"\r\n', encoding="ascii")
        else:
            fake = tools / "browser.sh"
            fake.write_text(f'#!/bin/sh\necho "$1" >> "{opened}"\n')
            fake.chmod(0o755)
        env = {**os.environ, "CTXAGENT_DATA": str(self.data), "BROWSER": str(fake), "PYTHONIOENCODING": "utf-8"}
        env.pop("OPENROUTER_API_KEY", None)
        env.pop("ANTHROPIC_API_KEY", None)
        argv = [sys.executable, "-B", "-m", "ctxagent.desktop", "--browser"]
        first = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: first.poll() is None and first.kill())

        def lines():
            return opened.read_text().split() if opened.exists() else []
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline and not lines():
            time.sleep(0.2)
        info = desktop._read_json(desktop.instance_file(self.data))
        self.assertEqual(info.get("mode"), "browser")
        self.assertEqual(lines(), [info["url"]])
        second = subprocess.run(argv, cwd=ROOT, env=env, capture_output=True, text=True, timeout=90)
        self.assertEqual(second.returncode, 0, second.stderr[-500:])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and len(lines()) < 2:
            time.sleep(0.1)
        self.assertEqual(lines(), [info["url"]] * 2)
        first.kill()
        first.wait(10)


class ServerHostTests(unittest.TestCase):
    def test_in_process_server_and_focus_route(self):
        from ctxagent import server
        shell = desktop.Shell("server", "", "secret")
        host = desktop.ServerHost(server.app, desktop.bind_loopback(0))
        desktop.install_routes(server.app, shell)
        desktop.install_routes(server.app, shell)  # idempotent
        self.assertEqual(sum(getattr(r, "path", "") == "/api/desktop/focus" for r in server.app.routes), 1)
        host.start(timeout=60)
        try:
            self.assertTrue(desktop.probe_health(host.url)["ok"])
            self.assertEqual(desktop._post(host.url + "api/desktop/focus", "secret"), {"ok": True, "mode": "server"})
            self.assertIsNone(desktop._post(host.url + "api/desktop/focus", "wrong"))
            self.assertEqual(host.running_turns(), 0)
        finally:
            host.stop(grace=1)
        self.assertFalse(host.thread.is_alive())


def _make_data(folder: Path, chats: int, keep_open=False):
    folder.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(folder / "ctxagent.db")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE IF NOT EXISTS chats (id TEXT PRIMARY KEY, title TEXT)")
    for i in range(chats):
        conn.execute("INSERT INTO chats VALUES (?, ?)", (f"{folder.name}-{i}", f"chat {i}"))
    conn.commit()
    (folder / "blobs").mkdir(exist_ok=True)
    (folder / "blobs" / "b1.txt").write_text("blob", encoding="utf-8")
    (folder / "logs").mkdir(exist_ok=True)
    (folder / "logs" / "piagent.log").write_text("old log", encoding="utf-8")
    (folder / "instance.lock").write_bytes(b"")
    (folder / "providers.json").write_text("{}", encoding="utf-8")
    if keep_open:
        return conn
    conn.close()


class ImportDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="piagent-import-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_copies_into_empty_folder_and_leaves_source(self):
        source, target = self.tmp / "old", self.tmp / "new"
        conn = _make_data(source, 3, keep_open=True)  # uncheckpointed WAL, writer still open
        self.addCleanup(conn.close)
        before = sorted(p.relative_to(source) for p in source.rglob("*"))
        result = desktop.import_data(source, target)
        self.assertEqual(result["chats"], 3)
        self.assertEqual(desktop.count_chats(target / "ctxagent.db"), 3)
        self.assertEqual((target / "blobs" / "b1.txt").read_text(encoding="utf-8"), "blob")
        self.assertTrue((target / "providers.json").exists())
        self.assertFalse((target / "logs").exists())
        self.assertFalse((target / "instance.lock").exists())
        self.assertEqual(sorted(p.relative_to(source) for p in source.rglob("*")), before)
        self.assertEqual(desktop.count_chats(source / "ctxagent.db"), 3)

    def test_fresh_target_database_is_replaced(self):
        source, target = self.tmp / "old", self.tmp / "new"
        _make_data(source, 2)
        _make_data(target, 0)
        self.assertEqual(desktop.import_data(source, target)["chats"], 2)

    def test_target_with_chats_needs_replace_and_is_backed_up(self):
        source, target = self.tmp / "old", self.tmp / "new"
        _make_data(source, 2)
        _make_data(target, 5)
        with self.assertRaises(desktop.ImportDataError):
            desktop.import_data(source, target)
        self.assertEqual(desktop.count_chats(target / "ctxagent.db"), 5)
        result = desktop.import_data(source, target, replace=True)
        self.assertEqual(result["chats"], 2)
        self.assertEqual(desktop.count_chats(Path(result["backup"]) / "ctxagent.db"), 5)

    def test_rejects_non_data_folders_and_overlap(self):
        (self.tmp / "empty").mkdir()
        with self.assertRaises(desktop.ImportDataError):
            desktop.import_data(self.tmp / "empty", self.tmp / "new")
        _make_data(self.tmp / "old", 1)
        with self.assertRaises(desktop.ImportDataError):
            desktop.import_data(self.tmp / "old", self.tmp / "old" / "inner")

    def test_cli_flag_imports_before_start(self):
        import logging
        source, target = self.tmp / "old", self.tmp / "new"
        _make_data(source, 1)
        root = logging.getLogger()
        before = list(root.handlers)
        try:
            with patch.dict(os.environ), \
                    patch.object(desktop.App, "run_server", lambda self, sock: (sock.close(), 0)[1]):
                code = desktop.main(["--server", "--port", str(desktop.pick_free_port()),
                                     "--data", str(target), "--import-data", str(source)])
        finally:
            for handler in [h for h in root.handlers if h not in before]:
                root.removeHandler(handler)
                handler.close()
        self.assertEqual(code, 0)
        self.assertEqual(desktop.count_chats(target / "ctxagent.db"), 1)
        self.assertTrue((target / "logs" / "ctxagent.log").exists())   # the app's own log


class CloseConfirmationTests(unittest.TestCase):
    def app(self, turns, jobs):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        app = desktop.App(tmp, "window", desktop.parse_args([]), tmp / "log")

        class Host:
            def running_turns(self):
                return turns
        app.host = Host()
        patcher = patch.object(desktop, "running_jobs", lambda data: [object()] * jobs)
        patcher.start()
        self.addCleanup(patcher.stop)
        return app

    def test_idle_closes_without_asking(self):
        with patch.object(desktop, "_message_box", side_effect=AssertionError("asked")):
            self.assertTrue(self.app(0, 0).confirm_close(window=None))

    def test_busy_asks_and_respects_cancel(self):
        app = self.app(1, 2)
        with patch.object(desktop, "_message_box", return_value=2) as box:  # IDCANCEL
            self.assertFalse(app.confirm_close(window=None))
        self.assertIn("1 chat turn still running and 2 background jobs", box.call_args[0][0])
        with patch.object(desktop, "_message_box", return_value=1):  # IDOK
            self.assertTrue(app.confirm_close(window=None))
        with patch.object(desktop, "_message_box", side_effect=AssertionError("asked twice")):
            self.assertTrue(app.confirm_close(window=None))


class WindowStateTests(unittest.TestCase):
    def test_defaults_clamp_and_round_trip(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        state = desktop.WindowState(tmp)
        self.assertEqual((state.width, state.height, state.x), (*desktop.DEFAULT_SIZE, None))
        state.width, state.height, state.x, state.y, state.maximized = 1000, 700, 3000, 100, True
        state.save(port=23456)
        again = desktop.WindowState(tmp)
        self.assertEqual((again.width, again.height, again.x, again.y, again.maximized), (1000, 700, 3000, 100, True))
        self.assertEqual(desktop.preferred_port(tmp), 23456)

        class Screen:
            x, y, width, height = 0, 0, 1920, 1080
        again.clamp_to_screens([Screen()])
        self.assertIsNone(again.x)  # 3000 is off the only monitor
        again.x, again.y = 200, 100
        again.clamp_to_screens([Screen()])
        self.assertEqual((again.x, again.y), (200, 100))


class LinuxPlatformTests(unittest.TestCase):
    """The Linux bundle's platform code, testable on any OS."""

    def test_user_config_dir_per_platform(self):
        home = Path("/home/u")
        self.assertEqual(paths.user_config_dir("linux", {}, home), home / ".config" / "ctxagent")
        self.assertEqual(paths.user_config_dir("linux", {"XDG_CONFIG_HOME": "/c"}, home), Path("/c/ctxagent"))
        self.assertEqual(paths.user_config_dir("win32", {"LOCALAPPDATA": "C:/L"}, home), Path("C:/L/CTXAgent"))

    def test_children_get_the_users_library_path_back(self):
        env = {"LD_LIBRARY_PATH": "/opt/PiAgent/_internal:/usr/local/lib", "LD_LIBRARY_PATH_ORIG": "/usr/local/lib"}
        self.assertTrue(paths.restore_library_path(env, PurePosixPath("/opt/PiAgent/_internal")))
        self.assertEqual(env, {"LD_LIBRARY_PATH": "/usr/local/lib"})
        env = {"LD_LIBRARY_PATH": "/opt/PiAgent/_internal"}  # the user had none
        self.assertTrue(paths.restore_library_path(env, PurePosixPath("/opt/PiAgent/_internal")))
        self.assertEqual(env, {})
        self.assertFalse(paths.restore_library_path({}, PurePosixPath("/opt/PiAgent/_internal")))

    def test_sandbox_is_dropped_only_where_user_namespaces_are_restricted(self):
        proc = Path(tempfile.mkdtemp(prefix="piagent-proc-"))
        self.addCleanup(shutil.rmtree, proc, True)
        self.assertFalse(desktop.userns_restricted(proc))  # files absent: not restricted
        (proc / "kernel").mkdir()
        (proc / "kernel" / "apparmor_restrict_unprivileged_userns").write_text("0\n")
        self.assertFalse(desktop.userns_restricted(proc))
        (proc / "kernel" / "apparmor_restrict_unprivileged_userns").write_text("1\n")  # Ubuntu 24.04
        self.assertTrue(desktop.userns_restricted(proc))
        gpu_on = {"CTXAGENT_WEBVIEW_GPU": "1"}
        with patch.object(desktop, "userns_restricted", return_value=True):
            env = dict(gpu_on)
            desktop.prepare_qtwebengine(env)
            self.assertEqual(env, {**gpu_on, "QTWEBENGINE_DISABLE_SANDBOX": "1"})
            env = {**gpu_on, "QTWEBENGINE_DISABLE_SANDBOX": "0"}  # set by the user: left alone
            desktop.prepare_qtwebengine(env)
            self.assertEqual(env, {**gpu_on, "QTWEBENGINE_DISABLE_SANDBOX": "0"})
        with patch.object(desktop, "userns_restricted", return_value=False):
            env = dict(gpu_on)
            desktop.prepare_qtwebengine(env)
            self.assertEqual(env, gpu_on)

    def test_software_rendering_unless_the_gpu_is_asked_for(self):
        with patch.object(desktop, "userns_restricted", return_value=False):
            env = {"QTWEBENGINE_CHROMIUM_FLAGS": "--foo"}
            desktop.prepare_qtwebengine(env)
            self.assertEqual(env, {"QTWEBENGINE_CHROMIUM_FLAGS": "--foo --disable-gpu"})
            desktop.prepare_qtwebengine(env)  # not twice
            self.assertEqual(env, {"QTWEBENGINE_CHROMIUM_FLAGS": "--foo --disable-gpu"})

    def test_focus_keeps_a_maximized_window_maximized(self):
        calls = []

        class Window:
            native = None
            show = lambda self: calls.append("show")  # noqa: E731
            restore = lambda self: calls.append("restore")  # noqa: E731
            maximize = lambda self: calls.append("maximize")  # noqa: E731

        shell = desktop.Shell("window", "", "t")
        shell.window, shell.state = Window(), desktop.WindowState(None)
        shell.state.maximized = True
        shell.focus()
        self.assertEqual(calls, ["show", "restore", "maximize"])
        calls.clear()
        shell.state.maximized = False
        shell.focus()
        self.assertEqual(calls, ["show", "restore"])

    def test_file_key_store_says_why_on_linux(self):
        from ctxagent import keys
        self.assertNotIn(";", keys.FileBackend().label)
        label = keys.FileBackend(reason="no Secret Service keyring is running").label
        self.assertTrue(label.endswith("; no Secret Service keyring is running"))


if __name__ == "__main__":
    unittest.main()
