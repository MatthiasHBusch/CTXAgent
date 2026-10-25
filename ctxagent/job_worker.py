"""Standalone supervisor. Invoked only by jobs.start, with a generated job directory."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

# Executable by absolute script path from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ctxagent.jobs import atomic_json, shell_args  # noqa: E402


def managed_exclusions(children):
    """Prune independently supervised job/agent trees (not a security boundary)."""
    protected = set()
    import psutil
    script = str(Path(__file__).resolve())
    for child in children:
        try:
            if script in child.cmdline():
                protected.add(child.pid)
                protected.update(p.pid for p in child.children(recursive=True))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return protected


def kill_tree(process, preserve_managed=False):
    """Terminate descendants before root; process objects validate PID creation time."""
    import psutil
    try:
        children = process.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    protected = managed_exclusions(children) if preserve_managed else set()
    victims = [p for p in reversed(children) if p.pid not in protected] + [process]
    for p in victims:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(victims, timeout=3)


def supervise(dest: Path) -> None:
    import psutil
    request = json.loads((dest / 'request.json').read_text(encoding='utf-8'))
    worker = psutil.Process()
    status = {**request, 'state': 'running', 'started_at': time.time(),
              'worker_pid': os.getpid(), 'worker_created': worker.create_time()}
    atomic_json(dest / 'status.json', status)
    dropped = {'stdout': 0, 'stderr': 0}
    written = {'stdout': 0, 'stderr': 0}
    preserve = request.get('preserve_managed_children', False)
    proc = None
    tree = None
    threads = []
    try:
        if (dest / 'cancel').exists():
            status['state'] = 'cancelled'
            return
        env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
        proc = subprocess.Popen(shell_args(request['command']), cwd=request['cwd'], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0, start_new_session=os.name != 'nt',
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        tree = psutil.Process(proc.pid)
        status.update(pid=proc.pid, process_created=tree.create_time())
        atomic_json(dest / 'status.json', status)

        def drain(name, pipe):
            with pipe, (dest / (name + '.log')).open('ab', buffering=0) as output:
                while True:
                    block = pipe.read(65536)
                    if not block:
                        break
                    keep = max(0, min(len(block), request['log_limit_bytes'] - written[name]))
                    output.write(block[:keep])
                    written[name] += keep
                    dropped[name] += len(block) - keep

        for name in ('stdout', 'stderr'):
            thread = threading.Thread(target=drain, args=(name, getattr(proc, name)), daemon=True)
            thread.start()
            threads.append(thread)
        deadline = time.monotonic() + request['timeout']
        last_report = 0.0
        known_children = {}
        protected = set()
        while proc.poll() is None:
            try:
                children = tree.children(recursive=True)
                if preserve:
                    protected.update(managed_exclusions(children))
                for child in children:
                    known_children[(child.pid, child.create_time())] = child
            except psutil.NoSuchProcess:
                pass
            if (dest / 'cancel').exists():
                status['state'] = 'cancelled'
                kill_tree(tree, preserve)
                break
            if time.monotonic() >= deadline:
                status['state'] = 'timed_out'
                kill_tree(tree, preserve)
                break
            if time.monotonic() - last_report > 1:
                atomic_json(dest / 'status.json', {**status, 'dropped_bytes': dict(dropped)})
                last_report = time.monotonic()
            time.sleep(0.05)
        status['exit_code'] = proc.wait(timeout=5)
        # Background children must not outlive their managed job. This is best-
        # effort process supervision, not isolation against intentionally escaped children.
        for child in known_children.values():
            try:
                if child.pid not in protected and child.is_running():
                    kill_tree(child, preserve)
            except psutil.NoSuchProcess:
                pass
        for thread in threads:
            thread.join(timeout=3)
        status['logs_complete'] = not any(t.is_alive() for t in threads)
        if status['state'] == 'running':
            status['state'] = 'succeeded' if proc.returncode == 0 else 'failed'
    except BaseException as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        if tree is not None:
            kill_tree(tree, preserve)
    finally:
        status.update(finished_at=time.time(), dropped_bytes=dict(dropped))
        atomic_json(dest / 'status.json', status)


if __name__ == '__main__':
    supervise(Path(sys.argv[1]))
