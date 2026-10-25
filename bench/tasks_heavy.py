"""Container-backed task sources for the heavy sessions (bench/RUN_PLAN.md section 8): Harbor-format
tasks from Terminal-Bench 4.0 and SWE-bench Pro V2, each in its prebuilt image. The agent works from
the host: the task directory (/app in the container) is extracted from the image, bind-mounted back,
and reachable with the file tools; project commands run through `docker exec`. The check runs the
task's own verifier (tests/test.sh) the way Harbor does. Linux (WSL) with Docker only.

    Terminal-Bench 4.0 (BENCH_TB4, default ~/tb4/terminal-bench; `harbor datasets download
        terminal-bench/terminal-bench@4.0.0`): schema 1.3 tasks with a separate verifier container.
        The check collects the task's declared artifacts from the agent container, re-materialises
        them at the same absolute paths in a container of the verifier image (which already holds
        /tests), runs /tests/test.sh there and reads /logs/verifier/reward.txt|json.
    SWE-bench Pro V2 (BENCH_SWEPRO, default ~/swepro/v2/tasks; github.com/scaleapi/SWE-bench_Pro-os):
        schema 1.4 tasks verified in the agent container itself: tests/ is copied to /tests and
        /tests/test.sh resets the test files, runs the suite and writes the reward. The agent phase
        runs without network, as task.toml demands (bridge disconnected); the verifier reconnects it.

The selection (bench/external/heavy_selection.json, seed-disjoint) names the tasks and images.
Containers carry the label bench.session=<tag>; `cleanup(tag)` removes them and the extracted task
directories (root-owned after the container ran, so deleted through a container of the same image).
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SELECTION = ROOT / "bench" / "external" / "heavy_selection.json"
TB4_ROOT = Path(os.environ.get("BENCH_TB4") or (Path.home() / "tb4" / "terminal-bench"))
SWEPRO_ROOT = Path(os.environ.get("BENCH_SWEPRO") or (Path.home() / "swepro" / "v2" / "tasks"))
LABEL = "bench.session"
DOCKER_TIMEOUT = 600
APP = "/app"


# ------------------------------------------------------------------- docker helpers
def docker(*args: str, timeout: int = DOCKER_TIMEOUT, input: str | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout, input=input)
    except FileNotFoundError:  # no docker on this host (e.g. a dry run on Windows): every call fails softly
        return subprocess.CompletedProcess(["docker", *args], 127, "", "docker: command not found")


def docker_available() -> bool:
    try:
        return docker("info", timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def running(name: str) -> bool:
    r = docker("inspect", "-f", "{{.State.Running}}", name, timeout=60)
    return r.returncode == 0 and r.stdout.strip() == "true"


_HOSTDIRS: dict[str, list[tuple[Path, str]]] = {}  # tag -> [(extracted task dir, image)]


def cleanup(tag: str) -> int:
    """Remove every container of this session (agent and verifier containers) and the task directories
    extracted for it; returns how many containers were removed."""
    r = docker("ps", "-aq", "--filter", f"label={LABEL}={tag}", timeout=60)
    ids = r.stdout.split()
    if ids:
        docker("rm", "-f", *ids, timeout=300)
    for hostdir, image in _HOSTDIRS.pop(tag, []):
        if hostdir.exists():
            docker("run", "--rm", "-v", f"{hostdir.parent}:/w", image, "rm", "-rf", f"/w/{hostdir.name}", timeout=600)
            if hostdir.exists():
                shutil.rmtree(hostdir, ignore_errors=True)
    return len(ids)


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", s)[:60]


def exec_logged(name: str, cmd: list[str], log: Path, timeout: int, workdir: str | None = None) -> tuple[int, bool]:
    """docker exec with the combined output written to `log`; returns (returncode, timed_out)."""
    args = ["exec"] + (["-w", workdir] if workdir else []) + [name, *cmd]
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        try:
            p = subprocess.run(["docker", *args], stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            return p.returncode, False
        except subprocess.TimeoutExpired:
            f.write(f"\n\nTimeout error: {timeout} seconds exceeded.\n")
            return -1, True


def read_reward(name: str) -> tuple[float | None, str]:
    """The reward a verifier wrote in container `name` (reward.json preferred, then reward.txt)."""
    r = docker("exec", name, "sh", "-c", "cat /logs/verifier/reward.json 2>/dev/null || cat /logs/verifier/reward.txt 2>/dev/null",
               timeout=60)
    txt = r.stdout.strip()
    if not txt:
        return None, "no reward written"
    try:
        val = json.loads(txt)
    except ValueError:
        return None, f"unreadable reward: {txt[:80]!r}"
    if isinstance(val, dict):  # labelled rewards: the mean of the numeric entries
        nums = [float(v) for v in val.values() if isinstance(v, (int, float))]
        val = sum(nums) / len(nums) if nums else 0.0
    try:
        return float(val), ""
    except (TypeError, ValueError):
        return None, f"unreadable reward: {txt[:80]!r}"


# ------------------------------------------------------------------- selection
def selection() -> dict:
    return json.loads(SELECTION.read_text(encoding="utf-8"))


def tb_selection(seed: int) -> list[str]:
    return [t["task"] for t in selection()["tb4"][str(seed)]]


def swe_selection(seed: int) -> list[str]:
    return [t["instance_id"] for t in selection()["swepro"][str(seed)]]


# ------------------------------------------------------------------- Harbor task
class HarborTask:
    """One Harbor-format task (task.toml, instruction.md, tests/) in its prebuilt image."""

    def __init__(self, source: str, task_dir: Path, tag: str, workdir: Path, logdir: Path):
        self.source, self.dir, self.tag = source, task_dir, tag
        self.task = task_dir.name
        if not (task_dir / "task.toml").exists():
            raise SystemExit(f"{source} task missing: {task_dir}")
        self.toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
        env = self.toml.get("environment", {})
        self.image = env["docker_image"]
        self.cpus = str(env.get("cpus") or 1)
        self.memory_mb = int(env.get("memory_mb") or 4096)
        self.no_network = self.toml.get("agent", {}).get("network_mode") == "no-network"
        ver = self.toml.get("verifier", {})
        self.verifier_timeout = int(ver.get("timeout_sec") or 900)
        self.separate = ver.get("environment_mode") == "separate"
        venv = ver.get("environment", {})
        self.verifier_image = venv.get("docker_image") if self.separate else None
        self.verifier_cpus = str(venv.get("cpus") or self.cpus)
        self.verifier_memory_mb = int(venv.get("memory_mb") or self.memory_mb)
        arts = [a if isinstance(a, str) else a.get("source") for a in self.toml.get("artifacts", [])
                if isinstance(a, str) or not a.get("service") or a.get("service") == "main"]
        self.artifacts = [a for a in arts if a]
        self.meta = self.toml.get("metadata", {})
        short = safe_name(self.task if source == "terminal-bench" else self.task.split("instance_")[-1][:40])
        kind = "tb" if source == "terminal-bench" else "swe"
        # container names: a short hash of the trial tag keeps names unique across arms and seeds that
        # run side by side (the full tag would be truncated and the arm suffix lost)
        h = hashlib.sha1(tag.encode("utf-8")).hexdigest()[:8]
        self.name = safe_name(f"{kind}-{h}-{short}")
        self.verifier_name = safe_name(f"vf-{h}-{short}")
        self.hostdir = workdir / "tasks" / short
        self.logdir = logdir
        self.logdir.mkdir(parents=True, exist_ok=True)

    # -- lifecycle
    def _extract_app(self) -> None:
        """Copy /app out of the image so the agent's file tools can reach the task files."""
        self.hostdir.parent.mkdir(parents=True, exist_ok=True)
        cid = docker("create", self.image, timeout=300).stdout.strip()
        if not cid:
            raise RuntimeError(f"docker create failed for {self.image}")
        try:
            r = docker("cp", f"{cid}:{APP}", str(self.hostdir), timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(f"docker cp {APP} failed: {r.stderr[-300:]}")
        finally:
            docker("rm", cid, timeout=120)

    def start(self) -> None:
        if running(self.name):
            return
        docker("rm", "-f", self.name, timeout=120)
        if not self.hostdir.exists():
            self._extract_app()
        r = docker("run", "-d", "--name", self.name, "--label", f"{LABEL}={self.tag}", "--cpus", self.cpus,
                   "--memory", f"{self.memory_mb}m", "-v", f"{self.hostdir}:{APP}", "-w", APP,
                   self.image, "sh", "-c", "sleep infinity", timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"docker run failed for {self.name}: {r.stderr[-300:]}")
        _HOSTDIRS.setdefault(self.tag, []).append((self.hostdir, self.image))
        # the bind-mounted /app belongs to the host user while the container runs as root: without this
        # git refuses the repository ("dubious ownership") and test runners that call git abort
        docker("exec", self.name, "sh", "-c", "git config --global --add safe.directory '*' 2>/dev/null || true", timeout=60)
        if self.no_network:  # agent phase offline (a container created with --network none could not be reconnected)
            docker("network", "disconnect", "bridge", self.name, timeout=60)

    def prompt(self, k: int) -> str:
        instr = (self.dir / "instruction.md").read_text(encoding="utf-8")
        instr = re.sub(r"<!--.*?-->\s*", "", instr, flags=re.S).strip()  # drop the canary comment
        net = "The container has no network access. " if self.no_network else ""
        return (f"Q{k}: The next task has to be carried out in the Linux container `{self.name}`. Its working "
                f"directory `{APP}` is mounted from `{self.hostdir}` on this machine, so read and edit the task "
                f"files there with your file tools; run every project command inside the container, for example "
                f"`docker exec -w {APP} {self.name} bash -lc '<command>'`. {net}Do not touch the container itself "
                f"(no docker run/rm/stop) and keep your work inside `{APP}` unless the task says otherwise. Files "
                f"that commands create inside the container belong to root; change those through docker exec.\n\n"
                f"Task:\n{instr}\n\nReply with a short summary when the task is done.")

    # -- verification
    def _verify_in_place(self, det: dict) -> tuple[float, dict]:
        if self.no_network:  # the verifier phase may need the network (task.toml: verifier.network_mode = public)
            r = docker("network", "connect", "bridge", self.name, timeout=60)
            det["network_reconnected"] = r.returncode == 0 or "already exists" in r.stderr
        docker("exec", self.name, "rm", "-rf", "/tests", "/logs/verifier", timeout=60)
        docker("cp", str(self.dir / "tests"), f"{self.name}:/tests", timeout=300)
        docker("exec", self.name, "mkdir", "-p", "/logs/verifier", timeout=60)
        log = self.logdir / f"verify_{safe_name(self.task)}.log"
        rc, timed_out = exec_logged(self.name, ["bash", "/tests/test.sh"], log, self.verifier_timeout)
        det.update(returncode=rc, timed_out=timed_out, log=str(log))
        return self._finish(self.name, det)

    def _verify_separate(self, det: dict) -> tuple[float, dict]:
        art = self.logdir / f"artifacts_{safe_name(self.task)}"
        shutil.rmtree(art, ignore_errors=True)
        art.mkdir(parents=True)
        collected = []
        for src in self.artifacts:  # mirror each declared path under the host artifacts dir
            dst = art / src.strip("/")
            dst.parent.mkdir(parents=True, exist_ok=True)
            r = docker("cp", f"{self.name}:{src.rstrip('/') or '/'}", str(dst), timeout=600)
            if r.returncode == 0:
                collected.append(src)
        det["artifacts"] = collected
        docker("rm", "-f", self.verifier_name, timeout=120)
        r = docker("run", "-d", "--name", self.verifier_name, "--label", f"{LABEL}={self.tag}", "--cpus", self.verifier_cpus,
                   "--memory", f"{self.verifier_memory_mb}m", self.verifier_image, "sh", "-c", "sleep infinity", timeout=300)
        if r.returncode != 0:
            det["error"] = f"verifier container failed: {r.stderr[-200:]}"
            return 0.0, det
        try:
            for src in collected:
                target = src.rstrip("/")
                docker("exec", self.verifier_name, "mkdir", "-p", str(Path(target).parent), timeout=60)
                docker("exec", self.verifier_name, "rm", "-rf", target, timeout=60)
                docker("cp", str(art / src.strip("/")), f"{self.verifier_name}:{target}", timeout=600)
            if docker("exec", self.verifier_name, "test", "-f", "/tests/test.sh", timeout=60).returncode != 0:
                docker("cp", str(self.dir / "tests"), f"{self.verifier_name}:/tests", timeout=300)
            docker("exec", self.verifier_name, "mkdir", "-p", "/logs/verifier", timeout=60)
            log = self.logdir / f"verify_{safe_name(self.task)}.log"
            rc, timed_out = exec_logged(self.verifier_name, ["bash", "/tests/test.sh"], log, self.verifier_timeout)
            det.update(returncode=rc, timed_out=timed_out, log=str(log))
            return self._finish(self.verifier_name, det)
        finally:
            docker("rm", "-f", self.verifier_name, timeout=120)

    def _finish(self, name: str, det: dict) -> tuple[float, dict]:
        score, err = read_reward(name)
        if score is None:
            det["error"] = err
            return 0.0, det
        det["reward"] = score
        return (1.0 if score >= 1.0 else score), det

    def check(self, final: str) -> tuple[float, dict]:
        det = {"source": self.source, "task": self.task}
        if not running(self.name):
            det["error"] = "container not running"
            return 0.0, det
        patch = docker("exec", "-w", APP, self.name, "sh", "-c", "git diff 2>/dev/null | head -c 2000000", timeout=120)
        (self.logdir / f"patch_{safe_name(self.task)}.diff").write_text(patch.stdout, encoding="utf-8")
        return self._verify_separate(det) if self.separate else self._verify_in_place(det)


# ------------------------------------------------------------------- seqdep filler factories
def _factories(source: str, root: Path, names: list[str], tag: str, workdir: Path, logdir: Path, meta_fn) -> list:
    """`f(k) -> seqdep.Turn` per task; the Turn's meta carries the start hook under `_start`
    (removed and wired to the turn's pre hook by the session builder)."""
    from bench import seqdep
    out = []
    for n in names:
        task = HarborTask(source, root / n, tag, workdir, logdir)

        def make(k: int, task=task):
            meta = {"source": source, "task": task.task, "role": "filler", "_start": task.start, **meta_fn(task)}
            return seqdep.Turn(k, "question", task.prompt(k), meta=meta, check=task.check)
        out.append(make)
    return out


def swe_factories(seed: int, n: int, tag: str, workdir: Path, logdir: Path) -> list:
    info = {t["instance_id"]: t for t in selection()["swepro"][str(seed)]}
    return _factories("swebench-pro", SWEPRO_ROOT, swe_selection(seed)[:n], tag, workdir, logdir,
                      lambda t: {"instance_id": t.task, "repo": info[t.task]["repo"], "language": info[t.task]["language"],
                                 "difficulty": t.meta.get("difficulty")})


def tb_factories(seed: int, n: int, tag: str, workdir: Path, logdir: Path) -> list:
    return _factories("terminal-bench", TB4_ROOT, tb_selection(seed)[:n], tag, workdir, logdir,
                      lambda t: {"category": t.meta.get("category"), "expert_hours": t.meta.get("expert_time_estimate_hours")})


def register_cleanup(tag: str) -> None:
    atexit.register(cleanup, tag)
