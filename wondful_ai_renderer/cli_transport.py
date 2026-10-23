"""Bounded subprocess transport. No credential files, token extraction or shell interpolation.

On POSIX, the platform's script(1) gives agy a controlling terminal. A pipe alone
is insufficient on CLI versions that read OAuth codes from /dev/tty. We never
retry a generation automatically: a retry could spend another image request.
"""
from __future__ import annotations

import codecs
import os
import platform
import queue
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time

ANSI = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
AUTH_PROMPT = re.compile(
    r"authentication required|not logged in|please (?:log|sign)[ -]?in|"
    r"paste (?:the |your |an )?(?:authorization|authentication|verification) code|"
    r"enter (?:the |your |an )?(?:authorization|authentication|verification) code",
    re.I,
)
INTERACTIVE_AUTH_PROMPT = re.compile(
    r"paste (?:the |your |an )?(?:authorization|authentication|verification) code|"
    r"enter (?:the |your |an )?(?:authorization|authentication|verification) code",
    re.I,
)


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text or "").replace("\r\n", "\n")


def safe_diagnostic(text: str, secrets=()) -> str:
    text = strip_ansi(text)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[redacted]")
    text = re.sub(r"https?://[^\s<>\"']+", "[URL omitted]", text)
    text = re.sub(r"(?i)((?:access_token|refresh_token|authorization|code_verifier|code|token|state)[\"']?\s*[=:]\s*[\"']?)[^\s,;\"']+", r"\1[redacted]", text)
    text = re.sub(r"\beyJ[A-Za-z0-9_.-]{25,}\b", "[token omitted]", text)
    return text[-1600:]


def tty_command(cli: str, args: list[str], system: str | None = None):
    system = system or platform.system()
    script = shutil.which("script", path="/usr/bin:/bin:/usr/local/bin")
    if system not in {"Darwin", "Linux"} or not script:
        return [cli, *args], False
    if system == "Darwin":
        return [script, "-q", "/dev/null", cli, *args], True
    # Linux script(1) requires -c. Quote EVERY argument, never interpolate an
    # unquoted prompt, code, file path or executable into the command string.
    return [script, "-q", "-e", "-f", "-c", shlex.join([cli, *args]), "/dev/null"], True


_ACTIVE_PROCESSES: set[ManagedProcess] = set()
_PROCESSES_LOCK = threading.Lock()
_CANCEL_EVENT = threading.Event()


def is_cancellation_requested() -> bool:
    return _CANCEL_EVENT.is_set()


def request_cancellation() -> None:
    _CANCEL_EVENT.set()
    kill_active_processes()


def reset_cancellation() -> None:
    _CANCEL_EVENT.clear()


def kill_active_processes() -> None:
    with _PROCESSES_LOCK:
        procs = list(_ACTIVE_PROCESSES)
    for p in procs:
        try:
            p.stop()
        except Exception:
            pass


class ManagedProcess:
    """Own only this subprocess and its POSIX process group; drain output promptly."""
    def __init__(self, cli, args, *, cwd, env, use_tty=True):
        self.command, self.is_tty = tty_command(cli, args) if use_tty else ([cli, *args], False)
        self.proc = subprocess.Popen(
            self.command, cwd=cwd, env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0,
            start_new_session=(os.name != "nt"),
        )
        with _PROCESSES_LOCK:
            _ACTIVE_PROCESSES.add(self)
        self.output = queue.Queue()
        self._write_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True, name="WondfulCLIOutput")
        self._reader.start()

    def _read(self):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = os.read(self.proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                self.output.put(decoder.decode(chunk))
            last = decoder.decode(b"", final=True)
            if last:
                self.output.put(last)
        except (OSError, ValueError):
            pass
        finally:
            self.output.put(None)

    def read_available(self):
        parts = []
        while True:
            try:
                value = self.output.get_nowait()
            except queue.Empty:
                break
            if value is not None:
                parts.append(value)
        return "".join(parts)

    def write_line(self, text):
        # OAuth codes must not contain terminal controls or multiple commands.
        if not text or any(ord(c) < 32 or ord(c) == 127 for c in text):
            raise ValueError("Authorization code must be one non-empty line without control characters.")
        with self._write_lock:
            if self.proc.poll() is not None:
                raise RuntimeError("The original login process has already exited. Start a new login.")
            self.proc.stdin.write((text + "\n").encode("utf-8"))
            self.proc.stdin.flush()

    def stop(self):
        if self.proc.poll() is None:
            try:
                if os.name == "nt":
                    self.proc.terminate()
                else:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    if os.name == "nt":
                        self.proc.kill()
                    else:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    self.proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass

    def close(self):
        with _PROCESSES_LOCK:
            _ACTIVE_PROCESSES.discard(self)
        self.stop()
        self._reader.join(timeout=1)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except (OSError, AttributeError):
                pass


def run_headless(cli, args, *, cwd, env, timeout, input_text=None):
    """Run a print-mode request without giving the CLI an interactive TTY.

    The official headless contract uses cached credentials and exits with an
    auth error when none are available.  A PTY makes agy treat the request as a
    local interactive session and can launch a browser before Wondful can
    report the failure.  Interactive OAuth is owned by ``AuthSession`` instead.
    """
    if is_cancellation_requested():
        raise RuntimeError("任务已被用户主动取消。")
    process = ManagedProcess(cli, args, cwd=cwd, env=env, use_tty=False)
    start = time.monotonic()
    if input_text is not None:
        def feed():
            try:
                process.proc.stdin.write(input_text.encode("utf-8"))
                process.proc.stdin.close()
            except (OSError, ValueError):
                pass
        threading.Thread(target=feed, daemon=True, name="WondfulCLIInput").start()
    text = ""
    try:
        while True:
            if is_cancellation_requested():
                process.stop()
                raise RuntimeError("任务已被用户主动取消。")
            text += process.read_available()
            started_agent = bool(re.search(r'"event"\s*:\s*"(?:init|result)"|"status"\s*:\s*"SUCCESS"', text))
            # AGY 1.1.x can briefly log "not logged in" before its asynchronous
            # Keychain restore succeeds. Only stop for an actual interactive code
            # prompt; final auth failures are handled from the process result.
            if not started_agent and INTERACTIVE_AUTH_PROMPT.search(strip_ansi(text[-8000:])):
                process.stop()
                return subprocess.CompletedProcess([cli, *args], 1, "", "authentication required: use Wondful Connect; this task did not accept an OAuth code.")
            if timeout and time.monotonic() - start > timeout:
                raise subprocess.TimeoutExpired([cli], timeout)
            if process.proc.poll() is not None:
                process._reader.join(timeout=1)
                text += process.read_available()
                return subprocess.CompletedProcess([cli, *args], process.proc.returncode, strip_ansi(text), "")
            time.sleep(0.04)
    finally:
        process.close()

