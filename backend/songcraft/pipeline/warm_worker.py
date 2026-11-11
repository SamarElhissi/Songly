"""Long-lived ACE-Step process. One song at a time, JSON lines on stdin/stdout."""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import IO

logger = logging.getLogger("songcraft.pipeline")

_SHUTDOWN_GRACE_SECONDS = 5.0
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


class AceStepWorkerError(Exception):
    """Failure text that is safe to store on the song."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class WarmWorker:
    """Starts the ACE-Step process on the first song and reuses it until idle or shutdown."""

    def __init__(self, command: list[str], *, idle_seconds: float, song_timeout_seconds: float) -> None:
        self._command = command
        self._idle_seconds = idle_seconds
        self._song_timeout = song_timeout_seconds
        self._mu = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._lines: queue.Queue[bytes | None] | None = None
        self._job: int | None = None
        self._loaded = False
        self._busy = False
        self._closing = False
        self._epoch = 0
        self._idle_timer: threading.Timer | None = None

    def worker_pid(self) -> int | None:
        with self._mu:
            proc = self._proc
        if proc is None or proc.poll() is not None:
            return None
        return proc.pid

    def run_song(
        self,
        request: dict[str, object],
        *,
        on_loaded: Callable[[float], None] | None = None,
    ) -> tuple[dict[str, object], float | None, float]:
        """Return the result payload, load seconds when this song loaded, and generate seconds."""
        with self._mu:
            self._epoch += 1
            timer = self._idle_timer
            self._idle_timer = None
            self._busy = True
        if timer is not None:
            timer.cancel()
        try:
            return self._run_song(request, on_loaded)
        finally:
            with self._mu:
                self._busy = False
            if not self._closing:
                self._arm_idle()

    def shutdown(self) -> None:
        """Stop the process. A song that is running is killed so the API can exit."""
        self._closing = True
        with self._mu:
            self._epoch += 1
            timer = self._idle_timer
            self._idle_timer = None
            busy = self._busy
        if timer is not None:
            timer.cancel()
        self._stop(graceful=not busy)

    def _run_song(
        self,
        request: dict[str, object],
        on_loaded: Callable[[float], None] | None,
    ) -> tuple[dict[str, object], float | None, float]:
        deadline = time.monotonic() + self._song_timeout
        job_id = str(request.get("job_id") or "")
        try:
            loaded_now = self._ensure_ready(request, job_id, deadline, on_loaded)
            started = time.perf_counter()
            payload = self._rpc(
                {
                    "op": "generate",
                    "job_id": job_id,
                    "prompt": request["prompt"],
                    "duration_seconds": request["duration_seconds"],
                    "output_path": request["output_path"],
                },
                deadline,
                expect="result",
            )
            generating = time.perf_counter() - started
        except AceStepWorkerError:
            self._stop(graceful=False)
            raise
        if payload.get("fatal"):
            self._stop(graceful=False)
        if not payload.get("ok"):
            raise _worker_error(payload.get("error"))
        return payload, loaded_now, generating

    def _ensure_ready(
        self,
        request: dict[str, object],
        job_id: str,
        deadline: float,
        on_loaded: Callable[[float], None] | None,
    ) -> float | None:
        with self._mu:
            alive = self._proc is not None and self._proc.poll() is None
            if not alive:
                self._clear_locked()
                self._spawn_locked()
                self._loaded = False
            needs_load = not self._loaded
        if not needs_load:
            return None
        started = time.perf_counter()
        ready = self._rpc(
            {
                "op": "load",
                "job_id": job_id,
                "project_root": request["project_root"],
                "checkpoint_dir": request["checkpoint_dir"],
                "device": request["device"],
                "model": request["model"],
                "lm_model": request["lm_model"],
            },
            deadline,
            expect="ready",
        )
        if not ready.get("ok"):
            self._stop(graceful=False)
            raise _worker_error(ready.get("error"))
        loading = time.perf_counter() - started
        with self._mu:
            self._loaded = True
        if on_loaded is not None:
            on_loaded(round(loading, 3))
        logger.info(
            "stage_finished job_id=%s stage=loading_models duration_s=%.3f",
            job_id or "-",
            loading,
        )
        return loading

    def _rpc(self, payload: dict[str, object], deadline: float, *, expect: str) -> dict[str, object]:
        with self._mu:
            proc = self._proc
            lines = self._lines
        if proc is None or lines is None or proc.poll() is not None:
            raise AceStepWorkerError("ACE-Step worker stopped unexpectedly")
        try:
            _write(proc, payload)
            reply = _read(lines, deadline)
        except TimeoutError:
            self._stop(graceful=False)
            raise AceStepWorkerError("ACE-Step took too long and was stopped") from None
        except (OSError, AceStepWorkerError):
            self._stop(graceful=False)
            raise AceStepWorkerError("ACE-Step worker stopped unexpectedly") from None
        if reply.get("op") != expect:
            self._stop(graceful=False)
            raise AceStepWorkerError("ACE-Step worker stopped unexpectedly")
        sent = payload.get("job_id")
        if sent and reply.get("job_id") not in (None, sent):
            self._stop(graceful=False)
            raise AceStepWorkerError("ACE-Step worker stopped unexpectedly")
        return reply

    def _spawn_locked(self) -> None:
        flags = 0
        if sys.platform == "win32":
            # Own process group, and no extra console window.
            flags = _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
        proc = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=flags,
        )
        lines: queue.Queue[bytes | None] = queue.Queue()
        threading.Thread(target=_pump, args=(proc.stdout, lines), name="acestep-stdout", daemon=True).start()
        threading.Thread(
            target=_pump_stderr,
            args=(proc.stderr,),
            name="acestep-stderr",
            daemon=True,
        ).start()
        self._proc = proc
        self._lines = lines
        self._job = _assign_kill_on_close(proc.pid)

    def _stop(self, *, graceful: bool, epoch: int | None = None) -> None:
        with self._mu:
            if epoch is not None and (epoch != self._epoch or self._busy or self._closing):
                return
            proc = self._proc
            lines = self._lines
            job = self._job
            self._proc = None
            self._lines = None
            self._loaded = False
            self._job = None
        if proc is None:
            return
        if graceful and proc.poll() is None:
            try:
                _write(proc, {"op": "shutdown"})
                if lines is not None:
                    lines.get(timeout=_SHUTDOWN_GRACE_SECONDS)
            except (OSError, queue.Empty):
                pass
        if proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=_SHUTDOWN_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        _close_job(job)

    def _clear_locked(self) -> None:
        proc = self._proc
        if proc is not None:
            proc.wait(timeout=5)
        _close_job(self._job)
        self._job = None
        self._proc = None
        self._lines = None
        self._loaded = False

    def _arm_idle(self) -> None:
        if self._closing or self._idle_seconds <= 0:
            return
        if self.worker_pid() is None:
            return
        epoch = self._epoch
        timer = threading.Timer(self._idle_seconds, self._idle_fire, args=(epoch,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_fire(self, epoch: int) -> None:
        with self._mu:
            if epoch != self._epoch or self._busy or self._closing:
                return
        logger.info("stage_finished job_id=- stage=idle_shutdown duration_s=%.3f", self._idle_seconds)
        self._stop(graceful=True, epoch=epoch)


def _write(proc: subprocess.Popen[bytes], payload: dict[str, object]) -> None:
    stdin = proc.stdin
    if stdin is None:
        raise OSError("worker stdin is closed")
    stdin.write(json.dumps(payload).encode("utf-8") + b"\n")
    stdin.flush()


def _read(lines: queue.Queue[bytes | None], deadline: float) -> dict[str, object]:
    while True:
        timeout = deadline - time.monotonic()
        if timeout <= 0:
            raise TimeoutError
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError from None
        if not line:
            raise AceStepWorkerError("ACE-Step worker stopped unexpectedly")
        try:
            payload = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            # ACE-Step sometimes prints progress on stdout. Skip it and keep waiting.
            logger.info("worker_stdout %s", line.decode("utf-8", errors="replace").rstrip()[:300])
            continue
        if not isinstance(payload, dict):
            continue
        return payload


def _pump(stream: IO[bytes] | None, lines: queue.Queue[bytes | None]) -> None:
    try:
        if stream is not None:
            for line in iter(stream.readline, b""):
                lines.put(line)
    finally:
        lines.put(None)


def _pump_stderr(stream: IO[bytes] | None) -> None:
    if stream is None:
        return
    for line in iter(stream.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            logger.info("%s", text)


def _worker_error(error: object) -> AceStepWorkerError:
    text = " ".join(str(error or "").split())
    if not text:
        text = "ACE-Step worker stopped unexpectedly"
    if len(text) > 300:
        text = text[:300].rstrip() + "…"
    return AceStepWorkerError(text)


def _kernel32() -> ctypes.WinDLL:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    return kernel32


def _assign_kill_on_close(pid: int) -> int | None:
    """Windows Job Object. Closing the handle kills the worker, including after a hard API exit."""
    if sys.platform != "win32":
        return None
    import ctypes

    kernel32 = _kernel32()
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    info = _extended_limits()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(job)
        return None
    process = kernel32.OpenProcess(0x1F0FFF, False, pid)
    if not process or not kernel32.AssignProcessToJobObject(job, process):
        if process:
            kernel32.CloseHandle(process)
        kernel32.CloseHandle(job)
        logger.warning("windows_job_assign_failed pid=%s error=%s", pid, ctypes.get_last_error())
        return None
    kernel32.CloseHandle(process)
    return int(ctypes.cast(job, ctypes.c_void_p).value or 0)


def _close_job(handle: int | None) -> None:
    if not handle or sys.platform != "win32":
        return
    import ctypes

    _kernel32().CloseHandle(ctypes.c_void_p(handle))


def _extended_limits() -> ctypes.Structure:
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class Extended(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", Basic),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return Extended()
